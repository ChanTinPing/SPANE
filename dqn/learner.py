import torch.optim as optim
import torch as th
import torch.nn as nn
import numpy as np

class QLearner:
    def __init__(self, agent, args):
        self.args      = args
        self.agent     = agent
        weight_decay   = getattr(args, 'weight_decay', 0)  # Set to 0 if args.weight_decay is not provided
        self.optimizer = optim.Adam(self.agent.online_net.parameters(), lr=args.lr, weight_decay=weight_decay)
        self.loss_fn   = nn.MSELoss()
        self.learn_cnt = 0

        # Learning rate scheduler
        if args.scheduler == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.99)  # Decay learning rate every 1000 steps
        elif args.schduler == 'cos':
            self.scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=3000)  # Cosine annealing

    @th.no_grad()
    def update_target_network(self):
        self.agent.update_target_network()
    
    def to_tensor(self, array):
        if isinstance(array, np.ndarray):
            return th.tensor(array, dtype=th.float32)
        elif isinstance(array, th.Tensor):
            return array.clone().detach().float()
        else:
            raise ValueError("Unsupported data type")

    def train(self, memory, batch_size):
        if memory.__len__() < batch_size:
            return None

        # Sample from experience replay buffer
        transitions = memory.sample(batch_size)
        batch = list(zip(*transitions))

        # Ensure all data are PyTorch tensors and convert to float32
        obs         = th.stack([self.to_tensor(b) for b in batch[0]])
        feat        = th.stack([self.to_tensor(b) for b in batch[1]])
        actions     = th.tensor(batch[2], dtype=th.int64)  # actions usually are indices, so int64
        rewards     = th.tensor(batch[3], dtype=th.float32)
        next_obs    = th.stack([self.to_tensor(b) for b in batch[4]])
        next_feat   = th.stack([self.to_tensor(b) for b in batch[5]])
        dones       = th.tensor(np.array(batch[6], dtype=int)).view(-1).float()  # Convert to float32
        actual_ns   = th.tensor(batch[7], dtype=th.float32)  # actual steps

        # Combine into states and next_states
        states      = [obs, feat]
        next_states = [next_obs, next_feat]

        # Compute Q-value of current state
        q_values = self.agent.online_net(states)
        # Compute Q-value of next state (from online network)
        next_q_values = self.agent.online_net(next_states)
        # Compute Q-value of next state (from target network)
        next_q_state_values = self.agent.target_net(next_states).detach()

        # Select Q-value of current action
        q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        # Select Q-value of next state using target network
        next_q_value = next_q_state_values.gather(1, th.argmax(next_q_values, dim=1, keepdim=True)).squeeze(1)
        # Compute expected Q-value (multi-step return)
        expected_q_value = rewards + (self.args.gamma ** actual_ns) * next_q_value * (1 - dones)

        # Backpropagation
        loss = self.loss_fn(q_value, expected_q_value)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.learn_cnt += 1
        if self.learn_cnt % self.args.target_update_interval == 0:
            self.update_target_network()
        
        self.scheduler.step()

        return loss.item()
