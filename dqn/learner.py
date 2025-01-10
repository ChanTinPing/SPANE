import torch.optim as optim
import torch as th
import torch.nn as nn
import numpy as np

class QLearner:
    def __init__(self, agent, args):
        self.args      = args
        self.agent     = agent
        weight_decay   = getattr(args, 'weight_decay', 0)  # 如果没有args.weight_decay则设成0
        self.optimizer = optim.Adam(self.agent.online_net.parameters(), lr=args.lr, weight_decay=weight_decay)
        self.loss_fn   = nn.MSELoss()
        self.learn_cnt = 0

        # 学习率调度器
        if args.scheduler == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.99)  # 每1000步衰减学习率
        elif args.schduler == 'cos':
            self.scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=3000)  # 余弦退火

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

        # 从经验回放缓冲区中采样
        transitions = memory.sample(batch_size)
        batch = list(zip(*transitions))

        # 确保所有数据都是PyTorch张量，并转换为float32类型
        obs         = th.stack([self.to_tensor(b) for b in batch[0]])
        feat        = th.stack([self.to_tensor(b) for b in batch[1]])
        actions     = th.tensor(batch[2], dtype=th.int64)  # actions usually are indices, so int64
        rewards     = th.tensor(batch[3], dtype=th.float32)
        next_obs    = th.stack([self.to_tensor(b) for b in batch[4]])
        next_feat   = th.stack([self.to_tensor(b) for b in batch[5]])
        dones       = th.tensor(np.array(batch[6], dtype=int)).view(-1).float()  # 转换为float32
        actual_ns   = th.tensor(batch[7], dtype=th.float32)  # 实际步数

        # 组合 states 和 next_states
        states      = [obs, feat]
        next_states = [next_obs, next_feat]

        # 计算当前状态的Q值
        q_values = self.agent.online_net(states)
        # 计算下一个状态的Q值（来自在线网络）
        next_q_values = self.agent.online_net(next_states)
        # 计算下一个状态的Q值（来自目标网络）
        next_q_state_values = self.agent.target_net(next_states).detach()

        # 选择当前动作的Q值
        q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        # 使用目标网络选择下一个状态的Q值
        next_q_value = next_q_state_values.gather(1, th.argmax(next_q_values, dim=1, keepdim=True)).squeeze(1)
        # 计算期望的Q值（多步返回）
        expected_q_value = rewards + (self.args.gamma ** actual_ns) * next_q_value * (1 - dones)

        # 计算损失
        loss = self.loss_fn(q_value, expected_q_value)
        # 优化步骤
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.learn_cnt += 1
        if self.learn_cnt % self.args.target_update_interval == 0:
            self.update_target_network()
        
        self.scheduler.step()

        return loss.item()
