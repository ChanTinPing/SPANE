import torch.nn as nn
import torch.nn.init as init
import torch as th
import numpy as np
import os

'''
obs: tensor([
[[40., 90.],[40., 90.]],
[[40., 90.],[40., 90.]],
[[40., 90.],[40., 90.]],
[[40., 90.],[40., 90.]],
[[40., 90.],[40., 90.]]
]) 

feat: tensor(
[   [[2., 4.],[0., 0.]],
    [[0., 0.],[2., 4.]] ]
)
'''

def init_weights(m):
    if isinstance(m, nn.Linear):
        init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            init.constant_(m.bias, 0)

def check_obs(obs):  # Ensure the type is tensor and check the shape
    if not isinstance(obs, th.Tensor): obs = th.tensor(obs, dtype=th.float32)
    if   len(obs.shape) == 3:  return obs.unsqueeze(0)
    elif len(obs.shape) == 4:  return obs       # (batch size, ., ., .)
    else                    :  raise ValueError("Shape Incorrect")

def check_feat(feat):  # Ensure the type is tensor and check the shape
    if not isinstance(feat, th.Tensor): feat = th.tensor(feat, dtype=th.float32)
    if   len(feat.shape) == 1:  return feat.unsqueeze(0)
    elif len(feat.shape) == 2:  return feat       # (batch size, ...)
    else                     :  raise ValueError("Shape Incorrect")

def normalize_data(obs, feat, cpu, mem):
    obs[:, :, :, 0]  = obs[:, :, :, 0] / cpu
    obs[:, :, :, 1]  = obs[:, :, :, 1] / mem
    feat[:, 0] = feat[:, 0] / cpu
    feat[:, 1] = feat[:, 1] / mem
    return obs, feat

def get_layers(relu_no, initial_width, width, final_width):
    layers = nn.ModuleList([nn.Linear(initial_width, width)])
    for _ in range(relu_no-1):
        layers.extend([
            nn.ReLU(),
            nn.Linear(width, width)
        ])
    layers.extend([
        nn.ReLU(),
        nn.Linear(width, final_width)
    ])
    return layers


class DQNAgent(nn.Module):
    def __init__(self, N, cpu, mem, nn_widths):
        super(DQNAgent, self).__init__()
        self.cpu = cpu
        self.mem = mem
      
        # flatten (to 1D) layer
        self.flat = nn.Flatten()

        # hidden layers
        nn_widths.append(N * 2)
        input_size = (N * 2 * 2) + 3  

        self.fc = nn.ModuleList([
            nn.Linear(input_size, nn_widths[0]),
            nn.ReLU()
        ])
        for i in range(1, len(nn_widths)):
            self.fc.extend([
                nn.Linear(nn_widths[i-1], nn_widths[i]),
                nn.ReLU()
            ])

        # Value, Adv layers
        self.value = nn.Linear(nn_widths[-1], 1)
        self.adv = nn.Linear(nn_widths[-1], N * 2)

        self.apply(init_weights) # initialize weights

    def forward(self, state):
        obs, feat = state[0], state[1]
        obs  = check_obs(obs)   # Ensure obs is a tensor with shape (batch_size, N, 2, 2) (default batch_size = 1)
        feat = check_feat(feat)  # Ensure feat is a tensor with shape (batch_size, 2, 2, 2) (default batch_size = 1)
        obs, feat = normalize_data(obs, feat, self.cpu, self.mem)

        obs, feat = self.flat(obs), self.flat(feat)
        x = th.cat([obs, feat], dim=-1)
        for layer in self.fc:
            x = layer(x)
        adv = self.adv(x)  
        value = self.value(x)  
        q_values = value + adv - th.mean(adv, dim=1, keepdim=True)  # Dueling DQN detail
        return q_values

class DQNAgent_sym(nn.Module): 
    def __init__(self, N, cpu, mem, nn_widths, aggs, easy):
        super(DQNAgent_sym, self).__init__()
        self.cpu = cpu
        self.mem = mem
        self.N   = N
        self.aggs = aggs  # aggregation (mean)
        self.easy = easy
        feat_dim = 3

        # embedding layers
        width_server, relu_no = nn_widths['server']
        self.server_embedding = get_layers(relu_no, 4, width_server, width_server)
        width_cluster = width_server * len(aggs)
        # Value layers
        width_value, relu_no = nn_widths['value']
        self.value_layers     = get_layers(relu_no, width_cluster + feat_dim, width_value, 1)
        # Advantages layers
        width_adv, relu_no = nn_widths['adv']
        if easy:
            self.adv_layers = get_layers(relu_no, width_server, width_adv, 2)
        else:
            self.adv_layers = get_layers(relu_no, width_server + width_cluster + feat_dim, width_adv, 2)

        self.apply(init_weights)  # initialize weights

    def forward(self, state, is_easy_valid=False):
        # Data process
        obs, feat = state[0], state[1]
        obs, feat = check_obs(obs), check_feat(feat)   # Ensure obs is a tensor with shape (batch_size, N, 2, 2) (default batch_size = 1), and feat has shape (batch_size, 2, 2, 2).
        obs, feat = normalize_data(obs, feat, self.cpu, self.mem)

        if not is_easy_valid:
            # calc server
            server_embeddings = obs.view(-1, self.N, 4)
            for layer in self.server_embedding:
                server_embeddings = layer(server_embeddings)
            # calc cluster embedding
            agg_dict = {
                'mean': th.mean,
                'max': lambda x, dim: th.max(x, dim)[0],
                'std': th.std
            }
            cluster_embedding = []
            for agg in self.aggs:
                cluster_embedding.append(agg_dict[agg](server_embeddings, dim=1))
            cluster_embedding = th.cat(cluster_embedding, dim=1)

            # calc value
            value = th.cat([cluster_embedding, feat], dim=1)
            for layer in self.value_layers:
                value = layer(value)

            # calc advantages
            if self.easy:
                advs = server_embeddings
            else:
                cluster_embedding_rpt = cluster_embedding.unsqueeze(1).repeat(1, self.N, 1)
                feat_rpt = feat.unsqueeze(1).repeat(1, self.N, 1)
                advs = th.cat([server_embeddings, cluster_embedding_rpt, feat_rpt], dim=2)
            for layer in self.adv_layers:
                advs = layer(advs)
            advs = advs.view(-1, 2 * self.N)

            q_values = value + advs - th.mean(advs, dim=1, keepdim=True)
            return q_values

        else:
            advs = obs.view(-1, self.N, 4)  
            for layer in self.adv_layers:
                advs = layer(advs)
            advs = advs.view(-1, 2 * self.N)   
            return advs
        
class DoubleDQNAgent:  # Split the DQN network into two parts to improve stability. See Double DQN for details.
    def __init__(self, args, mode='basic'):
        if mode == 'basic':
            self.online_net = DQNAgent(args.N, args.cpu, args.mem, args.nn_width.copy())
            self.target_net = DQNAgent(args.N, args.cpu, args.mem, args.nn_width.copy())
        elif mode == 'sym':
            self.online_net = DQNAgent_sym(args.N, args.cpu, args.mem, args.nn_width.copy(), args.cluster_agg, easy=0)
            self.target_net = DQNAgent_sym(args.N, args.cpu, args.mem, args.nn_width.copy(), args.cluster_agg, easy=0)
        elif mode == 'sym_easy':
            self.online_net = DQNAgent_sym(args.N, args.cpu, args.mem, args.nn_width.copy(), args.cluster_agg, easy=1)
            self.target_net = DQNAgent_sym(args.N, args.cpu, args.mem, args.nn_width.copy(), args.cluster_agg, easy=1)
        else:
            raise ValueError("DoubleDQNAgent mode incorrect")
        self.update_target_network()

    def update_target_network(self):
        self.target_net.load_state_dict(self.online_net.state_dict())
    
    def select_action(self, state, epsilon, is_easy_valid=False):  # state = {'obs':..., 'feat':..., 'avail':...}
        avail = state['avail']

        if len(avail.shape) == 1:  # Single env case
            if np.random.rand() < epsilon:
                valid_action = np.random.choice(np.where(avail == 1)[0])
                return valid_action.item()
            else:
                with th.no_grad():
                    q_values = self.online_net([state['obs'], state['feat']])[0]
                    q_values[avail == 0] = float('-inf')  # Set the Q-values of unavailable actions to negative infinity
                    return th.argmax(q_values).cpu().item()

        elif len(avail.shape) == 2: # 多个env
            if np.random.rand() < epsilon:
                valid_actions = [np.random.choice(np.where(avail[i] == 1)[0]) for i in range(state[0].shape[0])]
                return np.array(valid_actions)
            else:
                with th.no_grad():
                    q_values = self.online_net([state['obs'], state['feat']])
                    q_values[avail == 0] = float('-inf')  # Set the Q-values of unavailable actions to negative infinity
                    return th.argmax(q_values, dim=1).cpu().numpy()
                    
        else:
            raise ValueError("avail shape incorrect!")
           
    def save(self, filepath):
        dir_path = os.path.dirname(filepath)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        th.save(self.online_net.state_dict(), filepath)

    def load(self, filepath):
        self.online_net.load_state_dict(th.load(filepath, weights_only=True))
        self.update_target_network()
