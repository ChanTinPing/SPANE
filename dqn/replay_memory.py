import random
import torch as th
import numpy as np
from collections import deque
import pickle
import os

'''
n_step_buffer 举例:
deque([
    (obs1, feat1, action1, reward1, next_obs1, next_feat1, done1),
    (obs2, feat2, action2, reward2, next_obs2, next_feat2, done2),
    (obs3, feat3, action3, reward3, next_obs3, next_feat3, done3)
], maxlen=3)

memory单元素示例:
(obs, feat, action, reward, next_obs, next_feat, done, actual_n)

* obs, feat type 可能是ndarray可能是tensor, 目前没问题
'''

class ReplayMemory:
    def __init__(self, capacity, n_step, gamma):
        self.capacity      = capacity                # 经验回放缓冲区的容量
        self.memory        = np.zeros(capacity, dtype=object)  # 用于存储处理后的多步经验的主缓冲区
        self.position      = 0                       # 当前插入位置，用于循环覆盖（环形缓冲区）
        self.n_step        = n_step                  # n步长度，用于计算多步返回
        self.gamma         = gamma                   # 折扣因子，用于计算累积奖励
        self.n_step_buffer = deque(maxlen=n_step)    # 用于临时存储最近n步的经验

    def push(self, *args):
        # 检查是否传入了多个经验
        if len(args[0].shape) == 4:  # (batch size, N, 2, 2)
            for experience in zip(*args):
                self._push_single(*experience)
        else:
            self._push_single(*args)
    
    def _push_single(self, *args):
        self.n_step_buffer.append(args)              # 将新经验添加到n_step_buffer
        
        # 如果n_step_buffer满了，将buffer的内容处理，并存储在memory
        if len(self.n_step_buffer) == self.n_step:
            obs, feat, action, reward, next_obs, next_feat, done, actual_n = self._get_n_step_info()
            self.memory[self.position] = (obs, feat, action, reward, next_obs, next_feat, done, actual_n)
            self.position = (self.position + 1) % self.capacity  # 环形缓冲区位置更新
            self.n_step_buffer.popleft()             # 从n_step_buffer中移除第一个经验

        # 如果到达终止状态，将剩余的经验处理并存储到memory中
        if args[-1]:                                 # 检查是否为终止状态
            self.cut()
    
    def cut(self):
        while len(self.n_step_buffer) > 0:
            n = len(self.n_step_buffer)
            obs, feat, action, reward, next_obs, next_feat, done, actual_n = self._get_n_step_info(n)
            if len(self.memory) < self.capacity:
                self.memory.append(None)         # 如果主缓冲区未满，则添加一个新位置
            self.memory[self.position] = (obs, feat, action, reward, next_obs, next_feat, done, actual_n)
            self.position = (self.position + 1) % self.capacity  # 环形缓冲区位置更新
            self.n_step_buffer.popleft()         # 从n_step_buffer中移除已处理的经验

    def _get_n_step_info(self, n=None):     
        if n is None:
            n = self.n_step

        # 获取n步缓冲区中第一个经验的obs, feat, 和动作
        obs, feat = self.n_step_buffer[0][:2]
        action = self.n_step_buffer[0][2]

        # 获取n步缓冲区中最后一个经验的下一状态和完成标志
        next_obs, next_feat = self.n_step_buffer[n-1][4:6]
        done = self.n_step_buffer[n-1][6]

        # 计算n步累积奖励
        reward = sum([self.gamma ** i * self.n_step_buffer[i][3] for i in range(n)])

        return obs, feat, action, reward, next_obs, next_feat, done, th.tensor([n])

    def sample(self, batch_size):
        return self.memory[np.random.choice(len(self), batch_size, replace=False)]
    
    def merge(self, other_memory, ratio):
        if not isinstance(other_memory, ReplayMemory):
            raise ValueError("other_memory must be an instance of ReplayMemory")
        num_samples = int(len(other_memory) * ratio)
        sampled_experiences = random.sample(other_memory.memory, num_samples)
        if sampled_experiences:
            self.push(*zip(*sampled_experiences))

    def __len__(self):
        return min(self.position, self.capacity)

    def save_to_file(self, filepath):
        if not os.path.exists(filepath):
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            open(filepath, 'wb').close()

        with open(filepath, 'wb') as f:
            pickle.dump(self.memory, f)

    def load_from_file(self, filepath):
        with open(filepath, 'rb') as f:
            self.memory = pickle.load(f)
            self.position = len(self.memory) % self.capacity
