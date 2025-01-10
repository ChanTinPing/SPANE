"""
DQN 模块实现了基于深度Q网络 (DQN) 和双重DQN (Double DQN) 的强化学习算法。主要包含以下几个模块：

1. DoubleDQNAgent (DQNagent.py) 包含dueling与double
2. QLearner (learner.py)
3. ReplayMemory (replay_memory.py)
"""

"""
DoubleDQNAgent (DQNagent.py)

创建:
- DoubleDQNAgent(args)
    - args: 包含训练过程中的超参数
"""

"""
QLearner (learner.py)
提供了一个训练函数和更新目标网络的功能。

创建：
- QLearner(agent, args)
    - agent: DoubleDQNAgent 对象
    - args: 包含训练过程中的超参数

使用：
- learner.train(memory, batch_size)
    - 从 memory 里抽取 batch 训练
    - memory: ReplayMemory 对象
    - batch_size: 批处理大小
"""

"""
ReplayMemory (replay_memory.py)
- 实现了一个用于存储和采样经验的回放缓冲区。
- 支持n步返回用于计算累积奖励。

创建：
- ReplayMemory(capacity, n_step, gamma)
    - capacity: 缓冲区的容量
    - n_step: n步长度, 用于计算多步返回
    - gamma: 折扣因子

使用：
- memory.push(obs, feat, action, reward, next_obs, next_feat, done)
    - obs: 形状为 (N, 2, 2)
    - feat: 形状为 (2, 2, 2)
    - action: 动作 
    - reward: 奖励
    - next_obs: 形状为 (N, 2, 2)
    - next_feat: 形状为 (2, 2, 2)
    - done: 是否完成

- memory.sample(batch_size)
    - batch_size: 批处理大小
    - 输出: transitions, 每个元素包含 (obs, feat, action, reward, next_obs, next_feat, done, actual_n)
        - 这里的 next_obs 是 actual_n 步后的了

数据结构：
- n_step_buffer: 一个临时缓冲区, 用于存储最近的n步经验
- memory: 主缓冲区，用于存储处理后的多步经验
- 每个经验包含 (obs, feat, action, reward, next_obs, next_feat, done, actual_n)
"""
