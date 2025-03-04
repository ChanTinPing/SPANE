"""
The DQN module implements reinforcement learning algorithms based on Deep Q-Networks (DQN) and Double DQN. It mainly consists of the following components:

1. DoubleDQNAgent (DQNagent.py) includes dueling and double
2. QLearner (learner.py)
3. ReplayMemory (replay_memory.py)
"""

"""
DoubleDQNAgent (DQNagent.py)

Initialization:
- DoubleDQNAgent(args)
    - args: Hyperparameters used during the training process
"""

"""
QLearner (learner.py)
Provides a training function, and functionality for target network learning.

Initialization:
- QLearner(agent, args)
    - agent: DoubleDQNAgent object
    - args: Hyperparameters used during the training process

Use:
- learner.train(memory, batch_size)
    - Samples a batch from memory for training
    - memory: ReplayMemory object
"""

"""
ReplayMemory (replay_memory.py)
- Implements a replay buffer for storing and sampling experiences.
- Supports n-step returns for calculating accumulated rewards.

Initialization:
- ReplayMemory(capacity, n_step, gamma)
    - capacity: Buffer capacity
    - n_step: Length of n steps, used for calculating multi-step returns
    - gamma: Discount factor

Use:
- memory.push(obs, feat, action, reward, next_obs, next_feat, done)
    - obs: Shape (N, 2, 2)
    - feat: Shape (2, 2, 2)
    - next_obs: Shape (N, 2, 2)
    - next_feat: Shape (2, 2, 2)

- memory.sample(batch_size)
    - Output: transitions, each element contains (obs, feat, action, reward, next_obs, next_feat, done, actual_n)
        - Here, next_obs refers to the state after actual_n steps

Data structure:
- n_step_buffer: A temporary buffer used to store the most recent n steps of experiences
- memory: The main buffer for storing processed multi-step experiences
- Each experience contains (obs, feat, action, reward, next_obs, next_feat, done, actual_n)
"""
