# SPANE: A Deep Reinforcement Learning Approach for Dynamic VM Scheduling
This code is the experimental implementation from the paper *"Symmetry-Preserving Architecture for Multi-NUMA Environments (SPANE): A Deep Reinforcement Learning Approach for Dynamic VM Scheduling."* The experimental environment for scheduling in this study is built upon modifications to [VMAgent](https://github.com/mail-ecnu/VMAgent), a platform designed for applying Reinforcement Learning (RL) to Virtual Machine (VM) scheduling tasks.

Due to time constraints, the currently available code has not been fully organized, and it cannot run properly. We will reorganize and provide the complete code, along with full instructions, by March 5.


## License
This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.

## Data
The VM dataset used in this paper is the [Huawei-East-1](https://github.com/huaweicloud/VM-placement-dataset/tree/main?tab=readme-ov-file) dataset from Huawei Cloud. The original dataset is available at [`data/Huawei-East-1.csv`](data/Huawei-East-1.csv). In our environment, VMs may experience delays, meaning their start times are not fixed. Therefore, we modify the dataset format to treat the VM's runtime (`lt`, length time) as an inherent property of the VM.

## Scheduling Environment
The scheduling environment used in this study is implemented in the [schedgym/sched_env.py](schedgym/sched_env.py). This environment interacts with the RL agent by processing actions through the `step` function, which returns the `next_state`, `reward`, and `done` flag.

## Methods
There are two main types of RL agents:  
1. Agents that use heuristic methods, located in the [`baseline_agent`](baseline_agent) folder. For more details, refer to the comments inside, including those in [`__init__.py`](baseline_agent/__init__.py).
2. Neural network agents trained with DQN, located in the [`dqn`](dqn) folder.

## DQN Training + Hyperparameters Choosing


## Experiment (Wait Time Analysis)


## Experiment (Flexibility of Spane)

