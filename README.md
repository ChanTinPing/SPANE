# SPANE: A Deep Reinforcement Learning Approach for Dynamic VM Scheduling
This code is the experimental implementation from the paper *"Symmetry-Preserving Architecture for Multi-NUMA Environments (SPANE): A Deep Reinforcement Learning Approach for Dynamic VM Scheduling."* The experimental environment for scheduling in this study is built upon modifications to [VMAgent](https://github.com/mail-ecnu/VMAgent), a platform designed for applying Reinforcement Learning (RL) to Virtual Machine (VM) scheduling tasks.

Due to time constraints, the currently available code has not been fully organized, and it cannot run properly. We will reorganize and provide the complete code, along with full instructions, by March 5.


## License
This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.

## Data
The VM dataset used in this paper is the [Huawei-East-1](https://github.com/huaweicloud/VM-placement-dataset/tree/main?tab=readme-ov-file) dataset from Huawei Cloud. The original dataset is available at [`data/Huawei-East-1.csv`](data/Huawei-East-1.csv). In our environment, VMs may experience delays, meaning their start times are not fixed. Therefore, we modify the dataset format to treat the VM's runtime (`lt`, length time) as an inherent property of the VM.

## Scheduling Environment
The scheduling environment used in this study is implemented in the [schedgym/sched_env.py](schedgym/sched_env.py). This environment interacts with the RL agent by processing actions through the `step` function, which returns the `next_state`, `reward`, and `done` flag.

There are two termination conditions for the environment.  
1. The environment terminates after scheduling a certain number of VMs.  
2. The environment terminates when the first VM cannot be immediately deployed, plus an additional number of scheduled VMs.
  
When selecting the VM queue, we first pick an initial VM from the dataset and apply the **first-fit scheduling** strategy until encountering the first VM that cannot be deployed. The total number of scheduled VMs under this first-fit strategy, plus 40, determines the length of the VM queue corresponding to this initial VM.

## Methods
There are two main types of RL agents:  
1. Agents that use heuristic methods, located in the [`baseline_agent`](baseline_agent) folder. For more details, refer to the comments inside, including those in [`__init__.py`](baseline_agent/__init__.py).
2. Neural network agents trained with DQN, located in the [`dqn`](dqn) folder.

## DQN Training + Hyperparameters Selection
The `para_search.py` and `run_exp.py` scripts are used to find the best-performing hyperparameters and generate the corresponding agent. Running `para_search.py` will create a `log_para_{time}` directory to store the results. `run_exp.py` is the training script, while `para_search.py` is responsible for running `run_exp.py` in parallel.

- The **`fig`** subfolder stores the results of various hyperparameters in the form of distribution plots.  
- The **`result`** subfolder contains the hyperparameters and results of each experiment, with an `all_results.json` file that consolidates the results of all experiments.  
- The **`tensorboard`** subfolder stores model data and process data (such as training and validation loss) for each experiment. Model parameters are stored in the **`models`** subfolder within the corresponding experiment directory. 

To view process data, use the command:  
```bash
tensorboard --logdir=log_para_{time}
```
This command displays data for all experiments. To view data for a specific experiment, replace the `logdir` parameter with the corresponding experiment directory, e.g.,  
```bash
tensorboard --logdir=log_para_{time}/{trial_id}_...
```

## Experiment (Wait Time Analysis)


## Experiment (Flexibility of Spane)

