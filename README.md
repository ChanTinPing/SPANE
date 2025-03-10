# SPANE: A Deep Reinforcement Learning Approach for Dynamic VM Scheduling
This code is the experimental implementation from the paper *"Symmetry-Preserving Architecture for Multi-NUMA Environments (SPANE): A Deep Reinforcement Learning Approach for Dynamic VM Scheduling."* The experimental environment for scheduling in this study is built upon modifications to [VMAgent](https://github.com/mail-ecnu/VMAgent), a platform designed for applying Reinforcement Learning (RL) to Virtual Machine (VM) scheduling tasks.

Due to time constraints, the currently available code has not been fully organized, and it cannot run properly. We will reorganize and provide the complete code, along with full instructions, by March 5.


## License
This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.

## Data
The VM dataset used in this paper is the [Huawei-East-1](https://github.com/huaweicloud/VM-placement-dataset/tree/main?tab=readme-ov-file) dataset from Huawei Cloud. The original dataset is available at [`data/Huawei-East-1.csv`](data/Huawei-East-1.csv). In our environment, VMs may experience delays, meaning their start times are not fixed. Therefore, we modify the dataset format to treat the VM's runtime (`lt`, length time) as an inherent property of the VM.

## Scheduling Environment
The scheduling environment used in this study is implemented in the [schedgym/sched_env.py](schedgym/sched_env.py). This environment interacts with the RL agent by processing actions through the `step` function, which returns the `next_state`, `reward`, and `done` flag. Here, server index starts from 0.

There are two termination conditions for the environment.  
1. The environment terminates after scheduling a certain number of VMs.  
2. The environment terminates when the first VM cannot be immediately deployed, plus an additional number of scheduled VMs.

When selecting the VM queue, we first pick an initial VM from the dataset and apply the **first-fit** scheduling strategy until encountering the first VM that cannot be deployed. The total number of scheduled VMs under this first-fit strategy, plus 40, determines the length of the VM queue corresponding to this initial VM.

## Methods
There are two main types of RL agents:  
1. Agents that use heuristic methods, located in the [`baseline_agent`](baseline_agent) folder. For more details, refer to the comments inside, including those in [`__init__.py`](baseline_agent/__init__.py).
2. Neural network agents trained with DQN, located in the [`dqn`](dqn) folder.

## DQN Training and Hyperparameter Selection
The `para_search_{method}.py` and `train.py` scripts are designed for hyperparameter optimization and agent generation. The `method` variable can be set to one of the following algorithms: `SPANE`, `MLPDQN`, or `MLPAUG`. Specifically, `train.py` serves as the training execution script, while `para_search_{method}.py` manages parallel runs of the training process. The following is an introduction to the code and results, with the last paragraph providing implementation suggestions.

When executing `para_search_{method}.py`, it generates a folder `log_para_{time}` to store experimental outcomes, and the parameters of the `top_k` strongest-performing models across all trials are stored in the `models/{method}` folder. Users may customize the hyperparameter search space (`param_space`), total number of trials (`total_trials`), number of parallel tasks(`concurrent_trials`), whether to save the model (`save_model`), and number of stored models `top_k` within `para_search_{method}.py`. You can select the trial configuration generation method by commenting out specific sections of code.

The `log_para_{time}` directory organizes results as follows:  
- The **`fig`** subfolder contains distribution plots visualizing hyperparameter performance.  
- The **`result`** subfolder stores the hyperparameters and corresponding results for each experiment.
- The `all_results.json` file gather the outcomes of all experiments. 
- The **`tensorboard`** subfolder stores model checkpoints and training progress (e.g., loss curves) for each experiment. Inside each experiment's specific directory, the **`models`** subfolder contains the trained model parameters.

To view process data, use the command:  
```bash
tensorboard --logdir=log_para_{time}
```
This command displays data for all experiments. To view data for a specific experiment, replace the `logdir` parameter with the corresponding experiment directory, e.g.,  
```bash
tensorboard --logdir=log_para_{time}/{trial_id}_dqn
```

The suggested approach (which is also the one used in this paper) is as follows:  
1. Set a broad hyperparameter space and use random combination to generate hyperparameter sets. After running the experiments, analyze the plots in the `fig` folder and eliminate the poor-performing hyperparameter values.  
2. Once you have a smaller hyperparameter space, use normal combination to find the best hyperparameter set.  
3. After identifying the best hyperparameters, set `save_model=True` and run experiments to store models' parameters.
Notice that folder `models/{method}` currently stores the model parameters used in the paper.

## Experiment (Wait Time Analysis)
`experiment_wt_schedule.py`, `test_sym.py`, `test_mlp.py`

## Experiment (Flexibility of Spane)

