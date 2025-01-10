import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import numpy as np
from schedgym.sched_env import SchedEnv
from dqn.agent import DoubleDQNAgent
from dqn.learner import QLearner
from dqn.replay_memory import ReplayMemory
from baseline_agent import get_fit_func
from runx.logx import logx
from datetime import datetime
import json
from common import linear_decay
from config import Config
import argparse

DATA_PATH = f'data/Huawei-East-1-lt.csv'
valid_inds = np.load(f'data/test_random_time_1000.npy')


def obj(sp):
    # 1. Hyperparameters
    env = 'recovering'
    args = Config(env)  # Initialize configuration class
    args.alg = 'dqn'
    args.mode = 'sym'  # Options: basic, sym
    args.reward_type = 'basic'  # Options: basic, balance
    args.exceed_vm = 20

    if args.mode == 'basic':
        args.nn_width = [32, 32]  # Neural network hidden layer dimensions
    if args.mode == 'sym':
        args.nn_width = {
            'server': (sp['nn_width_server'], sp['nn_num_server']),
            'value': (sp['nn_width_value'], sp['nn_num_value']),
            'adv': (sp['nn_width_adv'], sp['nn_num_adv'])
        }
        aggs_dicts = {0: 'mean', 1: 'max', 2: 'std'}
        args.cluster_agg = [aggs_dicts[i] for i in sp['cluster_agg']]
    args.reward_weight = 0

    # DQN-specific parameters
    args.eps_1 = 0.6  # At the beginning of online training, the probability of the agent selecting an action randomly
    args.eps_2 = 0.0  # At the end of online training, the probability of the agent selecting an action randomly
    args.n_step = 50  # Multi-step return for the loss function (larger n_step reduces bias but increases variance)
    args.memory_capacity = 100000  # Size of the replay buffer
    args.weight_decay = 1e-8  # L2 regularization weight
    args.lr = sp['lr']  # Initial learning rate
    args.scheduler = 'step'  # Type of learning rate scheduler, see learner for details

    # Epoch parameters
    args.online_epoch = 5000  # Number of batches of data learned in each epoch
    args.run_no = 1  # Number of data collections with the environment in each epoch
    args.run_interval = None if args.run_no > 0 else 2
    args.batch_size = 1024
    args.test_interval = 250  # Interval (in epochs) for validation
    args.valid_num = 150  # Number of experiments during validation, average taken
    args.first_tot_wt = 37125.386  # Average result of the first valid_num experiments for First Fit during validation. Change this if valid_time or valid_num changes.
    args.bal_tot_wt = 44063.02     # Average result of the first valid_num experiments for Balance Fit during validation
    args.target_update_interval = 100

    # Create log folder with current timestamp
    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y%m%d_%H%M%S")
    logpath = f"{sp['log_dir']}/tensorboard/{sp['trial_id']}_{args.alg}_{args.env}_{formatted_time}"

    # 2. Functions
    def _run(env, initial_index, agent, fit_func, memory, eps, is_valid):
        ''' # Interact with the environment to return rewards or store experiences (required for online training) '''
        # First Fit to determine N_vm
        state = env.reset(initial_index, exceed_vm=1)
        done = False
        while not done:
            action = get_fit_func(0, args.cpu, args.mem)(state)
            state, _, done = env.step(action)
        N_vm = env.get_attr('length_vm') + 40

        # Reset environment with N_vm or fallback
        try:
            state = env.reset(initial_index, N_vm=N_vm)
        except ValueError as e:
            if str(e) == "index + N_vm exceeds total data length":
                state = env.reset(initial_index, exceed_vm=1)

        tot_reward = 0
        done = False
        while not done:
            # Agent action selection
            action = agent.select_action(state, eps) if agent else fit_func(state)
            state2, reward, done = env.step(action)
            tot_reward += reward

            # Store experience in memory if training
            if not is_valid:
                obs = state['obs']
                feat = state['feat']
                next_obs = state2['obs']
                next_feat = state2['feat']
                memory.push(obs, feat, action, reward, next_obs, next_feat, done)
            state = state2

        return tot_reward

    def run(env, initial_index, agent, memory, eps, is_valid):
        return _run(env, initial_index, agent, None, memory, eps, is_valid)

    def validate(env, agent, args, epoch):
        val_rewards = []
        total_wts = []

        for i in range(args.valid_num):
            initial_index = valid_inds[i]
            val_return = run(env, initial_index, agent, None, 0, True)
            val_rewards.append(val_return)
            total_wait_time = env.get_attr('total_wait_time')
            total_wts.append(total_wait_time)

        # Record validation results
        total_wt_mean = np.mean(total_wts)
        val_metric = {
            'tot_reward': np.mean(val_rewards),
            'tot_wt': total_wt_mean,
            'bal_tot_wt_diff': args.bal_tot_wt - total_wt_mean,
            'first_tot_wt_diff': args.first_tot_wt - total_wt_mean
        }
        logx.metric('val', val_metric, epoch)

        # Save model
        model_path = f'{logpath}/models/{epoch}.th'
        model_save(agent, model_path)

        return args.first_tot_wt - total_wt_mean

    def model_save(agent, model_path):
        dir_path = os.path.dirname(model_path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        agent.save(model_path)

    def interact(env, agent, memory_on, eps):
        initial_index = np.random.randint(0, 80000)
        return run(env, initial_index, agent, memory_on, eps, False)

    # 3. Preparation
    logx.initialize(logdir=logpath, coolname=True, tensorboard=True)
    print(f"logpath ({logpath}) created")

    env = SchedEnv(args.N, args.cpu, args.mem, DATA_PATH, args.double_thr, args.reward_type, args.reward_weight)
    agent = DoubleDQNAgent(args, args.mode)
    memory_on = ReplayMemory(args.memory_capacity, args.n_step, args.gamma)
    learner = QLearner(agent, args)

    network_shape = [(name, param.size()) for name, param in agent.online_net.named_parameters()]
    args_dict = vars(args)
    args_dict['network_shape'] = network_shape

    args_json = json.dumps(args_dict, indent=4)
    file_path = f"{logpath}/args.json"
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as file:
        file.write(args_json)

    # 4. Pre-collect data
    for _ in range(100):
        initial_time = np.random.randint(0, 80000)
        eps = args.eps_1
        run(env, initial_time, agent, memory_on, eps, False)

    # 5. Online training
    epoch = 0
    best_result = -100000
    same_result_count = 0
    last_result = None
    for _ in range(args.online_epoch + 1):
        # Collect data through interaction
        eps = linear_decay(epoch, args.online_epoch, args.eps_1, args.eps_2)
        if args.run_no > 0:  # Collect data run_no times per epoch
            for _ in range(args.run_no):
                train_reward = interact(env, agent, memory_on, eps)
        else:   # Collect data every few epochs
            if epoch % args.run_interval == 0:
                train_reward = interact(env, agent, memory_on, eps)

        # Model learning (train on data sampled from memory)
        loss = learner.train(memory_on, args.batch_size)
        # Store data
        if loss == None: loss = 0
        logx.metric('train', {'eps': eps, 'tot_reward': train_reward, 'loss': loss}, epoch)

        # Validation phase
        if epoch % args.test_interval == 0:
            result = validate(env, agent, args, epoch)
            if best_result < result:
                best_result = result
            same_result_count = same_result_count + 1 if result == last_result else 0
            last_result = result
            # Early stopping if the result is the same for three consecutive validations
            if same_result_count >= 3:
                break
        epoch += 1

    return best_result


if __name__ == "__main__":
    # Hyperparameters
    parser = argparse.ArgumentParser(description='Run DQN experiment')
    parser.add_argument('--nn_width_server', type=int, required=True)
    parser.add_argument('--nn_num_server', type=int, required=True)
    parser.add_argument('--nn_width_value', type=int, required=True)
    parser.add_argument('--nn_num_value', type=int, required=True)
    parser.add_argument('--nn_width_adv', type=int, required=True)
    parser.add_argument('--nn_num_adv', type=int, required=True)
    parser.add_argument('--cluster_agg', type=str, required=True)
    parser.add_argument('--lr', type=float, required=True)
    parser.add_argument('--trial_id', type=str, required=True)
    parser.add_argument('--log_dir', type=str, required=True)
    sp = parser.parse_args()
    sp.cluster_agg = [int(x) for x in sp.cluster_agg.strip('[]').split(',')]
    sp_dict = vars(sp)

    # Run
    result = obj(sp_dict)
    print(f"Trial ID: {sp.trial_id}, Result: {result}")

    # Store results
    log_dir = sp.log_dir
    os.makedirs(f'{log_dir}/result', exist_ok=True)
    with open(f'{log_dir}/result/{sp.trial_id}.json', 'w') as f:
        json.dump({'trial_id': sp.trial_id, 'result': result}, f)
