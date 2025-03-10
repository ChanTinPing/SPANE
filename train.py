import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# Prevent each parallel process from internally spawning multiple threads, avoiding CPU resource contention
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import numpy as np
from types import SimpleNamespace
import json
import argparse
from runx.logx import logx
import itertools
import random
from schedgym.sched_env import SchedEnv
from dqn.agent import DoubleDQNAgent
from dqn.learner import QLearner
from dqn.replay_memory import ReplayMemory
from baseline_agent import get_fit_func
from common import linear_decay, trimmed_mean

DATA_PATH = 'data/Huawei-East-1-lt.csv'
valid_inds = np.load('data/valid_random_time_150.npy')
env_config = {
    'env': 'recovering',
    'N': 5,  
    'cpu': 40,
    'mem': 90,
    'allow_release': True, 
    'double_thr': 10,       # Memory threshold, if VM memory >= 10, the VM equally distributed
}
first_fit = get_fit_func(0, env_config['cpu'], env_config['mem'])

def main(sp: argparse.Namespace):
    '1. Hyperparameters'
    args = SimpleNamespace(**env_config)  # Initialize configuration class
    args.alg = 'dqn'
    args.method = sp.method
    if args.method == 'MLPDQN' or args.method == 'MLPAUG':
        args.mode = 'basic'  
    elif args.method == 'SPANE':
        args.mode = 'sym'
    args.reward_type = 'basic'
    args.extra_vm_num = 40

    if args.mode == 'basic':
        args.nn_width = [sp.width] * sp.layer_no   # Neural network hidden layer dimensions
    if args.mode == 'sym':
        args.nn_width = {
            'server': (sp.nn_width_server, sp.nn_num_server),
            'value': (sp.nn_width_value, sp.nn_num_value),
            'adv': (sp.nn_width_adv, sp.nn_num_adv)
        }
        aggs_dicts = {0: 'mean', 1: 'max', 2: 'std'}
        args.cluster_agg = [aggs_dicts[i] for i in sp.cluster_agg]

    # DQN-specific parameters
    args.eps_1 = 0.6  # At the beginning of online training, the probability of the agent selecting an action randomly
    args.eps_2 = 0.0  # At the end of online training, the probability of the agent selecting an action randomly
    args.n_step = 50  # Multi-step return for the loss function (larger n_step reduces bias but increases variance)
    args.memory_capacity = 100000  # Size of the replay buffer
    args.weight_decay = 1e-8  # L2 regularization weight
    args.lr = sp.lr # Initial learning rate
    args.scheduler = 'step'  # Type of learning rate scheduler, see learner for details
    args.gamma = 0.95  # Discount factor for the n-step reward

    # Epoch parameters
    args.online_epoch = 5000  # Number of batches of data learned in each epoch
    if args.method == 'MLPAUG':
        args.transform_num = sp['transform_num']
        args.run_no = None  # Number of data collections with the environment in each epoch
        args.run_interval = args.transform_num
    else:
        args.run_no = 1  # Number of data collections with the environment in each epoch
        args.run_interval = None 
    args.batch_size = 1024
    args.valid_interval = 250  # Interval (in epochs) for validation
    args.valid_num = 150  # Number of experiments during validation, average taken
    args.first_tot_wt = 235180.14  # Trimmed average result of the first valid_num experiments for First Fit during validation. Change this if valid_time or valid_num changes.
    args.bal_tot_wt = 199748.26    # Trimmed average result of the first valid_num experiments for Balance Fit during validation
    args.target_update_interval = 100

    # Create log folder with current timestamp
    logpath = os.path.join(sp.log_dir, 'tensorboard', f"{sp.trial_id}_{args.alg}")

    '2. Functions'
    def generate_random_permutations(N, num):
        ''' generate num random permutations, include identity, only used by MLPAUG '''
        sequence = list(range(N))
        all_permutations = list(itertools.permutations(sequence))
        if num > len(all_permutations):
            raise ValueError(f'generate random permutations number too much!')
        if num <= 0:
            raise ValueError(f'num in generate_random_permuations much be positive integer!')
        
        identity_permutation = tuple(sequence)
        result = [identity_permutation]
        remaining_permutations = [perm for perm in all_permutations if perm != identity_permutation]
        random_permutations = random.sample(remaining_permutations, num-1)
        result.extend(random_permutations)
        return result

    def transform_state_action(trans: tuple[int, ...], obs, action, next_obs):
        ''' return after transformation obs, action, next_obs, only used by MLPAUG '''
        if len(trans) != obs.shape[0]:
            raise ValueError(f"shape of transformation incorrect! trans_length: {len(trans)}, obs.shape[0]: {obs.shape[0]}")
        
        t_obs = obs[list(trans)]
        t_next_obs = next_obs[list(trans)]

        inv_trans = [0] * len(trans)
        for i, mapped_index in enumerate(trans):
            inv_trans[mapped_index] = i
        
        server, numa = action // 2, action % 2
        t_server = inv_trans[server]  
        t_action = 2 * t_server + numa
        
        return t_obs, t_action, t_next_obs

    def push_memory(replay_memory, method, obs, feat, action, reward, next_obs, next_feat, done):
        if method == 'MLPAUG':
            transformation_groups = generate_random_permutations(args.N, args.transform_num)
            for trans in transformation_groups: 
                t_obs, t_action, t_next_obs = transform_state_action(trans, obs, action, next_obs)
                replay_memory.push(t_obs, feat, t_action, reward, t_next_obs, next_feat, done) 
        else:
            replay_memory.push(obs, feat, action, reward, next_obs, next_feat, done)
            
    def run(env: SchedEnv, initial_index, agent: DoubleDQNAgent, memory: ReplayMemory, eps, is_valid):
        ''' # Interact with the environment to return rewards or store experiences (required for online training) '''
        # First Fit to determine N_vm
        state = env.reset(initial_index, exceed_vm=1)
        done = False
        while not done:
            action = first_fit(state)
            state, _, done = env.step(action)
        N_vm = env.get_attr('length_vm') + args.extra_vm_num

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
            action = agent.select_action(state, eps)
            state2, reward, done = env.step(action)
            tot_reward += reward

            # Store experience in memory if training
            if not is_valid:
                obs = state['obs']
                feat = state['feat']
                next_obs = state2['obs']
                next_feat = state2['feat']
                push_memory(memory, args.method, obs, feat, action, reward, next_obs, next_feat, done)
            state = state2

        return tot_reward

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
        total_wt_mean = trimmed_mean(total_wts)
        val_metric = {
            'tot_reward': trimmed_mean(val_rewards),
            'tot_wt': total_wt_mean,
            'bal_tot_wt_diff': args.bal_tot_wt - total_wt_mean,
            'first_tot_wt_diff': args.first_tot_wt - total_wt_mean,
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
        initial_index = np.random.randint(0, 50000)
        return run(env, initial_index, agent, memory_on, eps, False)

    '3. Preparation'
    logx.initialize(logdir=logpath, coolname=True, tensorboard=True)
    print(f"logpath ({logpath}) created")

    env = SchedEnv(args.N, args.cpu, args.mem, DATA_PATH, args.double_thr, args.reward_type)
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

    '4. Pre-collect data'
    if args.method == 'MLPAUG':
        precollect_epoch = int(100 + (args.transform_num-1) / args.transform_num)
    else:
        precollect_epoch = 100
    for _ in range(precollect_epoch):
        initial_time = np.random.randint(0, 50000)
        eps = args.eps_1
        run(env, initial_time, agent, memory_on, eps, False)

    '5. Online training'
    epoch = 0
    best_result = -100000
    best_result_epoch = 0
    same_result_count = 0
    last_result = None
    for _ in range(args.online_epoch + 1):
        # Collect data through interaction
        eps = linear_decay(epoch, args.online_epoch, args.eps_1, args.eps_2)
        if args.run_no is not None:  # Collect data run_no times per epoch
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
        if epoch % args.valid_interval == 0:
            result = validate(env, agent, args, epoch)
            if best_result < result:
                best_result = result
                best_result_epoch = epoch
            same_result_count = same_result_count + 1 if result == last_result else 0
            last_result = result
            # Early stopping if the result is the same for three consecutive validations
            if same_result_count >= 3:
                break
        epoch += 1

    return best_result, best_result_epoch


def parser_add_specific_args(parser, method):
    parser.add_argument('--method', type=str, required=True)
    if method == 'SPANE':
        parser.add_argument('--nn_width_server', type=int, required=True)
        parser.add_argument('--nn_num_server', type=int, required=True)
        parser.add_argument('--nn_width_value', type=int, required=True)
        parser.add_argument('--nn_num_value', type=int, required=True)
        parser.add_argument('--nn_width_adv', type=int, required=True)
        parser.add_argument('--nn_num_adv', type=int, required=True)
        parser.add_argument('--cluster_agg', type=str, required=True)
    elif method == 'MLPDQN':
        parser.add_argument('--layer_no', type=int, required=True)
        parser.add_argument('--width', type=int, required=True)    
    elif method == 'MLPAUG':
        parser.add_argument('--layer_no', type=int, required=True)
        parser.add_argument('--width', type=int, required=True)
        parser.add_argument('--transform_num', type=int, required=True)
    else:
        raise ValueError(f"Unsupported method: {method}. Valid options are 'SPANE', 'MLPDQN', or 'MLPAUG'.")
    parser.add_argument('--lr', type=float, required=True)
    parser.add_argument('--trial_id', type=str, required=True)
    parser.add_argument('--log_dir', type=str, required=True)

if __name__ == "__main__":
    # Parse
    first_parser = argparse.ArgumentParser(add_help=False)
    first_parser.add_argument('--method', type=str, required=True)
    method_args, remaining = first_parser.parse_known_args()
    method = method_args.method
    
    full_parser = argparse.ArgumentParser()
    parser_add_specific_args(full_parser, method)
    sp = full_parser.parse_args(namespace=method_args)
    
    if hasattr(sp, 'cluster_agg'):
        sp.cluster_agg = [int(x) for x in sp.cluster_agg.strip('[]').split(',')]
    sp_dict = vars(sp)

    # Run
    result, best_result_epoch = main(sp)
    print(f"Trial ID: {sp.trial_id}, Result: {result}, Best Epoch: {best_result_epoch}")

    # Store results
    log_dir = sp.log_dir
    os.makedirs(f'{log_dir}/result', exist_ok=True)
    with open(f'{log_dir}/result/{sp.trial_id}.json', 'w') as f:
        json.dump({'trial_id': sp.trial_id, 'result': result, 'best_epoch': best_result_epoch}, f)
