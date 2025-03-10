import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# Prevent each parallel process from internally spawning multiple threads, avoiding CPU resource contention
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import numpy as np
import argparse
from tqdm import trange
import glob
from types import SimpleNamespace
from schedgym.sched_env import SchedEnv
from dqn.agent import DoubleDQNAgent
from baseline_agent import get_fit_func
from common import trimmed_mean

DATA_PATH = 'data/Huawei-East-1-lt.csv'
test_inds = np.load('data/test_random_time_1000.npy')
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

    args.test_num = 1000

    '2. Functions'
    def run_test(env: SchedEnv, initial_index, agent: DoubleDQNAgent):
        ''' # Interact with the environment to return rewards '''
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
            action = agent.select_action(state, epsilon=0)
            state, reward, done = env.step(action)
            tot_reward += reward

        return tot_reward

    def testing(env, agent, args):
        val_rewards = []
        total_wts = []

        for i in trange(args.test_num):
            initial_index = test_inds[i]
            val_return = run_test(env, initial_index, agent)  
            val_rewards.append(val_return)
            total_wait_time = env.get_attr('total_wait_time')
            total_wts.append(total_wait_time)

        return total_wts

    '3. Preparation'
    env = SchedEnv(args.N, args.cpu, args.mem, DATA_PATH, args.double_thr, args.reward_type)
    agent = DoubleDQNAgent(args, args.mode)

    model_pattern = f"{sp.trial_id}_*.th"
    model_dir = f'models/{args.method}'
    matching_files = glob.glob(os.path.join(model_dir, model_pattern))
    model_path = matching_files[0]  # Use the first matching file if multiple exist
    agent.load(model_path)

    '4. Testing'
    result = testing(env, agent, args)   
    return result


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
    elif (method == 'MLPDQN') or (method == 'MLPAUG'):
        parser.add_argument('--layer_no', type=int, required=True)
        parser.add_argument('--width', type=int, required=True)
    else:
        raise ValueError(f"Unsupported method: {method}. Valid options are 'SPANE', 'MLPDQN', or 'MLPAUG'.")
    parser.add_argument('--trial_id', type=str, required=True)

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

    # Run
    total_wts = main(sp)
    mean_wts = trimmed_mean(total_wts)

    # Store results 
    print(f"{sp.method} Trial_id: {sp.trial_id}, Result: {mean_wts}")
    
    with open(f'result/wait_time/result.txt', 'a') as file:
        file.write(f'{sp.method} Trial_id: {sp.trial_id}, Result: {mean_wts} \n')
    result_dir = f'result/wait_time/{sp.method}'
    os.makedirs(result_dir, exist_ok=True)
    np.savetxt(f'{result_dir}/{sp.trial_id}.txt', np.array(total_wts), fmt='%d')
