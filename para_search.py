''' Hyperparameter Search '''
from itertools import product, cycle, islice
import random
import subprocess
import json
import time
import os
import signal
from datetime import datetime
from multiprocessing import Pool, Manager
from para_search_func import generate_trial_id, print_best_result, save_results, visualize_results

def run_experiment(args):
    params, trial_id, log_dir = args
    cmd = [
        "python", "run_exp.py",
        "--nn_width_server", str(params['nn_width_server']),
        "--nn_num_server", str(params['nn_num_server']),
        "--nn_width_value", str(params['nn_width_value']),
        "--nn_num_value", str(params['nn_num_value']),
        "--nn_width_adv", str(params['nn_width_adv']),
        "--nn_num_adv", str(params['nn_num_adv']),
        "--cluster_agg", str(params['cluster_agg']),
        "--lr", str(params['lr']),
        "--trial_id", trial_id,
        "--log_dir", log_dir
    ]
    # Run the experiment
    subprocess.run(cmd)

    # Path to the result file
    result_file = f'{log_dir}/result/{trial_id}.json'
    # Wait for the result file to be created
    while not os.path.exists(result_file):
        time.sleep(30)  
    # Load the result once the file is available
    with open(result_file, 'r') as f:
        result = json.load(f) 
    # Add hyperparameters to the result and save back to the file
    result['params'] = params
    with open(result_file, 'w') as f:
        json.dump(result, f, indent=2)

    return {
        'trial_id': trial_id,
        'params': params,
        'result': result['result']
    }

def main():
    # Define the hyperparameter space
    param_space = {
        'nn_width_server': [4, 8, 16, 32],
        'nn_num_server': [1, 2, 3, 4],
        'nn_width_value': [4, 8, 16, 32],
        'nn_num_value': [1, 2, 3],
        'nn_width_adv': [4, 8, 16, 32],
        'nn_num_adv': [1, 2, 3, 4],
        'cluster_agg': [[0]],    # [[0], [1], [2], [0,1], [0,2], [1,2], [0,1,2]], {0: 'mean', 1: 'max', 2: 'std'}
        'lr': [0.001, 0.01, 0.1]
    }
    # Total trials and number of concurrent trials
    total_trials = 1000
    concurrent_trials = 30

    # Create log directory
    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y%m%d_%H%M%S")
    log_dir = f'log_para_{formatted_time}'
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(f'{log_dir}/result', exist_ok=True)

    # Generate all trial configurations (random combination, for huge param_space)
    all_trials = []
    for _ in range(total_trials):
        params = {k: random.choice(v) for k, v in param_space.items()}
        trial_id = generate_trial_id()
        all_trials.append((params, trial_id, log_dir))
    # Generate all trial configurations (normal combination, for small param_space)
    # keys = list(param_space.keys())
    # all_param_combinations = list(product(*[param_space[k] for k in keys]))
    # trial_combinations = list(islice(cycle(all_param_combinations), total_trials))
    # all_trials = []
    # for values in trial_combinations:
    #     params = dict(zip(keys, values))
    #     trial_id = generate_trial_id()
    #     all_trials.append((params, trial_id, log_dir))

    # Use multiprocessing with interruption handling
    manager = Manager()
    results = manager.list()
    
    # Signal handler for interruption
    def signal_handler(signum, frame):
        print("Interrupt received, terminating all running trials...")
        pool.terminate()
        pool.join()
        print_best_result(results)
        visualize_results(list(results), log_dir)
        save_results(results, log_dir)
        exit(0)
    signal.signal(signal.SIGINT, signal_handler)

    # Parallel processing
    with Pool(processes=concurrent_trials) as pool:
        try:
            for result in pool.imap_unordered(run_experiment, all_trials):
                results.append(result)
        except KeyboardInterrupt:
            print("KeyboardInterrupt caught in main loop")
            pool.terminate()
        finally:
            pool.close()
            pool.join()

    # Process results
    print_best_result(results)
    visualize_results(list(results), log_dir)
    save_results(results, log_dir)

if __name__ == "__main__":
    main()
