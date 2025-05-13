' Compare the results of various methods (SPANE, MLPDQN, MLPAUG). '

import subprocess
import json
import os
from multiprocessing import Pool

def run_experiment(args):
    params, trial_id, method = args
    if method == 'SPANE':
        cmd = [
            "python", "wait_time_exp.py",
            "--method", method,
            "--nn_width_server", str(params['nn_width_server']),
            "--nn_num_server", str(params['nn_num_server']),
            "--nn_width_value", str(params['nn_width_value']),
            "--nn_num_value", str(params['nn_num_value']),
            "--nn_width_adv", str(params['nn_width_adv']),
            "--nn_num_adv", str(params['nn_num_adv']),
            "--cluster_agg", str(params['cluster_agg']),
            "--trial_id", trial_id
        ]
    elif (method == 'MLPDQN') or (method == 'MLPAUG'):
        cmd = [
            "python", "wait_time_exp.py",
            "--method", method,
            "--width", str(params['width']),
            "--layer_no", str(params['layer_no']),
            "--trial_id", trial_id
        ]
    subprocess.run(cmd)

def main():
    methods = ['SPANE', 'MLPDQN', 'MLPAUG']
    concurrent_trials = 20
    
    # Obtain trials
    trials = []
    for method in methods:
        json_path = f'models/{method}/results.json'
        with open(json_path, 'r') as f:
            trials_data = json.load(f)
        
        for trial_data in trials_data:
            params = trial_data['params']
            trial_id = trial_data['trial_id']
            trials.append((params, trial_id, method))

    os.makedirs('result/wait_time', exist_ok=True)
    # parallel run experiment
    with Pool(processes=concurrent_trials) as pool:
        pool.imap_unordered(run_experiment, trials)
        pool.close()
        pool.join()

if __name__ == "__main__":
    main()
