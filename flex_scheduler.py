' Compare the results of SPANE (trained in 5 PMs) in different number of PM. '

import subprocess
import json
import os
from multiprocessing import Pool

def run_experiment(args):
    params, trial_id, PM_num = args
    cmd = [
        "python", "flex_exp.py",
        "--method", 'SPANE',
        "--nn_width_server", str(params['nn_width_server']),
        "--nn_num_server", str(params['nn_num_server']),
        "--nn_width_value", str(params['nn_width_value']),
        "--nn_num_value", str(params['nn_num_value']),
        "--nn_width_adv", str(params['nn_width_adv']),
        "--nn_num_adv", str(params['nn_num_adv']),
        "--cluster_agg", str(params['cluster_agg']),
        "--PM_num", str(PM_num),
        "--trial_id", trial_id
    ]
    subprocess.run(cmd)

def main():
    PM_nums = range(2, 10)  # [2,3,4,5,6,7,8,9]
    concurrent_trials = 10
    
    # Obtain trials
    trials = []
    for PM_num in PM_nums:
        json_path = f'models/SPANE/results.json'
        with open(json_path, 'r') as f:
            trials_data = json.load(f)
        
        for trial_data in trials_data[0:3]:
            params = trial_data['params']
            trial_id = trial_data['trial_id']
            trials.append((params, trial_id, PM_num))
    
    os.makedirs('result/flex', exist_ok=True)
    # parallel run experiment
    with Pool(processes=concurrent_trials) as pool:
        pool.imap_unordered(run_experiment, trials)
        pool.close()
        pool.join()

if __name__ == "__main__":
    main()
