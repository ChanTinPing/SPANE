' 比较各个方法 (mlp, mlp_aug, sym) 的结果 '

import subprocess
import json
from multiprocessing import Pool

def run_experiment_sym(args):
    params, trial_id = args
    cmd = [
        "python", "test_sym.py",
        "--nn_width_server", str(params['nn_width_server']),
        "--nn_num_server", str(params['nn_num_server']),
        "--nn_width_value", str(params['nn_width_value']),
        "--nn_num_value", str(params['nn_num_value']),
        "--nn_width_adv", str(params['nn_width_adv']),
        "--nn_num_adv", str(params['nn_num_adv']),
        "--cluster_agg", str(params['cluster_agg']),
        "--lr", str(params['lr']),
        "--easy", str(params['easy']),
        "--trial_id", trial_id
    ]
    subprocess.run(cmd)

def run_experiment_mlp(args):
    params, trial_id = args
    cmd = [
        "python", "test_mlp.py",
        "--width", str(params['width']),
        "--layer_no", str(params['layer_no']),
        "--lr", str(params['lr']),
        "--trial_id", trial_id
    ]
    subprocess.run(cmd)


def main():
    ' 1. 全连接层 '
    mlp_json_path = 'models/mlp_aug/results.json'

    # 得到 mlp_trials
    # mlp_trials = []

    # with open(mlp_json_path, 'r') as f:
    #     trials_data = json.load(f)
    
    # for trial in trials_data:
    #     params = trial['params']
    #     trial_id = trial['trial_id']
    #     mlp_trials.append((params, trial_id))

    # # 并行运算
    # with Pool(processes=len(mlp_trials)) as pool:
    #     pool.imap_unordered(run_experiment_mlp, mlp_trials)
    #     pool.close()
    #     pool.join()

    # ' 2. sym '
    sym_json_path = 'models/spane/results.json'

    # 得到 sym_trials
    sym_trials = []

    with open(sym_json_path, 'r') as f:
        trials_data = json.load(f)
    
    for trial in trials_data:
        params = trial['params']
        trial_id = trial['trial_id']
        sym_trials.append((params, trial_id))

    # 并行运算
    with Pool(processes=len(sym_trials)) as pool:
        pool.imap_unordered(run_experiment_sym, sym_trials)
        pool.close()
        pool.join()


if __name__ == "__main__":
    main()
