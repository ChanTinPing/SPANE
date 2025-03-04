''' Evaluate the results of First Fit and Balance Fit '''

import numpy as np
from schedgym.sched_env import SchedEnv 
from baseline_agent import get_fit_func
from tqdm import trange
from common import stage_mean
DATA_PATH = "data/Huawei-East-1-lt.csv"

# Validation Result
valid_inds = np.load('data/valid_random_time_150.npy')
num_episodes = 150
# Test Result
# valid_inds = np.load('data/test_random_time_1000.npy')
# num_episodes = 1000

first_fit = get_fit_func(0, 40, 90)
def run_episode(env: SchedEnv, agent, index):
    # 1. Use First Fit to determine N_vm for the comparison experiment
    state = env.reset(index, exceed_vm=1)
    done = False
    while not done:
        action = first_fit(state)
        state, _, done = env.step(action)
    N_vm = env.get_attr('length_vm')
    N_vm += 40  

    # 2. Testing
    state = env.reset(index, N_vm=N_vm)
    done = False
    while not done:
        action = agent(state)
        state, _, done = env.step(action)

    return env.get_attr('total_wait_time')

def main():
    # Environment parameters
    N = 5  # Number of servers
    cpu = 40  # Total CPU per NUMA
    mem = 90  # Total memory per NUMA

    env = SchedEnv(N, cpu, mem, DATA_PATH, 10)
    first_fit = get_fit_func(0, cpu, mem)  # First Fit agent
    bal_fit = get_fit_func(2, cpu, mem)    # Balance Fit agent

    # Run multiple episodes
    res_f = []  # Results for First Fit
    res_b = []  # Results for Balance Fit
    for episode in trange(num_episodes):
        index = valid_inds[episode]  # Select a valid test index
        res_f.append(run_episode(env, first_fit, index))
        res_b.append(run_episode(env, bal_fit, index))
    print(f"first fit: {stage_mean(res_f)}")
    print(f"balance fit: {stage_mean(res_b)}")

if __name__ == "__main__":
    main()
