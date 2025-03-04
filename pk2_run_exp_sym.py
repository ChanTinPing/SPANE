import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import numpy as np
import torch as th
from schedgym.sched_env import SchedEnv
from dqn.agent import DoubleDQNAgent
from dqn.learner import QLearner
from dqn.replay_memory import ReplayMemory
from baseline_agent import get_fit_func
from runx.logx import logx  # 储存reward等
from datetime import datetime
import json
from common import linear_decay, time_format, stage_mean
from config import Config 
import argparse
from tqdm import trange
import glob
DATA_PATH = 'data/Huawei-East-1-lt.csv'
valid_inds = np.load('data/test_random_time_1000.npy')
model_dir = 'models/sym_new/'
N = 5   
'env 记得改, 最后的结果记录也要改'

def obj(sp):
    '1. 超参数'
    # 核心超参数
    env = 'recovering'
    args = Config(env)  # 初始化配置类
    args.N = N
    first_fit = get_fit_func(0, args.cpu, args.mem)
    args.valid_num = 1  # 进行验证时，做多少次实验取平均值
    args.alg = 'dqn'
    if sp['easy']:
        args.mode = 'sym_easy'
    else:
        args.mode = 'sym'  # basic, sym
    print(args.mode)
    args.reward_type = 'basic'  # basic, balance

    # 可调核心超参数
    # if args.mode == 'basic':
    #     args.nn_width = [32, 32]  # 神经网络隐藏层的维度
    if (args.mode == 'sym') or (args.mode == 'sym_easy'):
        args.nn_width = {
            'server': (sp['nn_width_server'], sp['nn_num_server']),
            'value': (sp['nn_width_value'], sp['nn_num_value']),
            'adv': (sp['nn_width_adv'], sp['nn_num_adv'])
        }
        aggs_dicts = {0: 'mean', 1: 'max', 2: 'std'}
        args.cluster_agg = [aggs_dicts[i] for i in sp['cluster_agg']]
    args.reward_weight = 0

    # dqn超参数
    # 在线训练时，agent 有以概率 eps 随机选择动作，以概率 1-eps 选择 agent 认为的最优动作（Q 值最大）
    # eps 在在线训练时会从 eps_1 线性衰减至 eps_2
    args.eps_1 = 0.6          # 刚开始在线训练时，agent 选择动作时采用随机方法的概率
    args.eps_2 = 0.0          # 在线训练结束时  ，agent 选择动作时采用随机方法的概率
    args.n_step = 50              # 损失函数的多步回放（n_step 越大，偏差越小，方差越大。n_step 越小，偏差越大，方差越小）
    args.memory_capacity = 100000 # 经验回放的大小
    args.weight_decay = 1e-8  # L2正则化权重
    args.lr = sp['lr']           # 初始学习率
    args.scheduler = 'step'   # 学习率衰减器类型，具体看learner

    # epoch超参数
    args.online_epoch = 5000  # 每个 epoch 取一个 batch_size 的数据学习
    args.run_no = 1           # 每次epoch与环境交互多少次收集数据
    if args.run_no > 0:
        args.run_interval = None
    else:
        args.run_interval = 2
    args.batch_size = 1024     # 每次训练的数据量
    args.target_update_interval = 100

    '2. 函数'
    # 与环境做交互，目标是返回 reward，或者存储经验（在线训练时需要存储经验）
    def _run(env, initial_index, agent, fit_func, memory, eps, is_valid):
        # 1. first fit 得到 N_vm
        state = env.reset(initial_index, exceed_vm=1)
        done = False
        while not done:
            action = first_fit(state)
            state, _, done = env.step(action)
        N_vm = env.get_attr('length_vm')
        N_vm += 40

        # 2. 测试
        # 重置环境，若环境太大则别的重置
        try:
            state = env.reset(initial_index, N_vm=N_vm)
        except ValueError as e:
            if str(e) == "index + N_vm 大于总数据长度":
                state = env.reset(initial_index, exceed_vm=1)
        tot_reward = 0  # 总 reward
        done = False
        while not done:
            # 采取行动，做交互
            if agent:  action = agent.select_action(state, eps)
            else:      action = fit_func(state)
            state2, reward, done = env.step(action)
            tot_reward += reward

            # 添加经验进去memory, 切换state
            if not is_valid:
                obs  = state['obs']
                feat = state['feat']
                next_obs  = state2['obs']
                next_feat = state2['feat']
                memory.push(obs, feat, action, reward, next_obs, next_feat, done)
            state = state2

        return tot_reward

    # _run 函数的接口
    def run(env, initial_index, agent, memory, eps, is_valid):
        if is_valid:  # 在线训练时要存储经验
            agent_reward = _run(env, initial_index, agent, None, memory, 0, 1)
        else:             # 验证时不需要存储经验
            agent_reward = _run(env, initial_index, agent, None, memory, eps, 0)
        return agent_reward

    # 验证代码
    def validate(env, agent, args, epoch):
        val_rewards = []
        total_wts = []

        for i in trange(args.valid_num):
            initial_index = valid_inds[i]
            val_return = run(env, initial_index, agent, None, 0, 1)  # 进行验证
            val_rewards.append(val_return)
            total_wait_time = env.get_attr('total_wait_time')
            total_wts.append(total_wait_time)

        # 记录验证结果
        return total_wts

    '3. 准备'
    env = SchedEnv(args.N, args.cpu, args.mem, DATA_PATH, args.double_thr, args.reward_type, args.reward_weight)
    agent = DoubleDQNAgent(args, args.mode)

    model_pattern = f"{sp['trial_id']}_*.th"
    matching_files = glob.glob(os.path.join(model_dir, model_pattern))
    model_path = matching_files[0]  # Use the first matching file if multiple exist
    agent.load(model_path)

    '5. 在线训练'
    # 验证环节
    result = validate(env, agent, args, 0)   
    return result


if __name__ == "__main__":
    # 超参数
    parser = argparse.ArgumentParser(description='Run DQN experiment')
    parser.add_argument('--nn_width_server', type=int, required=True)
    parser.add_argument('--nn_num_server', type=int, required=True)
    parser.add_argument('--nn_width_value', type=int, required=True)
    parser.add_argument('--nn_num_value', type=int, required=True)
    parser.add_argument('--nn_width_adv', type=int, required=True)
    parser.add_argument('--nn_num_adv', type=int, required=True)
    parser.add_argument('--cluster_agg', type=str, required=True)
    parser.add_argument('--easy', type=int, required=True)
    parser.add_argument('--lr', type=float, required=True)
    parser.add_argument('--trial_id', type=str, required=True)
    sp = parser.parse_args()
    sp.cluster_agg = [int(x) for x in sp.cluster_agg.strip('[]').split(',')]
    sp_dict = vars(sp)

    # 运行
    total_wts = obj(sp_dict)

    # 结果
    mean_wts = stage_mean(total_wts)
    print(f"Sym Trial ID: {sp.trial_id}, Result: {mean_wts}")
    with open(f'result/S/mean.txt', 'a') as file:
        file.write(f'Sym trial_id: {sp.trial_id},  mean: {mean_wts} \n')
    os.makedirs(f'result/S/sym_original', exist_ok=True)
    np.savetxt(f'result/S/sym_original0/{sp.trial_id}.txt', np.array(total_wts), fmt='%d')
