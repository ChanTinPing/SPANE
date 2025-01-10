''' Select the first NUMA or server that can accommodate the VM request '''
import numpy as np

def first_fit_one_env(avail):  
    # Select the first available NUMA or server that can place the VM
    return np.where(avail == 1)[0][0]  

def first_fit(state):
    avails = state["avail"].copy()
    action = first_fit_one_env(avails)
    return action
