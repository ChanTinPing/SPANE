''' Select the NUMA or server where the most recently installed VM was placed '''
import numpy as np

def movetofront_one_env(avail, action0):  
    # If the most recently used NUMA/server (action0) is available, return it
    if avail[action0] == 1:
        return action0
    else:
        # Otherwise, select the first available NUMA/server
        return np.where(avail == 1)[0][0]

def movetofront_fit(state, action0):
    avails = state["avail"].copy()
    action = movetofront_one_env(avails, action0)
    return action
