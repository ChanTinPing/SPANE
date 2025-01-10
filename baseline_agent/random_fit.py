''' Random '''
import numpy as np

def random_one_env(avail):  
    return np.random.choice(np.where(avail == 1)[0]) 

def random_fit(state):
    avail = state["avail"].copy()
    action = random_one_env(avail)
    return action
