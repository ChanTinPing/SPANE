'''
储存常见功能
'''
from schedgym.sched_env import SchedEnv
import numpy as np


def linear_decay(epoch, max_epoch, eps_start, eps_end):
    return max(eps_end, eps_start - (eps_start - eps_end) * (epoch / max_epoch))

def time_format(t):
    hour = t//3600
    t = t % 3600 
    mi = t//60 
    t = t % 60 
    se = t 
    return hour, mi, se

def trimmed_mean(data, percentage=0.1):
    ''' return the trimmed mean of data by excluding first percentage and last percentage, total 2*percentage data '''
    if not data:
        raise ValueError("The data list is empty")

    if not 0 <= percentage < 0.5:
        raise ValueError("Percentage must be between 0 and 0.5")

    n = len(data)
    k = int(n * percentage)
    
    # Sort the data
    sorted_data = sorted(data)
    
    # Remove the lowest and highest k elements
    trimmed_data = sorted_data[k:n-k]
    
    # Calculate the mean of the remaining data
    if not trimmed_data:
        raise ValueError("Trimmed data is empty, increase the data size or reduce the percentage")

    mean_value = sum(trimmed_data) / len(trimmed_data)
    return mean_value
