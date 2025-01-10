from .first_fit   import first_fit
from .best_fit    import best_fit
from .balance_fit import balance_fit
from .random_fit  import random_fit
from .move_to_front import movetofront_fit

get_action_funcs = [first_fit, best_fit, balance_fit, random_fit, movetofront_fit]

def get_fit_func(index, total_cpu, total_mem):
    fit_func = get_action_funcs[index]
    fit_func.total_cpu = total_cpu
    fit_func.total_mem = total_mem
    return fit_func

"""
Collection of classic algorithms.

Usage:
    1)
    from baseline_agent import get_action_funcs
    fit_func = get_action_funcs[index]
    fit_func.total_cpu = total CPU capacity
    fit_func.total_mem = total memory capacity
    action = fit_func(state)

    2)
    from baseline_agent import get_fit_func
    fit_func = get_fit_func(index, total_cpu, total_mem)
    action = fit_func(state)

Output of `fit_func`: An integer (`action`) between 0 and (number of NUMAs - 1),
    representing the NUMA to place the virtual machine. For large VMs requiring
    resources from both NUMAs, the output corresponds to the server that shares
    the NUMA.

Input of `fit_func`: A dictionary named 'state' with the following keys:
    {'obs': obs, 'feat': feat, 'avail': avail}

    - `obs`: ndarray of shape (number of servers, 2, 2)
        The 2nd dimension represents the two NUMAs in a server, and the 3rd 
        dimension represents the CPU and memory resources for each NUMA.
        obs[j, k, l] indicates the remaining resources of the `l`th type 
        (CPU or memory) in the `k`th NUMA of the `j`th server.

    - `feat`: ndarray of shape (2, 2, 2)
        Indicates how resources in a server will be reduced after placing a VM.
        Examples:
            Small VM: [ ([1, 2], [0, 0]), ([0, 0], [1, 2]) ]
            Large VM: [ ([4, 8], [4, 8]), ([4, 8], [4, 8]) ]

    - `avail`: ndarray of shape (number of NUMAs)
        Indicates whether each NUMA is available for the latest VM request.
"""