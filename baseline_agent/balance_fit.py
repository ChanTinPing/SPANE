''' Large VMs use the First Fit strategy, while small VMs are placed in the NUMA with the largest utilization difference (if utilization is equal, fall back to First Fit). '''
import numpy as np

def fit_one_env(obs, feat, avail, total_cpu, total_mem):  
    server_num = np.shape(obs)[0]
    # Determine whether the VM is split across NUMAs
    request_is_split = feat[2]
    
    # Normalize CPU and memory resources (scale `obs` by total_cpu and total_mem)
    obs[:, :, 0] = obs[:, :, 0].astype(float) / total_cpu
    obs[:, :, 1] = obs[:, :, 1].astype(float) / total_mem
    # Calculate the remaining resource rate for each NUMA (minimum of CPU and memory resources per NUMA)
    # Reshape `obs` to (n, 2) and take the smaller value
    numa_remain_resource = np.min(obs, axis=2)

    if request_is_split:
        # First Fit strategy
        action = np.where(avail == 1)[0][0]
    else:
        # 1. Calculate the resource difference between the two NUMAs in each server
        # Check which NUMAs or servers can accommodate the VM
        numa_avail = avail.reshape(server_num, 2)
        server_avail = np.max(numa_avail, axis=1)  # If the maximum value is 1, this server has at least one available NUMA
        # Compute the resource difference between the two NUMAs for each server. For unavailable servers, set the difference to 0
        differences = np.abs(numa_remain_resource[:, 0] - numa_remain_resource[:, 1]) * server_avail 

        # 2. Make an action decision
        if np.all(differences == 0):
            # If all differences are 0, it means the available servers have equal resource utilization differences.
            # In this case, select the first available NUMA
            action = np.where(numa_avail.ravel() == 1)[0][0]
        else: 
            # Otherwise, choose the server with the largest resource difference
            server_max_diff = np.argmax(differences)  # Find the server with the largest resource difference
            numa_action = np.argmax(numa_remain_resource[server_max_diff])  # Select the NUMA with more resources in that server
            action = (2 * server_max_diff) + numa_action
            if avail[action] == 0:  # It is possible that the NUMA with fewer resources is available, but the NUMA with more resources is not
                # In such cases, switch the action to the other NUMA
                numa_action = 1 - numa_action  # Switch to the other NUMA
                action = (2 * server_max_diff) + numa_action

    return action

def balance_fit(state):
    obs = state["obs"].copy()
    feat = state["feat"].copy()
    avail = state["avail"].copy()
    action = fit_one_env(obs, feat, avail, balance_fit.total_cpu, balance_fit.total_mem)
    return action
