''' Select the NUMA/server with the smallest remaining resources (highest utilization) for placing small/large VMs.
    For NUMA, resources are determined as the minimum of the two resource types (CPU, memory).
    For servers, resources are calculated as the average of the two NUMAs. '''
import numpy as np

def request_split(feat):
    # Determine if the VM request is split across NUMAs (large VM)
    if np.allclose(feat[0][0], feat[0][1]):
        return True
    else:
        return False

# Select the NUMA/server with the highest utilization
def best_fit_one_env(obs, feat, avail, total_cpu, total_mem):  
    server_num = np.shape(obs)[0]
    # Check if the VM request is split across NUMAs
    request_is_split = request_split(feat)
    
    # Normalize the remaining resources for CPU and memory (scale `obs` by total_cpu and total_mem)
    obs[:, :, 0] = obs[:, :, 0].astype(float) / total_cpu
    obs[:, :, 1] = obs[:, :, 1].astype(float) / total_mem
    # Ignore infeasible NUMAs (set the corresponding third dimension in `obs` to a large value based on `avail`)
    avail = avail.reshape(server_num, 2)
    obs[avail == 0] = 2

    # Compute the remaining resource rate for each NUMA (minimum of CPU and memory resources per NUMA)
    # Reshape `obs` to (n, 2) and take the smaller value
    obs_min = np.min(obs, axis=2)

    if request_is_split:
        # Compute the remaining resource rate for each server (average of the two NUMAs)
        # Find the server with the smallest remaining resources (sum the rows and find the index of the minimum value)
        row_sums = np.sum(obs_min, axis=1)  # For comparison, summing is equivalent to averaging
        action = 2 * np.argmin(row_sums)
    else:
        # Find the NUMA with the smallest remaining resources (locate the index of the minimum value in `obs_min`)
        action = np.argmin(obs_min.ravel())  # Use ravel() to flatten `obs_min` into a 1D array

    return action

def best_fit(state):
    obs = state["obs"].copy()
    feat = state["feat"].copy()
    avail = state["avail"].copy()
    action = best_fit_one_env(obs, feat, avail, best_fit.total_cpu, best_fit.total_mem)
    return action
