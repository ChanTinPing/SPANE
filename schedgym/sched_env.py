import pandas as pd
import numpy as np
import gymnasium as gym
from queue import PriorityQueue

def getData(path, double_thr=1e10):
    csv_data = pd.read_csv(path)
    data = csv_data.to_dict('records')
    for item in data:
        item['at'] = int(item['at'])
        item['lt'] = int(item['lt'])
        item['is_double'] = int(item['mem'] > double_thr)
    return data


class Cluster:
    def __init__(self, N, cpu, mem):
        self.N = N
        self.cpu = cpu
        self.mem = mem
        self.reset()

    def reset(self):
        # Initialize the resource matrix with shape (N, 2, 2)
        # The first dimension represents servers, the second represents NUMA nodes, 
        # and the third represents resource types (CPU, MEM)
        self.resources = np.full((self.N, 2, 2), [self.cpu, self.mem], dtype=float)
        # Use a dictionary to store all VMs, with keys as UUIDs and values as (server, numa, is_double)
        self.stored_vms = {}

    def handle(self, action, request):  # Install VM
        server, numa = action // 2, action % 2
        if request['is_double']:
            new_resources = self.resources[server] - np.array([request['cpu']/2, request['mem']/2])
            if np.any(new_resources < 0):
                raise ValueError("Insufficient resources for allocation")
            self.resources[server] = new_resources
        else:
            new_resources = self.resources[server, numa] - np.array([request['cpu'], request['mem']])
            if np.any(new_resources < 0):
                raise ValueError("Insufficient resources for allocation")
            self.resources[server, numa] = new_resources
        # Store VM information
        self.stored_vms[request['vmid']] = (server, numa, request['is_double'])

    def check_usable(self, request): 
        req = np.array([request['cpu'], request['mem']]).astype(float)
        if request['is_double']:
            req /= 2
            return np.all(np.all(self.resources >= req, axis=2), axis=1).repeat(2)
        else:
            return np.all(self.resources >= req, axis=2).reshape(-1)

    def delete(self, request): 
        if request['vmid'] not in self.stored_vms:
            raise ValueError("Delete unexist VM")

        server, numa, is_double = self.stored_vms.pop(request['vmid'])
        if is_double:
            self.resources[server] += np.array([request['cpu']/2, request['mem']/2])
        else:
            self.resources[server, numa] += np.array([request['cpu'], request['mem']])

    def describe(self):
        return self.resources

    def balance_score(self):
        numa_diffs = np.abs(self.resources[:, 0] - self.resources[:, 1])
        return -np.sum(numa_diffs / np.array([self.cpu, self.mem]))


class SchedEnv(gym.Env):
    def __init__(self, N, cpu, mem, data_path, double_thr=10, reward_type='basic', reward_weight=0):
        super(SchedEnv, self).__init__()
        self.N = N
        self.cpu = cpu
        self.mem = mem
        self.cluster = Cluster(N, cpu, mem)
        self.data = getData(data_path, double_thr)
        self.reward_type = reward_type
        self.reward_weight = reward_weight
        self.total_wait_time = 0

    def reset(self, index, N_vm=0, exceed_vm=0):
        # Reset the cluster and environment states
        self.cluster.reset()
        self.init_index = index
        self.index = index
        self.N_vm = N_vm
        self.exceed_vm = exceed_vm
        
        # Ensure only one of N_vm or exceed_vm is set
        if N_vm > 0 and exceed_vm > 0:
            raise ValueError("Only one of N_vm or exceed_vm can be set")
        if N_vm > 0:
            if ((self.index + self.N_vm) >= len(self.data)-1):
                raise ValueError("index + N_vm exceeds the total data length")
        elif exceed_vm > 0:
            self.first_unplaceable_encountered = False
            self.vms_after_unplaceable = 0 
            
        self.t = self.data[self.index]['at']  # Current time
        self.cnt = 0  # Number of created VMs
        self.vm_end_times = PriorityQueue()
        self.bal_score = 0  # Balance score for the cluster
        self.total_wait_time = 0  # Total wait time of VMs   
        return self.get_input()

    def step(self, action):
        # Validate the chosen action
        vm = self.data[self.index]
        if (self.cluster.check_usable(vm)[action] == 0):
            raise ValueError("Agent selected an unavailable NUMA")
        
        # Install the VM
        vm['st'] = self.t
        self.cluster.handle(action, vm)
        end_time = self.t + vm['lt']
        self.vm_end_times.put((end_time, self.index))
        self.cnt += 1
        self.index += 1
        
        # Calculate the wait time
        wait_time = self.t - vm['at']
        self.total_wait_time += wait_time

        # Calculate the reward
        reward = - np.log1p(wait_time)   # Advanced version of log(1 + wait_time)
        if (self.reward_type == 'balance'):
            bal_score_current = self.cluster.balance_score()
            balance_reward = bal_score_current - self.bal_score
            reward += self.reward_weight * balance_reward
            self.bal_score = bal_score_current

        # Check if the environment should terminate
        if self.index >= len(self.data) - 1:
            done = True
        else:
            if self.N_vm > 0:
                done = (self.cnt >= self.N_vm)
            elif self.exceed_vm > 0:
                if not self.first_unplaceable_encountered:
                    next_vm = self.data[self.index]
                    if not np.any(self.cluster.check_usable(next_vm)):
                        self.first_unplaceable_encountered = True
                if self.first_unplaceable_encountered:
                    self.vms_after_unplaceable += 1
                    done = (self.vms_after_unplaceable >= self.exceed_vm)
                else:
                    done = False

        # Advance to the next state, ensuring at least one NUMA is available
        if not done:  self._advance_time()  
        next_state = self.get_input()

        return next_state, reward, done

    def _advance_time(self):
        next_vm = self.data[self.index]
        next_vm_time = next_vm['at']

        # Handle all VMs that finish before the next VM arrives
        while (not self.vm_end_times.empty()) and (self.vm_end_times.queue[0][0] <= next_vm_time):
            end_time, vmid = self.vm_end_times.get()
            self.t = end_time
            ended_vm = self.data[vmid]
            self.cluster.delete(ended_vm)

        # Update time if the next VM can be placed
        if np.any(self.cluster.check_usable(next_vm)):
            self.t = max(self.t, next_vm_time)
            return

        # Advance time to the next VM's end time if it cannot be placed
        while True:
            next_end_time, vmid = self.vm_end_times.queue[0]
            self.t = next_end_time
            
            # Handle all VMs that finish at this time
            while (not self.vm_end_times.empty()) and (self.vm_end_times.queue[0][0] == next_end_time):
                _, vmid = self.vm_end_times.get()
                ended_vm = self.data[vmid]
                self.cluster.delete(ended_vm)

            # Check if the next VM can now be placed
            if np.any(self.cluster.check_usable(next_vm)):
                return

    def get_input(self):
        # Return the input features for the agent
        request = self.data[self.index]
        return {
            'obs': self.cluster.describe(),
            'feat': np.array([request['cpu'], request['mem'], request['is_double']]),
            'avail': self.cluster.check_usable(request)
        }
    
    def get_attr(self, attr_name):
        request = self.data[self.index]
        if attr_name == 'req':
            return np.array([request['cpu'], request['mem'], request['is_double']])
        elif attr_name == 'avail':  
            return self.cluster.check_usable(request)
        elif attr_name == 'obs':
            return self.cluster.describe()
        elif attr_name == 'full':
            return request
        elif attr_name == 'total_wait_time':  # 新增：返回总等待时间
            return self.total_wait_time
        elif attr_name == 'length_vm':
            return (self.index - self.init_index)
        return None
