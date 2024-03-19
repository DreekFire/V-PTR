from os import terminal_size
from typing import Union
from typing import Iterable, Optional
import jax 

import gym
import gym.spaces
import numpy as np

import copy

from vptr.data.dataset import Dataset, DatasetDict
import collections
from flax.core import frozen_dict

def _init_replay_dict(obs_space: gym.Space,
                      capacity: int) -> Union[np.ndarray, DatasetDict]:
    if isinstance(obs_space, gym.spaces.Box):
        return np.empty((capacity, *obs_space.shape), dtype=obs_space.dtype)
    elif isinstance(obs_space, gym.spaces.Dict):
        data_dict = {}
        for k, v in obs_space.spaces.items():
            data_dict[k] = _init_replay_dict(v, capacity)
        return data_dict
    else:
        raise TypeError()



class BetterReplayBuffer(Dataset):
    
    def __init__(self, observation_space: gym.Space, action_space: gym.Space, capacity: int, goal_conditioned: bool=False):
        self.observation_space = observation_space
        self.action_space = action_space
        self.capacity = capacity
        self.goal_conditioned = goal_conditioned

        print("making replay buffer of capacity ", self.capacity)

        observations = _init_replay_dict(self.observation_space, self.capacity)
        actions = np.empty((self.capacity, *self.action_space.shape), dtype=self.action_space.dtype)
        next_actions = np.empty((self.capacity, *self.action_space.shape), dtype=self.action_space.dtype)
        rewards = np.empty((self.capacity, ), dtype=np.float32)
        masks = np.empty((self.capacity, ), dtype=np.float32)
        trajectory_id = np.empty((self.capacity,), dtype=np.float32)
        dones = np.empty((self.capacity,), dtype=np.float32)
        mc_returns = np.empty((self.capacity, ), dtype=np.float32)


        self.data = {
            'observations': observations,
            'actions': actions,
            'next_actions': next_actions,
            'rewards': rewards,
            'masks': masks,
            'trajectory_id': trajectory_id,
            'dones': dones,
            'mc_returns': mc_returns
        }

        self.tasks = []

        self.size = 0
        self._traj_counter = 0
        self._start = 0
        self.traj_bounds = []
        self.streaming_buffer_size = None # this is for streaming the online data

    def __len__(self) -> int:
        return self.size

    def length(self) -> int:
        return self.size

    def increment_traj_counter(self):
        self.traj_bounds.append((self._start, self.size)) # [start, end)
        self._start = self.size
        self._traj_counter += 1

    def add_task(self, task):
        self.tasks.append(task)
    
    def get_trajs(self, traj_inds):
        trajs = []
        traj_inds = np.int64(traj_inds)
        for traj_ind in traj_inds:
            traj = {
                'observations': dict(),
                'next_observations': dict(),
            }

            start, end = self.traj_bounds[traj_ind]
            buffer_inds = np.arange(start, end, dtype=np.uint64)
            next_buffer_inds = np.minimum(buffer_inds + 1, self.size - 1)

            for k in self.data['observations']:
                traj['observations'][k] = self.data['observations'][k][buffer_inds]
                traj['next_observations'][k] = self.data['observations'][k][next_buffer_inds]
            for traj_k in ('actions', 'rewards', 'masks', 'mc_returns', 'trajectory_id'):
                traj[traj_k] = self.data[traj_k][buffer_inds]
            traj['terminals'] = 1 - traj['masks']
            traj['label'] = self.tasks[int(traj['trajectory_id'][0])]
            trajs.append(traj)
            
        return trajs
    
    def get_random_trajs(self, num_trajs: int):
        # traj_inds = np.random.choice(list(range(self._traj_counter)), num_trajs, replace=False)
        traj_inds = np.random.randint(0, self._traj_counter, num_trajs)
        return self.get_trajs(traj_inds)
        
    def insert(self, data_dict: DatasetDict, tasks: Optional[list] = None):
        if self.size == self.capacity:
            # Double the capacity
            observations = _init_replay_dict(self.observation_space, self.capacity)
            actions = np.empty((self.capacity, *self.action_space.shape), dtype=self.action_space.dtype)
            next_actions = np.empty((self.capacity, *self.action_space.shape), dtype=self.action_space.dtype)
            rewards = np.empty((self.capacity, ), dtype=np.float32)
            masks = np.empty((self.capacity, ), dtype=np.float32)
            mc_returns = np.empty((self.capacity, ), dtype=np.float32)
            trajectory_id = np.empty((self.capacity,), dtype=np.float32)
            dones = np.empty((self.capacity,), dtype=np.float32)

            data_new = {
                'observations': observations,
                'actions': actions,
                'next_actions': next_actions,
                'rewards': rewards,
                'masks': masks,
                'mc_returns': mc_returns,
                'trajectory_id': trajectory_id,
                'dones': dones,
            }

            for x in self.data:
                if isinstance(self.data[x], np.ndarray):
                    self.data[x] = np.concatenate((self.data[x], data_new[x]), axis=0)
                elif isinstance(self.data[x], dict):
                    for y in self.data[x]:
                        self.data[x][y] = np.concatenate((self.data[x][y], data_new[x][y]), axis=0)
                else:
                    raise TypeError()
            self.capacity *= 2


        for x in data_dict:
            if x in self.data:
                if isinstance(data_dict[x], dict):
                    for y in data_dict[x]:
                        self.data[x][y][self.size] = data_dict[x][y]
                else:                        
                    self.data[x][self.size] = data_dict[x]
        self.size += 1
    
    def compute_action_stats(self):
        actions = self.data['actions']
        return {'mean': actions.mean(axis=0), 'std': actions.std(axis=0)}

    def normalize_actions(self, action_stats):
        # do not normalize gripper dimension (last dimension)
        copy.deepcopy(action_stats)
        action_stats['mean'][-1] = 0
        action_stats['std'][-1] = 1
        self.data['actions'] = (self.data['actions'] - action_stats['mean']) / action_stats['std']
        self.data['next_actions'] = (self.data['next_actions'] - action_stats['mean']) / action_stats['std']

    def sample(self, batch_size: int, keys: Optional[Iterable[str]] = None, indx: Optional[np.ndarray] = None) -> frozen_dict.FrozenDict:
        if self.streaming_buffer_size:
            indices = np.random.randint(0, self.streaming_buffer_size, batch_size)
        else:
            indices = np.random.randint(0, self.size, batch_size)
        next_indices = np.minimum(indices + 1, self.size - 1)

        data_dict = {}
        for x in self.data:
            if isinstance(self.data[x], np.ndarray):
                data_dict[x] = self.data[x][indices]
                if x == 'observations':
                    data_dict['next_observations'] = self.data[x][next_indices]
            elif isinstance(self.data[x], dict):
                data_dict[x] = {}
                if x == 'observations':
                    data_dict['next_observations'] = {}
                for y in self.data[x]:
                    data_dict[x][y] = self.data[x][y][indices]
                    if x == 'observations':
                        data_dict['next_observations'][y] = self.data[x][y][next_indices]
            else:
                raise TypeError()
            
        if self.goal_conditioned:
            # todo: allow variable positive ratio
            give_reward_idxs = np.random.binomial(1, 0.1, size=batch_size) == 1
            goal_idxs = np.empty_like(indices)
            goal_idxs[give_reward_idxs] = indices[give_reward_idxs] + 1
            neg_traj_idxs = data_dict['trajectory_ids'][~give_reward_idxs]
            ranges = np.array([bound[1] for bound in self.traj_bounds[neg_traj_idxs]])
            goal_idxs[~give_reward_idxs] = np.random.randint(indices, ranges + 1, size=ranges.shape)

            data_dict['goals'] = self.data['observations'][goal_idxs]
        
        return frozen_dict.freeze(data_dict)

    def get_iterator(self, batch_size: int, keys: Optional[Iterable[str]] = None, indx: Optional[np.ndarray] = None, queue_size: int = 2):
        # See https://flax.readthedocs.io/en/latest/_modules/flax/jax_utils.html#prefetch_to_device
        # queue_size = 2 should be ok for one GPU.

        queue = collections.deque()

        def enqueue(n):
            for _ in range(n):
                data = self.sample(batch_size, keys, indx)
                queue.append(jax.device_put(data))

        enqueue(queue_size)
        while queue:
            yield queue.popleft()
            enqueue(1)


    def save(self, filename):
        save_dict = dict(
            data=self.data,
            size = self.size,
            _traj_counter = self._traj_counter,
            _start=self._start,
            traj_bounds=self.traj_bounds
        )
        np.save(filename, np.array([save_dict]))


    def restore(self, filename):
        save_dict = np.load(filename, allow_pickle=True)[0]
        # todo test this:
        self.data = save_dict['data']
        self.size = save_dict['size']
        self._traj_counter = save_dict['_traj_counter']
        self._start = save_dict['_start']
        self.traj_bounds = save_dict['traj_bounds']


            
class BetterReplayBufferParallel(BetterReplayBuffer):
    """
    Implements naive buffer with parallelism
    """
    def __init__(self, observation_space: gym.Space, action_space: gym.Space,
                 capacity: int, num_devices=len(jax.devices())):
        self.num_devices = num_devices
        super().__init__(observation_space=observation_space,
                         action_space=action_space,
                         capacity=capacity)
        
    def get_iterator(self,
                     batch_size: int,
                     keys: Optional[Iterable[str]] = None,
                     indx: Optional[np.ndarray] = None,
                     queue_size: int = 2):
        # See https://flax.readthedocs.io/en/latest/_modules/flax/jax_utils.html#prefetch_to_device
        # queue_size = 2 should be ok for one GPU.

        queue = collections.deque()

        def enqueue(n):
            assert batch_size % self.num_devices == 0
            effective_batch_size = batch_size // self.num_devices
            for _ in range(n):
                data = [self.sample(effective_batch_size, keys, indx) for _ in range(self.num_devices)]   
                queue.append(jax.device_put_sharded(data, jax.devices()))

        enqueue(queue_size)
        while queue:
            yield queue.popleft()
            enqueue(1)
