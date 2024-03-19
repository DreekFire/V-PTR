from bdb import effective
import collections
import copy
from typing import Iterable, Optional

import gym
import jax
import numpy as np
from flax.core import frozen_dict
from gym.spaces import Box

from vptr.data.dataset import DatasetDict, _sample
from vptr.data.replay_buffer import ReplayBuffer


class MemoryEfficientReplayBufferParallel(ReplayBuffer):

    def __init__(self, observation_space: gym.Space, action_space: gym.Space,
                 capacity: int, num_devices=len(jax.devices())):

        pixel_obs_space = observation_space.spaces['pixels']
        self._num_stack = pixel_obs_space.shape[-1]
        self._unstacked_dim_size = pixel_obs_space.shape[-2]
        low = pixel_obs_space.low[..., 0]
        high = pixel_obs_space.high[..., 0]
        unstacked_pixel_obs_space = Box(low=low,
                                        high=high,
                                        dtype=pixel_obs_space.dtype)
        observation_space = copy.deepcopy(observation_space)
        observation_space.spaces['pixels'] = unstacked_pixel_obs_space

        self._first = True
        self._is_correct_index = np.full(capacity, False, dtype=bool)
        self._all_indices = np.arange(capacity, dtype=np.int32)

        next_observation_space_dict = copy.deepcopy(observation_space.spaces)
        next_observation_space_dict.pop('pixels')
        next_observation_space = gym.spaces.Dict(next_observation_space_dict)

        self.num_devices = num_devices

        super().__init__(observation_space,
                         action_space,
                         capacity,
                         next_observation_space=next_observation_space)

    def insert(self, data_dict: DatasetDict):
        if self._insert_index == 0 and self._capacity == len(
                self) and not self._first:
            indxs = np.arange(len(self) - self._num_stack, len(self))
            for indx in indxs:
                element = super().sample(1, indx=indx)
                self._is_correct_index[self._insert_index] = False
                super().insert(element)

        data_dict = data_dict.copy()
        data_dict['observations'] = data_dict['observations'].copy()
        data_dict['next_observations'] = data_dict['next_observations'].copy()

        obs_pixels = data_dict['observations'].pop('pixels')
        next_obs_pixels = data_dict['next_observations'].pop('pixels')

        if self._first:
            for i in range(self._num_stack):
                data_dict['observations']['pixels'] = obs_pixels[..., i]
                self._is_correct_index[self._insert_index] = False
                super().insert(data_dict)

        data_dict['observations']['pixels'] = next_obs_pixels[..., -1]

        self._first = data_dict['dones']

        self._is_correct_index[self._insert_index] = True
        super().insert(data_dict)

        for i in range(self._num_stack):
            indx = (self._insert_index + i) % len(self)
            self._is_correct_index[indx] = False

    def split_arr(self, arr):
        split_arrs = np.split(arr, self.num_devices)
        return np.concatenate([np.expand_dims(x, axis=0) for x in split_arrs], axis=0)

    def device_split(self, data):
        data = data.unfreeze()
        for key in data:
            if type(data[key]) is np.ndarray:
                data[key] = self.split_arr(data[key])
            elif type(data[key]) is dict:
                for subkey in data[key]:
                    data[key][subkey] = self.split_arr(data[key][subkey])
            else:
                raise NotImplementedError
        return frozen_dict.freeze(data)

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

    def sample(self,
               batch_size: int,
               keys: Optional[Iterable[str]] = None,
               indx: Optional[np.ndarray] = None) -> frozen_dict.FrozenDict:

        correct_indices = self._all_indices[self._is_correct_index]
        if indx is None:
            if hasattr(self.np_random, 'integers'):
                indx = self.np_random.integers(len(correct_indices),
                                               size=batch_size)
            else:
                indx = self.np_random.randint(len(correct_indices),
                                              size=batch_size)
        indx = correct_indices[indx]

        if keys is None:
            keys = self.dataset_dict.keys()
        else:
            raise ValueError()

        keys = list(keys)
        keys.remove('observations')

        batch = super().sample(batch_size, keys, indx)
        batch = batch.unfreeze()

        obs_keys = self.dataset_dict['observations'].keys()
        obs_keys = list(obs_keys)
        obs_keys.remove('pixels')

        batch['observations'] = {}
        for k in obs_keys:
            batch['observations'][k] = _sample(
                self.dataset_dict['observations'][k], indx)

        obs_pixels = self.dataset_dict['observations']['pixels']
        obs_pixels = np.lib.stride_tricks.sliding_window_view(obs_pixels,
                                                              self._num_stack +
                                                              1,
                                                              axis=0)
        obs_pixels = obs_pixels[indx - self._num_stack]
        batch['observations']['pixels'] = obs_pixels

        return frozen_dict.freeze(batch)



