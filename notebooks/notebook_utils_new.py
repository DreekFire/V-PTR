import os
import pickle as pkl
import pandas as pd
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import flax
from flax import linen as nn
from flax.training import checkpoints
import roboverse
from icecream import ic
import json
from gym.spaces import Box, Dict
import gym
import yaml
import glob
import sys
from jaxrl2.wrappers.reaching_reward_wrapper import ReachingReward
from jaxrl2.data import MemoryEfficientReplayBuffer, MemoryEfficientReplayBufferParallel, NaiveReplayBuffer, BetterReplayBuffer

from jaxrl2.utils.general_utils import AttrDict
import sys
import numpy as np

from collections import OrderedDict
import json
import datetime

from jaxrl2.agents import PixelBCLearner, PixelRewardLearner
from jaxrl2.wrappers.prev_action_wrapper import PrevActionStack
from jaxrl2.wrappers.state_wrapper import StateStack
from jaxrl2.agents.sac.sac_learner import SACLearner
from jaxrl2.agents.cql.cql_learner import CQLLearner
from jaxrl2.agents.sarsa import PixelSARSALearner
from jaxrl2.agents.cql.pixel_cql_learner import PixelCQLLearner
from jaxrl2.agents.cql_parallel_overall.pixel_cql_learner import PixelCQLParallelLearner
from jaxrl2.agents.cql_encodersep.pixel_cql_learner import PixelCQLLearnerEncoderSep
from jaxrl2.agents.cql_encodersep_parallel.pixel_cql_learner import PixelCQLLearnerEncoderSepParallel
from jaxrl2.wrappers.rescale_actions_wrapper import RescaleActions
from jaxrl2.wrappers.normalize_actions_wrapper import NormalizeActions
from examples.configs.dataset_config_real import *
from examples.configs.toykitchen_pickplace_dataset import *
from examples.configs.get_door_openclose_data import *

from gym.spaces import Dict
import sys

import gym
import numpy as np
from gym.spaces import Box

from jaxrl2.data.dataset import MixingReplayBuffer, MixingReplayBufferParallel
from jaxrl2.data.replay_buffer import ReplayBuffer

from jaxrl2.utils.wandb_logger import WandBLogger, create_exp_name
from jaxrl2.agents import PixelIQLLearner, PixelBCLearner
from jaxrl2.agents import IQLLearner
from jaxrl2.wrappers import FrameStack

from examples.train_utils import offline_training_loop, trajwise_alternating_training_loop, load_buffer, run_evals_only, insert_data_real


from jaxrl2.wrappers.reaching_reward_wrapper import ReachingReward
from jaxrl2.utils.general_utils import add_batch_dim
from jaxrl2.data.utils import get_task_id_mapping
from examples.train_utils import offline_training_loop, trajwise_alternating_training_loop, load_buffer, run_evals_only, insert_data_real

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        for k, v in list(self.items()):
            if isinstance(v, dict) and 'value' in v:
                self[k] = v['value']
        self.__dict__ = self

def fetch_task_id_mapping(variant):
    if variant.dataset == 'single_task':
        train_tasks = train_dataset_single_task
        eval_tasks = eval_dataset_single_task
    elif variant.dataset =='11tasks':
        print("using 11 tasks")
        train_tasks = train_dataset_11_task
        eval_tasks = eval_dataset_11_task
    elif variant.dataset == 'tk1_pickplace':
        train_tasks, eval_tasks = get_toykitchen1_pickplace()
    elif variant.dataset == 'tk2_pickplace':
        train_tasks, eval_tasks = get_toykitchen2_pickplace()
    elif variant.dataset == 'all_pickplace':
        train_tasks, eval_tasks = get_all_pickplace()
    elif variant.dataset == 'open_micro_single':
        train_tasks = train_dataset_single_task_openmicro
        eval_tasks = eval_dataset_single_task_openmicro
    elif variant.dataset == 'openclose_all' or variant.dataset == 'open_close':
        train_tasks, eval_tasks = get_openclose_all()
    elif variant.dataset == 'online_reaching_pixels':
        train_tasks = online_reaching_pixels
        eval_tasks = online_reaching_pixels_val
    elif variant.dataset == 'online_reaching_pixels_first100':
        train_tasks = online_reaching_pixels_first100
        eval_tasks = online_reaching_pixels_val_first100
    elif variant.dataset == 'toykitchen1_pickplace':
        train_tasks, eval_tasks = get_toykitchen1_pickplace()
    elif variant.dataset == 'toykitchen2_pickplace':
        train_tasks, eval_tasks = get_toykitchen2_pickplace()
    elif variant.dataset == 'all_pickplace':
        train_tasks, eval_tasks = get_all_pickplace()
    elif variant.dataset == 'all_pickplace_except_tk6':
        train_tasks, eval_tasks = get_all_pickplace_exclude_tk6()
    elif variant.dataset == 'all_pickplace_v1_except_tk6':
        train_tasks, eval_tasks = get_all_pickplace_exclude_tk6()
    elif variant.dataset == 'toykitchen2_pickplace_simpler':
        train_tasks, eval_tasks = get_toykitchen2_pickplace_cardboardfence_reversible_simple()
    elif variant.dataset == 'openclose_exclude_tk1':
        train_tasks, eval_tasks = get_openclose_exclude_tk1()
    elif variant.dataset == 'tk1_targetdomain_openmicro':
        train_tasks, eval_tasks = tk1_targetdomain_openmicro()
    elif variant.dataset == 'pickplace_and_faucet':
        train_tasks, eval_tasks = get_all_pickplace_and_faucet()
    elif variant.dataset == 'tk6pickplace_and_faucet':
        train_tasks, eval_tasks = get_tk6pickplace_and_faucet()
    elif variant.dataset == 'tk1and6openclose_and_faucet':
        train_tasks, eval_tasks = get_tk1and6openclose_and_faucet()
    else:
        raise ValueError('dataset not found! ' + variant.dataset)
    if variant.target_dataset != '':
        if variant.target_dataset == 'toykitchen2_pickplace_cardboardfence_reversible':
            target_train_tasks, target_eval_tasks = get_toykitchen2_pickplace_cardboardfence_reversible()
            # target_train_tasks, target_eval_tasks = get_toykitchen2_pickplace_cardboardfence_reversible_simple()
        elif variant.target_dataset == 'toykitchen2_pickplace_simpler':
            target_train_tasks, target_eval_tasks = get_toykitchen2_pickplace_cardboardfence_reversible_simple()
        elif variant.target_dataset == 'toykitchen6_pickplace_reversible':
            target_train_tasks, target_eval_tasks = get_toykitchen6_pickplace_reversible()
        elif variant.target_dataset == 'toykitchen1_pickplace_cardboardfence_reversible':
            target_train_tasks, target_eval_tasks = get_toykitchen1_pickplace_cardboardfence_reversible()
        elif variant.target_dataset == 'toykitchen2_sushi_targetdomain':
            target_train_tasks, target_eval_tasks = get_toykitchen2_sushi_targetdomain()
        elif variant.target_dataset == 'tk1_targetdomain_openmicro':
            target_train_tasks, target_eval_tasks = tk1_targetdomain_openmicro()
        elif variant.target_dataset == 'tk1_closemicro':
            target_train_tasks, target_eval_tasks = tk1_closemicro()
        elif variant.target_dataset == 'tk1_targetdomain_7_19_openmicro-19':
            target_train_tasks, target_eval_tasks = tk1_targetdomain_7_19_openmicro()
        elif variant.target_dataset == 'tk1_targetdomain_7_19_closemicro-19':
            target_train_tasks, target_eval_tasks = tk1_targetdomain_7_19_closemicro()
        elif variant.target_dataset == 'online_open_micro':
            target_train_tasks, target_eval_tasks = online_open_micro()
        elif variant.target_dataset == 'tk1_targetdomain_openmicro_2': # this is collected at another robot on 2022/09/09
            target_train_tasks, target_eval_tasks = tk1_targetdomain_openmicro_2()
        elif variant.target_dataset == 'tk1_targetdomain_closemicro_2': # this is collected at another robot on 2022/09/09
            target_train_tasks, target_eval_tasks = tk1_targetdomain_closemicro_2()
        elif variant.target_dataset == 'tk1_targetdomain_openmicro_2_5demos':
            target_train_tasks, target_eval_tasks = tk1_targetdomain_openmicro_2_few_demo(5)
        elif variant.target_dataset == 'tk1_targetdomain_openmicro_2_3demos':
            target_train_tasks, target_eval_tasks = tk1_targetdomain_openmicro_2_few_demo(3)
        elif variant.target_dataset == 'tk1_targetdomain_openmicro_2_1demos':
            target_train_tasks, target_eval_tasks = tk1_targetdomain_openmicro_2_few_demo(1)
        elif variant.target_dataset == 'tk1_targetdomain_openmicro_2_5demos_20negs':
            target_train_tasks, target_eval_tasks = tk1_targetdomain_openmicro_2_few_demo_with_negs(5)
        elif variant.target_dataset == 'tk1_targetdomain_openmicro_2_3demos_20negs':
            target_train_tasks, target_eval_tasks = tk1_targetdomain_openmicro_2_few_demo_with_negs(3)
        elif variant.target_dataset == 'tk1_targetdomain_openmicro_2_1demos_20negs':
            target_train_tasks, target_eval_tasks = tk1_targetdomain_openmicro_2_few_demo_with_negs(1)
        elif variant.target_dataset == 'tk1_targetdomain_openmicro_3': # this is collected at another robot on 2022/09/28
            num_demo = variant.get('num_demo', None)
            target_train_tasks, target_eval_tasks = tk1_targetdomain_openmicro_3(num_demo)
        elif variant.target_dataset == 'tk1_targetdomain_closemicro_3': # this is collected at another robot on 2022/09/28
            target_train_tasks, target_eval_tasks = tk1_targetdomain_closemicro_3()
        elif variant.target_dataset == 'tk1_targetdomain_openmicro_4': # this is collected at another robot on 2022/11/09
            num_demo = variant.get('num_demo', None)
            target_train_tasks, target_eval_tasks = tk1_targetdomain_openmicro_4(num_demo)
        elif variant.target_dataset == 'tk1_targetdomain_closemicro_4': # this is collected at another robot on 2022/11/09
            target_train_tasks, target_eval_tasks = tk1_targetdomain_closemicro_4()
        elif variant.target_dataset == 'tk1_targetdomain_openmicro_5': # this is collected at another robot on 2022/11/15
            target_train_tasks, target_eval_tasks = tk1_targetdomain_openmicro_5()
        elif variant.target_dataset == 'tk1_targetdomain_openmicro_6': # this is collected at another robot on 2022/11/16
            num_demo = variant.get('num_demo', None)
            target_train_tasks, target_eval_tasks = tk1_targetdomain_openmicro_6(num_demo)
        elif variant.target_dataset == 'tk1_targetdomain_closemicro_6': # this is collected at another robot on 2022/11/16
            target_train_tasks, target_eval_tasks = tk1_targetdomain_closemicro_6()
        elif variant.target_dataset == 'tk6_faucet_left': 
            target_train_tasks, target_eval_tasks = tk6_target_faucet_left()
        elif variant.target_dataset == 'tk6_faucet_right': 
            target_train_tasks, target_eval_tasks = tk6_target_faucet_right()
        elif variant.target_dataset == 'tk1_targetdomain_openmicro_8': # this is collected at another robot on 2023/02/22
            num_demo = variant.get('num_demo', None)
            target_train_tasks, target_eval_tasks = tk1_targetdomain_openmicro_8(num_demo)
        elif variant.target_dataset == 'tk1_targetdomain_closemicro_8': # tthis is collected at another robot on 2023/02/22
            target_train_tasks, target_eval_tasks = tk1_targetdomain_closemicro_8()
        elif variant.target_dataset == 'tk1_targetdomain_openmicro_idling': # this is collected at another robot on 2023/02/22
            num_demo = variant.get('num_demo', None)
            target_train_tasks, target_eval_tasks = tk1_targetdomain_openmicro_idling(num_demo)
        elif variant.target_dataset == 'tk1_targetdomain_openmicro_0419': # this is collected at another robot on 2023/04/19
            num_demo = variant.get('num_demo', None)
            target_train_tasks, target_eval_tasks = tk1_targetdomain_openmicro_0419(num_demo)
        elif variant.target_dataset == 'tk1_targetdomain_openmicro_0425': # this is collected at another robot on 2023/04/25 (half open)
            num_demo = variant.get('num_demo', None)
            target_train_tasks, target_eval_tasks = tk1_targetdomain_openmicro_0425(num_demo)
        elif variant.target_dataset == 'tk1_targetdomain_openmicro_idling_0425': # this is collected at another robot on 2023/04/25 (half open)
            num_demo = variant.get('num_demo', None)
            target_train_tasks, target_eval_tasks = tk1_targetdomain_openmicro_idling_0425(num_demo)
        elif variant.target_dataset == 'tk1_targetdomain_openmicro_0523': # this is collected on 2023/05/15
            num_demo = variant.get('num_demo', None)
            target_train_tasks, target_eval_tasks = tk1_targetdomain_openmicro_0523(num_demo)
        elif variant.target_dataset == 'tk1_targetdomain_openmicro_distractors_0523': # this is collected on 2023/05/15
            num_demo = variant.get('num_demo', None)
            target_train_tasks, target_eval_tasks = tk1_targetdomain_openmicro_distractors_0523(num_demo)
        elif variant.target_dataset == 'tk1_targetdomain_openmicro_combined_0523': # this is collected on 2023/05/15
            num_demo = variant.get('num_demo', None)
            target_train_tasks, target_eval_tasks = tk1_targetdomain_openmicro_combined_0523(num_demo)
        elif variant.target_dataset == 'tk1_targetdomain_closemicro_0523': # this is collected on 2023/05/15
            num_demo = variant.get('num_demo', None)
            target_train_tasks, target_eval_tasks = tk1_targetdomain_closemicro_0523(num_demo)
        elif variant.target_dataset == 'tk1_targetdomain_closemicro_distractors_0523': # this is collected on 2023/05/15
            num_demo = variant.get('num_demo', None)
            target_train_tasks, target_eval_tasks = tk1_targetdomain_closemicro_distractors_0523(num_demo)
        elif variant.target_dataset == 'tk1_targetdomain_closemicro_combined_0523': # this is collected on 2023/05/15
            num_demo = variant.get('num_demo', None)
            target_train_tasks, target_eval_tasks = tk1_targetdomain_closemicro_combined_0523(num_demo)
        elif variant.target_dataset == 'tk6_targetdomain_cucumberpot_0515': # this is collected on 2023/05/15
            num_demo = variant.get('num_demo', None)
            target_train_tasks, target_eval_tasks = tk6_targetdomain_cucumberpot_0515(num_demo)
        elif variant.target_dataset == 'tk6_targetdomain_cucumberpot_distractors_0515': # this is collected on 2023/05/15
            num_demo = variant.get('num_demo', None)
        elif variant.target_dataset == 'tk6_targetdomain_cucumberpot_combined_0515': # this is collected on 2023/05/15
            num_demo = variant.get('num_demo', None)
            target_train_tasks, target_eval_tasks = tk6_targetdomain_cucumberpot_combined_0515(num_demo)
        elif variant.target_dataset == 'tk6_targetdomain_knifepan_0515': # this is collected on 2023/05/15
            num_demo = variant.get('num_demo', None)
            target_train_tasks, target_eval_tasks = tk6_targetdomain_knifepan_0515(num_demo)
        elif variant.target_dataset == 'tk6_targetdomain_knifepan_distractors_0515': # this is collected on 2023/05/15
            num_demo = variant.get('num_demo', None)
        elif variant.target_dataset == 'tk6_targetdomain_knifepan_combined_0515': # this is collected on 2023/05/15
            num_demo = variant.get('num_demo', None)
            target_train_tasks, target_eval_tasks = tk6_targetdomain_knifepan_combined_0515(num_demo)
        elif variant.target_dataset == 'tk6_targetdomain_sweetpotatoplate_0515': # this is collected on 2023/05/15
            num_demo = variant.get('num_demo', None)
            target_train_tasks, target_eval_tasks = tk6_targetdomain_sweetpotatoplate_0515(num_demo)
        elif variant.target_dataset == 'tk6_targetdomain_sweetpotatoplate_distractors_0515': # this is collected on 2023/05/15
            num_demo = variant.get('num_demo', None)
            target_train_tasks, target_eval_tasks = tk6_targetdomain_sweetpotatoplate_distractors_0515(num_demo)
        elif variant.target_dataset == 'tk6_targetdomain_sweetpotatoplate_combined_0515': # this is collected on 2023/05/15
            num_demo = variant.get('num_demo', None)
            target_train_tasks, target_eval_tasks = tk6_targetdomain_sweetpotatoplate_combined_0515(num_demo)
        elif variant.target_dataset == 'tk6_targetdomain_croissantcolander_0515': # this is collected on 2023/05/15
            num_demo = variant.get('num_demo', None)
            target_train_tasks, target_eval_tasks = tk6_targetdomain_croissantcolander_0515(num_demo)
        elif variant.target_dataset == 'tk6_targetdomain_croissantcolander_distractors_0515': # this is collected on 2023/05/15
            num_demo = variant.get('num_demo', None)
            target_train_tasks, target_eval_tasks = tk6_targetdomain_croissantcolander_distractors_0515(num_demo)
        elif variant.target_dataset == 'tk6_targetdomain_croissantcolander_0530':
            num_demo = variant.get('num_demo', None)
            target_train_tasks, target_eval_tasks = tk6_targetdomain_croissantcolander_0530(num_demo)
        elif variant.target_dataset == 'tk6_targetdomain_croissantcolander_combined_0515':
            num_demo = variant.get('num_demo', None)
            target_train_tasks, target_eval_tasks = tk6_targetdomain_croissantcolander_combined_0515(num_demo)
        elif variant.target_dataset == 'tk6_targetdomain_anycolander_0515':
            num_demo = variant.get('num_demo', None)
            target_train_tasks, target_eval_tasks = tk6_targetdomain_anycolander_0515(num_demo)
        elif variant.target_dataset == 'tk6_targetdomain_croissantcolander_all':
            num_demo = variant.get('num_demo', None)
            target_train_tasks, target_eval_tasks = tk6_targetdomain_croissantcolander_all(num_demo)
        elif variant.target_dataset == 'tk6_targetdomain_colander_all':
            num_demo = variant.get('num_demo', None)
            target_train_tasks, target_eval_tasks = tk6_targetdomain_colander_all(num_demo)
        else:
            raise ValueError('target dataset not found! ' + variant.target_dataset)

    else:
        target_train_tasks = []
        target_eval_tasks = []
    variant.ti = ti = (2,3)
    task_id_mapping = get_task_id_mapping(train_tasks + target_train_tasks, ALIASING_DICT)
    print('task_id_mapping:', task_id_mapping)
    num_tasks = len(task_id_mapping.keys())
    variant.num_tasks = num_tasks
    variant.task_id_mapping = task_id_mapping
    print('using {} tasks'.format(num_tasks))
    #     task_id_mapping = get_task_id_mapping(train_tasks + target_train_tasks, index = ti if variant.cond_interfing else -3)
    variant.task_id_mapping = task_id_mapping
    variant.train_tasks = train_tasks
    variant.target_train_tasks = target_train_tasks
    variant.eval_tasks = eval_tasks
    variant.target_eval_tasks = target_eval_tasks

def load_config(config_path, remove_keys=[]):
    with open(config_path, "r") as stream:
        try:
            variant = AttrDict(yaml.safe_load(stream))
            for k in remove_keys:
                del variant[k]
            fetch_task_id_mapping(variant)
            return variant

            return AttrDict(variant)
        except yaml.YAMLError as exc:
            print(exc)
         
def load_config_json(config_path, remove_keys=[]):
    with open(config_path, "r") as stream:
        try:
            variant = AttrDict(json.load(stream))
            for k in remove_keys:
                del variant[k]
            if 'lang_embedding' not in variant:
                variant.lang_embedding = None
            if 'relabel_actions' not in variant:
                variant['relabel_actions'] = False
            if 'use_pixel_sep_actor' in variant['train_kwargs']:
                variant['train_kwargs']['include_state_actor'] = not variant['train_kwargs'].pop('use_pixel_sep_actor')
            if 'use_pixel_sep' in variant['train_kwargs']:
                variant['train_kwargs']['use_pixel_sep_critic'] = variant['train_kwargs'].pop('use_pixel_sep')
            if 'smooth_gripper' not in variant:
                variant['smooth_gripper'] = 1.0
            fetch_task_id_mapping(variant)
            return variant

            return AttrDict(variant)
        except yaml.YAMLError as exc:
            print(exc)
            
TARGET_POINT = np.array([0.58996923,  0.21808016, -0.24382344])  # for reaching computed as the mean over all states.

def _process_image(obs):
    obs = (obs * 255).astype(np.uint8)
    obs = np.reshape(obs, (3, 128, 128))
    return np.transpose(obs, (1, 2, 0))

class Roboverse(gym.ObservationWrapper):

    def __init__(self, variant, env, num_tasks=7):
        super().__init__(env)

        # Note that the previous action i6:30pms multiplied by FLAGS.frame_stack to
        # account for the ability to pass in multiple previous actions in the
        # system. This is the case especially when the number of previous actions
        # is supposed to be many, when using multiple past frames
        self.variant = variant
        obs_dict = {}
        if not variant.from_states:
            obs_dict['pixels'] = Box(low=0, high=255, shape=(128, 128, 3), dtype=np.uint8)
        if variant.add_states:
            obs_dict['state'] = Box(low=-100000, high=100000, shape=(10,), dtype=np.float32)
        if num_tasks > 1:
            obs_dict['task_id'] = Box(low=0, high=1, shape=(num_tasks,), dtype=np.float32)
        self.observation_space = Dict(obs_dict)

    def observation(self, observation):
        out_dict = {}
        if 'image' in observation:
            out_dict['pixels'] = _process_image(observation['image'])[None]
        if self.variant.add_states:
            out_dict['state'] = observation['state'][None]
        return out_dict

def roboverse_env(env_name, variant, time_limit=40, num_tasks=7):
    env = roboverse.make(env_name, transpose_image=True)
    if variant.reward_type != 'final_one':
        env = ReachingReward(env, TARGET_POINT, variant.reward_type)
    env = Roboverse(variant, env, num_tasks=num_tasks)
    if variant.obs_latency:
        env = obs_latency.ObsLatency(env, variant.obs_latency)
    if variant.add_prev_actions:
        if variant.frame_stack == 1:
            action_queuesize = 1
        else:
            action_queuesize = variant.frame_stack - 1
        env = PrevActionStack(env, action_queuesize)
    if variant.add_states:
        env = StateStack(env, variant.frame_stack)
    if not variant.from_states:
        env = FrameStack(env, variant.frame_stack)
    env = gym.wrappers.TimeLimit(env, time_limit)
    return env

class ResizingEncoder(nn.Module):
    encoder: nn.Module
    new_shape: tuple

    @nn.compact
    def __call__(self, observations):
        ic(observations.shape)
        if len(observations.shape) == 3:
            print('Adding batch dimension')
            obs = jnp.expand_dims(observations, 0)
        else:
            obs = observations
        obs = jax.image.resize(obs, (*obs.shape[:-3], *self.new_shape, 3), method='bicubic')
        output = self.encoder(obs)

        if len(observations.shape) == 3:
            print('Removing batch dimension')
            output = jnp.squeeze(output, 0)
        return output
    
def load_agent(variant, env, checkpoint_steps=None, transformer=False, checkpoint_path=None):
    variant.pretrained_encoder = None
    kwargs = variant.train_kwargs.copy()
    if kwargs.pop('cosine_decay', False):
        kwargs['decay_steps'] = variant.max_steps
    sample_obs = add_batch_dim(env.observation_space.sample())
    sample_action = add_batch_dim(env.action_space.sample())
    print('sample obs shapes', [(k, v.shape) for k, v in sample_obs.items()])


    if variant.from_states:
        if variant.algorithm == 'iql':
            agent = IQLLearner(variant.seed, sample_obs, sample_action
                               , **kwargs)
        elif variant.algorithm == 'sac':
            agent = SACLearner(variant.seed, env.observation_space,
                               env.action_space, **kwargs)
    else:
        if variant.algorithm == 'iql':
            agent = PixelIQLLearner(variant.seed, sample_obs,
                                sample_action, **kwargs)
        elif variant.algorithm == 'bc':
            agent = PixelBCLearner(variant.seed, sample_obs,
                                    sample_action, **kwargs)
        elif variant.algorithm == 'cql':
            agent = PixelCQLLearner(variant.seed, sample_obs, 
                                    sample_action, **kwargs)
        elif variant.algorithm == 'sarsa':
            agent = PixelSARSALearner(variant.seed, sample_obs, 
                                    sample_action, **kwargs)
        elif variant.algorithm == 'cql_parallel':
            agent = PixelCQLParallelLearner(variant.seed, sample_obs,
                                            sample_action, **kwargs)
        elif variant.algorithm == 'cql_encodersep':
            agent = PixelCQLLearnerEncoderSep(variant.seed, sample_obs,
                                            sample_action, **kwargs)
        elif variant.algorithm == 'cql_encodersep_parallel':
            agent = PixelCQLLearnerEncoderSepParallel(variant.seed, sample_obs,
                                            sample_action, **kwargs)
    
    if checkpoint_path is None:
        if checkpoint_steps is not None:
            if checkpoint_steps > 0:
                checkpoint_path = f'{variant.outputdir}/checkpoint{checkpoint_steps}'
            elif checkpoint_steps == 0 and variant.restore_path:
                checkpoint_path = variant.restore_path
        else:
            checkpoint_path = variant.outputdir
    print('Using checkpoint', checkpoint_path)
    agent.restore_checkpoint(checkpoint_path)
    return agent

def create_env(variant, env_name='PutBallintoBowl-v0', num_tasks=7):
    def wrap(env, time_limit=40):
        if variant.reward_type != 'final_one':
            env = ReachingReward(env, TARGET_POINT, variant.reward_type)
        env = Roboverse(variant, env, num_tasks=num_tasks)
        if variant.obs_latency:
            env = obs_latency.ObsLatency(env, variant.obs_latency)
        if variant.add_prev_actions:
            if variant.frame_stack == 1:
                action_queuesize = 1
            else:
                action_queuesize = variant.frame_stack - 1
            env = PrevActionStack(env, action_queuesize)
        if variant.add_states:
            env = StateStack(env, variant.frame_stack)
        if not variant.from_states:
            env = FrameStack(env, variant.frame_stack)
        env = gym.wrappers.TimeLimit(env, time_limit)
        return env
    
    time_limit=40
    extra_kwargs=dict()
    env = roboverse.make(env_name, transpose_image=True, **extra_kwargs)
    env = wrap(env, time_limit=time_limit)
    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=1)
    return env

import collections

def make_buffer_and_insert(env, replay_buffer_class, task_id_mapping, tasks, variant, 
                           split_pos_neg=False, reward_function=None,
                           num_traj_cuttoff=-1):
    pos_buffer_size = neg_buffer_size = buffer_size = 0
    all_trajs = []
    data_count_dict = {}
    for dataset_file in tasks:
        task_size, trajs = load_buffer(dataset_file, variant, ALIASING_DICT, multi_viewpoint=variant.multi_viewpoint,
                                       data_count_dict=data_count_dict, split_pos_neg=split_pos_neg,
                                       num_traj_cutoff=num_traj_cuttoff)
        if split_pos_neg:
            pos_size, neg_size = task_size
            pos_buffer_size += pos_size
            neg_buffer_size += neg_size
        else:
            buffer_size += task_size
        all_trajs.append(trajs)
    print('size ', buffer_size)
    if split_pos_neg:
        pos_buffer = replay_buffer_class(env.observation_space, env.action_space, pos_buffer_size)
        neg_buffer = replay_buffer_class(env.observation_space, env.action_space, neg_buffer_size)
        buffer = MixingReplayBuffer([pos_buffer, neg_buffer], 0.5)
    else:
        buffer = replay_buffer_class(env.observation_space, env.action_space, buffer_size)
    print('inserting data...')
    for trajs in all_trajs:
        if variant.multi_viewpoint:
            [insert_data_real(variant, buffer, trajs, variant.reward_type, task_id_mapping, env=env,
                              image_key='images' + str(i), target_point=TARGET_POINT, split_pos_neg=split_pos_neg, reward_function=reward_function) for i in range(3)]
        else:
            insert_data_real(variant, buffer, trajs, variant.reward_type, task_id_mapping, env=env,
                             image_key='images0', target_point=TARGET_POINT, split_pos_neg=split_pos_neg, reward_function=reward_function)
    return buffer, data_count_dict



def is_positive_sample(traj, i, variant, task_name):
    return i >= len(traj['observations']) - variant.num_final_reward_steps

def is_positive_traj(traj):
    return traj['rewards'][-1, 0] >= 1

def is_positive_traj_timestep(traj, i):
    return traj['rewards'][i, 0] >= 1

def insert_data(variant, replay_buffer, trajs, run_test=False, task_id_mapping=None, split_pos_neg=False, split_by_traj=False):
    if split_pos_neg:
        assert isinstance(replay_buffer, MixingReplayBuffer)
        pos_buffer, neg_buffer = replay_buffer.replay_buffers

    if split_by_traj:
        num_traj_pos = 0
        num_traj_neg = 0

    for traj_id, traj in enumerate(trajs):
        if variant.frame_stack == 1:
            action_queuesize = 1
        else:
            action_queuesize = variant.frame_stack - 1
        prev_actions = collections.deque(maxlen=action_queuesize)
        current_states = collections.deque(maxlen=variant.frame_stack)

        for i in range(action_queuesize):
            prev_action = np.zeros_like(traj['actions'][0])
            if run_test:
                prev_action[0] = -1
            prev_actions.append(prev_action)

        for i in range(variant.frame_stack):
            state = traj['observations'][0]['state']
            if run_test:
                state[0] = 0
            current_states.append(state)

        if split_by_traj:
            positive_traj = is_positive_traj(traj)
            if positive_traj:
                num_traj_pos += 1
            else:
                num_traj_neg += 1

        # first process rewards, masks and mc_returns
        masks = [] 
        for i in range(len(traj['observations'])):
            if variant.reward_type != 'final_one':
                reward = compute_distance_reward(traj['observations'][i]['state'][:3], TARGET_POINT, variant.reward_type)
                traj['rewards'][i] = reward
                masks.append(1.0)
            else:
                reward = traj['rewards'][i]
            
                def def_rew_func(x):
                    return x * variant.reward_scale + variant.reward_shift
                    
                if not hasattr(variant, 'reward_func_type') or variant.reward_func_type == 0:
                    rew_func = def_rew_func
                elif variant.reward_func_type == 1:
                    def rew_func(rew):
                        if rew < 0:
                            return rew * 10 # buffers where terminate when place incorrectly
                        if rew == 2:
                            return 10
                        else:
                            return -1.0
                elif variant.reward_func_type == 2:    
                    def rew_func(rew):
                        if rew == 0:
                            return -10
                        elif rew == 1:
                            return -5
                        elif rew == 2:
                            return 100
                        else:
                            assert False
                elif variant.reward_func_type == 3:    
                    def rew_func(rew):
                        if rew == 0:
                            return -20
                        elif rew == 1:
                            return -10
                        elif rew == 2:
                            return -5
                        elif rew == 3:
                            return 10
                        else:
                            assert False
                else:
                    rew_func = def_rew_func
                    
                variant.reward_func = rew_func            
                reward = rew_func(reward)
                traj['rewards'][i] = reward
                    
                if traj['rewards'][i] == 10:
                    masks.append(0.0)
                else:
                    masks.append(1.0)
        # calculate reward to go
        monte_carlo_return = calc_return_to_go(traj['rewards'].squeeze().tolist(), masks, variant.discount)
        
        if variant.get("online_bound_nstep_return", -1) > 1:
            nstep_return = calc_nstep_return(variant.online_bound_nstep_return, traj['rewards'].squeeze().tolist(), masks, variant.discount)
        else:
            nstep_return = [0] * len(masks)


# process obs, next_obs, actions and insert to buffer
        for i in range(len(traj['observations'])):
            if not split_by_traj:
                is_positive = is_positive_sample(traj, i, variant, task_name=traj['task_description'])
            else:
                is_positive = is_positive_traj_timestep(traj, i)

            obs = dict()
            if not variant.from_states:
                obs['pixels'] = traj['observations'][i]['image']
                obs['pixels'] = obs['pixels'][..., np.newaxis]
                if run_test:
                    obs['pixels'][0, 0] = i
            if variant.add_states:
                obs['state'] = np.stack(current_states, axis=-1)
            if variant.add_prev_actions:
                obs['prev_action'] = np.stack(prev_actions, axis=-1)

            action_i = traj['actions'][i]
            if run_test:
                action_i[0] = i
            prev_actions.append(action_i)

            current_state = traj['next_observations'][i]['state']
            if run_test:
                current_state[0] = i + 1
            current_states.append(current_state)  # do not delay state, therefore use i instead of i

            next_obs = dict()
            if not variant.from_states:
                next_obs['pixels'] = traj['next_observations'][i]['image']
                next_obs['pixels'] = next_obs['pixels'][..., np.newaxis]
                # if i == 0:
                #     obs['pixels'] = np.tile(obs['pixels'], [1, 1, 1, variant.frame_stack])
                if run_test:
                    next_obs['pixels'][0, 0] = i + 1
            if variant.add_states:
                next_obs['state'] = np.stack(current_states, axis=-1)
            if variant.add_prev_actions:
                next_obs['prev_action'] = np.stack(prev_actions, axis=-1)

            if task_id_mapping is not None:
                if len(task_id_mapping.keys()) > 1:
                    task_id = np.zeros((len(task_id_mapping.keys())))
                    task_id[task_id_mapping[traj['task_description']]] = 1
                    obs['task_id'] = task_id
                    next_obs['task_id'] = task_id

            if split_pos_neg:
                if positive_traj:
                    trajectory_id=pos_buffer._traj_counter
                else:
                    trajectory_id=neg_buffer._traj_counter
            else:
                trajectory_id=replay_buffer._traj_counter

            insert_dict =  dict(observations=obs,
                     actions=traj['actions'][i],
                     next_actions=traj['actions'][i+1] if len(traj['actions']) > i+1 else traj['actions'][i],
                     rewards=traj['rewards'][i],
                     next_observations=next_obs,
                     masks=masks[i],
                     dones=bool(i == len(traj['observations']) - 1),
                     trajectory_id=trajectory_id,
                     mc_returns=monte_carlo_return[i],
                     nstep_returns=nstep_return[i],
                     is_offline = 1
                     )

            if split_pos_neg:
                if positive_traj:
                    pos_buffer.insert(insert_dict)
                else:
                    neg_buffer.insert(insert_dict)
            else:
                replay_buffer.insert(insert_dict)

        if split_by_traj:
            if positive_traj:
                pos_buffer.increment_traj_counter()
            else:
                neg_buffer.increment_traj_counter()
        else:
            replay_buffer.increment_traj_counter()

    if split_by_traj:
        print('num traj pos', num_traj_pos)
        print('num traj neg', num_traj_neg)


RETURN_TO_GO_DICT = dict()

def calc_return_to_go(rewards, masks, gamma):
    global RETURN_TO_GO_DICT
    rewards_str = str(rewards) + str(masks) + str(gamma)
    if rewards_str in RETURN_TO_GO_DICT.keys():
        reward_to_go = RETURN_TO_GO_DICT[rewards_str]
    else:
        reward_to_go = [0]*len(rewards)
        prev_return = 0
        for i in range(len(rewards)):
            reward_to_go[-i-1] = rewards[-i-1] + gamma * prev_return * masks[-i-1]
            prev_return = reward_to_go[-i-1]
        RETURN_TO_GO_DICT[rewards_str] = reward_to_go
    return reward_to_go

NSTEP_RETURN_DICT = dict()
def calc_nstep_return(n, rewards, masks, gamma):
    global NSTEP_RETURN_DICT
    rewards_str = str(rewards) + str(masks) + str(gamma)
    if rewards_str in NSTEP_RETURN_DICT.keys():
        nstep_return = NSTEP_RETURN_DICT[rewards_str]
    else:
        nstep_return = [0]*len(rewards)
        prev_return = 0
        terminal_counts=1
        for i in range(len(rewards)):
            if i < n + terminal_counts - 1:
                nstep_return[-i-1] = rewards[-i-1] + gamma * prev_return * masks[-i-1]
            else:
                nstep_return[-i-1] = rewards[-i-1] + gamma * prev_return * masks[-i-1] - (gamma**n) * rewards[-i-1+n] * masks[-i-1]
            prev_return = nstep_return[-i-1]
            
            if i!= 0 and masks[-i-1] == 0:
                terminal_counts+=1
        NSTEP_RETURN_DICT[rewards_str] = nstep_return
    return nstep_return

def load_data(variant, tasks, env, num_traj_cutoff=-1):
    replay_buffer_class = BetterReplayBuffer
    split_pos_neg = variant.split_pos_neg if 'split_pos_neg' in variant else False
    target_replay_buffer_train, data_count_dict_target = make_buffer_and_insert(
                env, replay_buffer_class, variant.task_id_mapping, tasks, variant,
                split_pos_neg=split_pos_neg, reward_function=None,
                )
    print(data_count_dict_target)
    return target_replay_buffer_train

from functools import partial
from collections import defaultdict
import numpy as np
import jax.numpy as jnp
import tqdm

def convert_obs(
    obs, 
    image_key='image', 
    state_key='state', 
    consolidate_state=False, 
    remove_consolidate_state=False
):
    assert image_key in obs and state_key in obs, f"image_key: {image_key} and state_key: {state_key} must be in obs keys: {obs.keys()}"
    agent_obs = dict()
    if obs[image_key].shape[-1] == 1:
        agent_obs['pixels'] = obs[image_key].squeeze(-1)
    else:
        agent_obs['pixels'] = obs[image_key]
    
    if agent_obs['pixels'].ndim == 3:
        agent_obs['pixels'] = jnp.expand_dims(agent_obs['image'], 0)
        
    if consolidate_state:
        state_keys = [k for k in obs if image_key not in k]
        agent_obs['state'] = jnp.concatenate([obs.pop(k) for k in state_keys], axis=1)
    elif remove_consolidate_state:
        state_keys = [k for k in obs if image_key not in k and k != state_key]
        agent_obs['state'] = jnp.concatenate([obs.pop(k) for k in state_keys], axis=1)
    else:
        agent_obs['state'] = obs[state_key]
    
    if agent_obs['state'].ndim == 1:
        agent_obs['state'] = jnp.expand_dims(agent_obs['state'], 0)   
        if agent_obs['state'].shape[-1] == 9:
            new_shape = list(agent_obs['state'].shape)
            new_shape[-1] = 70
            agent_obs['state'] = jnp.concatenate([agent_obs['state'], jnp.zeros(new_shape)], axis=1)
    return agent_obs

def rollout(
    agent, 
    env, 
    max_steps, 
    which_tid=0, 
    consolidate_state=False, 
    remove_consolidate_state=False, 
    add_task_id=True,
    image_key='image',
    state_key='state',
):
    """Rollout the agent in the environment for max_steps.

    Args:
        agent (Agent): Agent to rollout.
        env (gym.Env): Environment to rollout in.
        max_steps (int): Maximum number of steps to rollout.

    Returns:
        dict: Rollout results.
    """
    if add_task_id:
        task_id = np.zeros_like(env.observation_space.spaces['task_id'].sample())
        task_id[which_tid] = 1
        task_id = task_id.astype(np.float32)
        task_id = jnp.array(task_id)
        task_id = jnp.expand_dims(task_id, (0, 2))
    
    obs = env.reset()
    
    if 'task_id' not in obs and add_task_id:
        obs['task_id'] = task_id
            
    done = False
    total_reward = 0
    
    all_obs, all_actions, all_rewards, all_dones, all_infos = [], [], [], [], []
    for step in range(max_steps):
        obs = convert_obs(
            obs, 
            image_key=image_key, 
            state_key=state_key, 
            consolidate_state=consolidate_state, 
            remove_consolidate_state=remove_consolidate_state
        )
        
        action = agent.eval_actions(obs)
#         action = agent_gd_actions(agent, obs)
        next_obs, reward, done, info = env.step(action.squeeze())
        
        if 'task_id' not in next_obs and add_task_id:
            next_obs['task_id'] = task_id
        
        all_obs.append(obs)
        all_actions.append(action)
        all_rewards.append(reward)
        all_infos.append(info)
        all_dones.append(done)
        
        total_reward += reward
        if done:
            break
        obs = next_obs
        
    #NOTE: the observation at the end of the rollout is added to the list of observations
    try:
        obs = convert_obs(
            obs, 
            image_key=image_key, 
            state_key=state_key, 
            consolidate_state=consolidate_state, 
            remove_consolidate_state=remove_consolidate_state
        )
        all_obs.append(obs)
    except Exception as e:
        print(e)
        all_obs.append(obs)
    
    return dict(
        observations=all_obs,
        actions=all_actions,
        rewards=all_rewards,
        dones=all_dones,
        infos=all_infos,
        steps=step,
    )
    
    
def get_rollouts(
    agent, 
    env, 
    num_rollouts=10, 
    max_steps=60, 
    add_task_id=True, 
    which_tid=0,
    image_key='image',
    state_key='state',
    consolidate_state=False, 
    remove_consolidate_state=False,
):
    rollouts = defaultdict(list)
        
    for _ in tqdm.tqdm(range(num_rollouts), leave=False):
        rll = rollout(
            agent, 
            env, 
            max_steps, 
            add_task_id=add_task_id, 
            which_tid=which_tid, 
            image_key=image_key, 
            state_key=state_key,
            consolidate_state=consolidate_state, 
            remove_consolidate_state=remove_consolidate_state,
        )
        for k, v in rll.items():
            rollouts[k].append(v)
    
    return dict(rollouts) # return a dict of lists


    

def create_env(variant):
    class DummyEnv():
        def __init__(self):
            super().__init__()
            obs_dict = dict()
            if not variant.from_states:
                obs_dict['pixels'] = Box(low=0, high=255, shape=(128, 128, 3), dtype=np.uint8)
            if variant.add_states:
                obs_dict['state'] = Box(low=-100000, high=100000, shape=(7,), dtype=np.float32)
            if variant.num_tasks > 1:
                obs_dict['task_id'] = Box(low=0, high=1, shape=(variant.num_tasks,), dtype=np.float32)
            self.observation_space = Dict(obs_dict)
            self.spec = None
            self.action_space = Box(
                np.asarray([-0.05, -0.05, -0.05, -0.25, -0.25, -0.25, 0.]),
                np.asarray([0.05, 0.05, 0.05, 0.25, 0.25, 0.25, 1.0]),
                dtype=np.float32)

        def seed(self, seed):
            pass

    def wrap(env):
        assert not (variant.normalize_actions and variant.rescale_actions)
        if variant.reward_type == 'dense':
            env = ReachingReward(env, TARGET_POINT, variant.reward_type)
        if variant.add_prev_actions:
            # Only do this to the extent that there is one less frame to be
            # stacked, since this only concats the previous actions, not the
            # current action....
            if variant.frame_stack == 1:
                num_action_stack = 1
            else:
                num_action_stack = variant.frame_stack - 1
            env = PrevActionStack(env, num_action_stack)
        if variant.rescale_actions:
            print ('Rescaling actions in environment..............')
            env = RescaleActions(env)
        elif variant.normalize_actions:
            print ('Normalizing actions in environment.............')
            env = NormalizeActions(env)
        if variant.add_states:
            env = StateStack(env, variant.frame_stack)
        if not variant.from_states:
            env = FrameStack(env, variant.frame_stack)
        env = gym.wrappers.TimeLimit(env, variant.episode_timelimit)
        return env
    return wrap(DummyEnv())



from matplotlib import pyplot as plt
from matplotlib import animation
from IPython.display import HTML, display
def display_video(video):
    fig = plt.figure()
    im = plt.imshow(video[0,:,:,:])

    plt.close() # this is required to not display the generated image

    def init():
        im.set_data(video[0,:,:,:])

    def animate(i):
        im.set_data(video[i,:,:,:])
        return im

    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=video.shape[0],
                                interval=50)
    display(HTML(anim.to_html5_video()))