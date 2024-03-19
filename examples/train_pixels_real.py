#! /usr/bin/env python
from vptr.utils.general_utils import AttrDict
import numpy as np

from collections import OrderedDict
import json
import datetime
import os

from vptr.wrappers.prev_action_wrapper import PrevActionStack
from vptr.agents.cql_encodersep_parallel.pixel_cql_learner import PixelCQLLearnerEncoderSepParallel
from vptr.wrappers.rescale_actions_wrapper import RescaleActions
from vptr.wrappers.normalize_actions_wrapper import NormalizeActions
from jaxrl_m.vision import encoders as encoders
# from icvf_video.src import icvf_learner as learner
from examples.configs.dataset_config_real import *
from examples.configs.bridge_all_dataset import dataset_fns, target_dataset_fns
import tensorflow as tf
import pickle

from gym.spaces import Dict, Box

import gym
import numpy as np

from vptr.data import BetterReplayBuffer, BetterReplayBufferParallel 
from vptr.data.dataset import MixingReplayBuffer, MixingReplayBufferParallel
from vptr.data.replay_buffer import ReplayBuffer

from vptr.utils.wandb_logger import WandBLogger, create_exp_name
from vptr.wrappers import FrameStack

from examples.train_utils import offline_training_loop, trajwise_alternating_training_loop, load_buffer, insert_data_real, embed_dicts


from vptr.wrappers.reaching_reward_wrapper import ReachingReward
from vptr.utils.general_utils import add_batch_dim
from vptr.data.utils import get_task_id_mapping, load_task_id_mapping
import tensorflow as tf

import jax
import flax

from functools import partial

TARGET_POINT = np.array([0.28425417, 0.04540814, 0.07545623])  # mean
# TARGET_POINT = np.array([0.23, 0., 0.1])

@jax.jit
def embed_images(encoder, obs_dict):
    if encoder.batch_stats is not None:
        embeds = encoder.apply_fn({'params': encoder.params, 'batch_stats': encoder.batch_stats},
                                obs_dict, training=False)
    else:
        embeds = encoder.apply_fn({'params': encoder.params}, obs_dict, training=False)
    return embeds

def batched_apply(fn, batch_size):
    """Turns a function that applies to a fixed batch size into one that applies to a variable batch size.

    Currently assumes that the first axis is the batch axis.
    """

    def pad_to_batch_size(arr):
        return np.pad(arr, ((0, batch_size - len(arr)), *[(0, 0)] * (arr.ndim - 1)))

    def get_batch_size(tree):
        return next(iter(jax.tree_util.tree_leaves(tree))).shape[0]

    def wrapped_fn(*args, **kwargs):
        input_batch_size = get_batch_size((args, kwargs))
        # print(input_batch_size, batch_size)
        outputs = []
        for i in range(0, input_batch_size, batch_size):
            step_batch_size = min(batch_size, input_batch_size - i)
            step_args, step_kwargs = jax.tree_map(
                lambda arr: pad_to_batch_size(arr[i : i + batch_size]), (args, kwargs)
            )
            step_output = fn(*step_args, **step_kwargs)
            outputs.append(
                jax.tree_map(
                    lambda arr: arr[:step_batch_size],
                    step_output,
                )
            )
        return jax.tree_map(lambda *args: np.concatenate(args, axis=0), *outputs)

    return wrapped_fn

# embed_images = batched_apply(embed_images, 8)

# def convert_obs(trajs, task_id_mapping):
#     all_pixels = []
#     all_tids = []
#     for traj in trajs:
#         pixels = np.array([obs['images0'] for obs in traj['observations']])[..., None]
#         # todo: make more general
#         task_ids = task_id_mapping[traj['task_description']]
#         oh_tids = np.zeros((len(pixels), len(task_id_mapping)))
#         oh_tids[range(len(pixels)), task_ids] = 1
#         all_pixels.append(pixels)
#         all_tids.append(oh_tids)
#     while len(all_pixels) % 8 != 0:
#         all_pixels.append(pixels)
#         all_tids.append(oh_tids)
#     return dict(
#         pixels=np.stack(all_pixels),
#         task_id=np.stack(all_tids),
#     )

def convert_obs(traj, task_id_mapping):
    pixels = np.array([obs['images0'] for obs in traj['observations']])[..., None]
    task_ids = task_id_mapping[traj['task_description']]
    oh_tids = np.zeros((len(pixels), len(task_id_mapping)))
    oh_tids[range(len(pixels)), task_ids] = 1
    return dict(
        pixels=pixels,
        task_id=oh_tids,
    )

# def preprocess_mean_embeddings(encoder, all_trajs, task_id_mapping):
#     num_devices = len(jax.devices())
#     all_last_frames = np.array([traj['observations'][-1]['images0'] for sub_trajs in all_trajs for traj in sub_trajs])
#     all_task_ids = [task_id_mapping[traj['task_description']] for sub_trajs in all_trajs for traj in sub_trajs]
#     task_unique, task_counts = np.unique(all_task_ids, return_counts=True)
#     count_dict = dict(zip(task_unique, task_counts))
#     reward_preprocess_dict = dict()
#     batch_size = 32
#     print(len(all_last_frames))
#     for i in range(0, len(all_last_frames) + batch_size, batch_size):
#         frames = all_last_frames[i:i + batch_size]
#         frames = np.pad(frames, ((0, batch_size - len(frames)),) + ((0, 0),) * (len(frames.shape) - 1))
#         batch_task_ids = all_task_ids[i:i + batch_size]
#         orig_len = len(batch_task_ids)
#         batch_task_ids = np.pad(batch_task_ids, ((0, batch_size - len(batch_task_ids)), (0, 0)))
#         # todo: handle language - can't really take mean of certain task id with language labels though
#         oh_task_ids=np.zeros((batch_size, len(task_id_mapping)))
#         oh_task_ids[range(batch_size), batch_task_ids] = 1
#         obs_dict = dict(
#             pixels=frames.reshape(num_devices, batch_size // num_devices, *frames.shape[1:], 1),
#             task_id=oh_task_ids.reshape(num_devices, batch_size // num_devices, -1),
#         )
#         print(obs_dict['pixels'].shape)
#         embeds = embed_images(encoder, obs_dict)
#         embeds = jax.tree_util.tree_map(lambda x: x.reshape(batch_size, *x.shape[2:]), embeds)
#         for j, tid in enumerate(batch_task_ids[:orig_len]):
#             if tid not in reward_preprocess_dict:
#                 reward_preprocess_dict[tid] = np.zeros(embeds['pixels'][0].shape)
#             reward_preprocess_dict[tid] = reward_preprocess_dict[tid] + embeds['pixels'][j] / count_dict[tid]
#     return flax.core.FrozenDict(reward_preprocess_dict)

def preprocess_mean_embeddings(encoder, all_trajs, task_id_mapping):
    all_last_frames = np.array([traj['observations'][-1]['images0'] for sub_trajs in all_trajs for traj in sub_trajs])
    all_task_ids = [task_id_mapping[traj['task_description']] for sub_trajs in all_trajs for traj in sub_trajs]
    task_unique, task_counts = np.unique(all_task_ids, return_counts=True)
    count_dict = dict(zip(task_unique, task_counts))
    reward_preprocess_dict = dict()
    for frame, tid in zip(all_last_frames, all_task_ids):
        oh_task_id=np.zeros((len(task_id_mapping)))
        oh_task_id[tid] = 1
        obs_dict = dict(
            pixels=frame[..., None],
            task_id=oh_task_id,
        )
        embeds = embed_images(encoder, obs_dict)
        if tid not in reward_preprocess_dict:
            reward_preprocess_dict[tid] = np.zeros(embeds['pixels'].shape)
        reward_preprocess_dict[tid] = reward_preprocess_dict[tid] + embeds['pixels'] / count_dict[tid]
    return reward_preprocess_dict

def embedding_goal_distance_reward(encoder, traj, task_id_mapping, preprocess_dict=None, normalize=False):
    embed_2 = None
    obs_dict = convert_obs(traj, task_id_mapping)
    if preprocess_dict is not None:
        embed_2 = preprocess_dict.get(np.argmax(obs_dict['task_id'][-1]))
    embed_1 = embed_images(encoder, obs_dict)['pixels']
    if embed_2 is None:
        embed_2 = embed_images(encoder, jax.tree_util.tree_map(lambda x: x[-1, None], obs_dict))['pixels']
    
    dists = np.linalg.norm(embed_1 - embed_2, axis=-1)
    if normalize:
        print("normalizing by %f" % dists[0])
        dists = dists / dists[0]
    return np.concatenate((-np.diff(dists, axis=0), [0]))

# def embedding_goal_distance_reward(encoder, trajs, task_id_mapping, preprocess_dict=None):
#     embed_2 = None
#     obs_dict = convert_obs(trajs, task_id_mapping)
#     orig_len = len(trajs)
#     if preprocess_dict is not None:
#         embed_2 = np.stack([preprocess_dict.get(np.argmax(obs_dict['task_id'][i, -1])) for i in range(orig_len)])
#     embed_1 = embed_images(encoder, obs_dict)[:orig_len]
#     if embed_2 is None:
#         embed_2 = embed_images(encoder, jax.tree_util.tree_map(lambda x: x[:, -1, None], obs_dict))[:orig_len]
    
#     dists = np.linalg.norm(embed_1 - embed_2[None], axis=-1)
#     return np.pad(-np.diff(dists), (0, 0), (0, 1))

def icvf_value_reward(icvf_agent, traj):
    obs = traj['observations']
    obs = obs['pixels'].squeeze(-1)
    assert obs.shape[1:] == (224, 224, 3)
    goals = obs[-1][None]
    assert goals.shape == obs.shape
    # breakpoint()
    vals = np.min(icvf_agent.value(obs, goals, goals), axis=1)
    return np.concatenate((np.diff(vals, axis=0), [0]), axis=0)

def main(variant):
    # Different per process:
    try:
        device_num = int(os.environ["TPU_VISIBLE_DEVICES"])
        # os.environ["TPU_VISIBLE_DEVICES"] = "0" # "1", "2", "3"
        # Pick a unique port per process
        os.environ["TPU_MESH_CONTROLLER_ADDRESS"] = f"localhost:{variant.tpu_port + device_num}"
        os.environ["TPU_MESH_CONTROLLER_PORT"] = f"{variant.tpu_port+device_num}"
    except:
        # print("Using TPU", os.environ["TPU_VISIBLE_DEVICES"])
        pass
    
    if variant.dataset in dataset_fns:
        train_tasks, eval_tasks = dataset_fns[variant.dataset]()
    else:
        raise ValueError('dataset not found! ' + variant.dataset)

    if variant.target_dataset != '':
        if variant.target_dataset in target_dataset_fns:
            target_train_tasks, target_eval_tasks = target_dataset_fns[variant.target_dataset]()
        else:
            raise ValueError('target dataset not found! ' + variant.target_dataset)
    else:
        target_train_tasks = []
        target_eval_tasks = []
    
    print('TRAIN TASKS', train_tasks)
    print('TARGET TRAIN TASKS', target_train_tasks)

    kwargs = variant['train_kwargs']
    if kwargs.pop('cosine_decay', False):
        kwargs['decay_steps'] = variant.max_steps

    if 'hidden_dims' in variant['train_kwargs']:
        variant['train_kwargs']['hidden_dims'] = tuple(variant['train_kwargs']['hidden_dims'])
    if 'variant_reward' in variant:
        if 'hidden_dims' in variant.variant_reward['train_kwargs']:
            variant.variant_reward['train_kwargs']['hidden_dims'] = tuple(
                variant.variant_reward['train_kwargs']['hidden_dims'])

    if variant.lang_embedding is not None:
        task_id_mapping = None
    else:
        if variant.restore_path:
            task_id_mapping = load_task_id_mapping(
                variant.restore_path,
                target_train_tasks,
                ALIASING_DICT,
                placeholder_task = variant.placeholder_task,
            )
            print('task_id_mapping:', task_id_mapping)
            num_tasks = len(task_id_mapping.keys())
            variant.num_tasks = num_tasks
            variant.task_id_mapping = task_id_mapping
            print('using {} tasks'.format(num_tasks))
        elif variant.num_eval_tasks == -1:
            task_id_mapping = get_task_id_mapping(train_tasks + target_train_tasks, ALIASING_DICT)
            print('task_id_mapping:', task_id_mapping)
            num_tasks = len(task_id_mapping.keys())
            variant.num_tasks = num_tasks
            variant.task_id_mapping = task_id_mapping
            print('using {} tasks'.format(num_tasks))
        else:
            num_tasks = variant.num_eval_tasks
            task_id_mapping = None
        
        # TODO: fix hardconding for relabellin taskid for faucet
        if variant.target_dataset in ['tk6_faucet_left', 'tk6_faucet_right']: 
            v_left = task_id_mapping['turn_faucet_left']
            v_right = task_id_mapping['turn_faucet_right']
            k_0 = list(task_id_mapping.keys())[0]
            k_1 =  list(task_id_mapping.keys())[1]
            task_id_mapping[k_0] = v_left
            task_id_mapping[k_1] = v_right
            task_id_mapping['turn_faucet_left'] = 0
            task_id_mapping['turn_faucet_right'] = 1
            # task_id_mapping = sorted(task_id_mapping.items(), key=lambda x: x[1])
            print('task_id_mapping:', task_id_mapping)

        variant.taskid2string = {v:k for (k, v) in task_id_mapping.items()}

    class DummyEnv():
        def __init__(self):
            super().__init__()
            obs_dict = dict()
            if not variant.from_states:
                obs_dict['pixels'] = Box(low=0, high=255, shape=(224, 224, 3), dtype=np.uint8)
            if variant.add_states:
                obs_dict['state'] = Box(low=-100000, high=100000, shape=(7,), dtype=np.float32)
            if variant.lang_embedding is not None:
                embed_dict = embed_dicts[variant.lang_embedding]
                embedding_dim = next(iter(embed_dict.values())).size
                obs_dict['language'] = Box(low=-0.5, high=0.5, shape=(embedding_dim,))
            else:
                obs_dict['task_id'] = Box(low=0, high=1, shape=(num_tasks,), dtype=np.float32)
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
        # if variant.add_states:
        #     env = StateStack(env, variant.frame_stack)
        if not variant.from_states:
            env = FrameStack(env, variant.frame_stack)
        env = gym.wrappers.TimeLimit(env, variant.episode_timelimit)
        return env

    env = DummyEnv()

    env = wrap(env)
    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=1)
    env.seed(variant.seed)
    if variant.from_states:
        env.disable_render()

    if variant.reward_type == 'dense':
        assert not variant.add_states

    eval_env = env

    sample_obs = add_batch_dim(env.observation_space.sample())
    sample_action = add_batch_dim(env.action_space.sample())
    print('sample obs shapes', [(k, v.shape) for k, v in sample_obs.items()])

    reward_function = None

    kwargs = variant['train_kwargs']
    if kwargs.pop('cosine_decay', False):
        kwargs['decay_steps'] = variant.max_steps

    expname = create_exp_name(variant.prefix, seed=variant.seed)
    if "gs://" in os.environ['EXP']:
        outputdir = tf.io.gfile.join(os.environ['EXP'], expname)
    else:
        outputdir = os.path.join( os.environ['EXP'], expname)
    variant.outputdir = outputdir
    print('writing to output dir ', outputdir)

    group_name = variant.prefix + '_' + variant.launch_group_id
    wandb_logger = WandBLogger(
        variant.prefix != '',
        variant,
        variant.wandb_project,
        experiment_id=expname,
        output_dir=os.path.join(os.environ['WANDB_EXP'], expname),
        group_name=group_name,
        team=variant.team,
    )

    algorithm_fns = {
        'cql_encodersep_parallel': PixelCQLLearnerEncoderSepParallel,
    }
    agent = algorithm_fns[variant.algorithm](variant.seed, sample_obs, sample_action, **kwargs)

    # reward_preprocess_function = None
    # if variant.get('vip_reward_encoder'):
    #     print("Loading VIP reward encoder")
    #     # breakpoint()
    #     vip_kwargs = dict(kwargs)
    #     vip_kwargs['pretrained_encoder'] = variant.vip_reward_encoder
    #     vip_kwargs['encoder_type'] = 'pretrained_resnet'
    #     vip_kwargs['freeze_encoders'] = True
    #     vip_encoder = algorithm_fns['cql_encodersep_parallel'](variant.seed, sample_obs, sample_action, **vip_kwargs)._critic_encoder
    #     vip_encoder = flax.jax_utils.unreplicate(vip_encoder)
    #     reward_function = partial(embedding_goal_distance_reward, encoder=vip_encoder, task_id_mapping=task_id_mapping)
    #     reward_preprocess_function = partial(preprocess_mean_embeddings, encoder=vip_encoder, task_id_mapping=task_id_mapping)
    # elif variant.get('icvf_reward_encoder'):
    #     print("Loading ICVF reward encoder")
    #     # breakpoint()
    #     icvf_kwargs = dict(kwargs)
    #     icvf_kwargs['pretrained_encoder'] = variant.icvf_reward_encoder
    #     icvf_kwargs['encoder_type'] = 'resnetv2-50-1'
    #     icvf_kwargs['freeze_encoders'] = True
    #     icvf_encoder = algorithm_fns['cql_encodersep_parallel'](variant.seed, sample_obs, sample_action, **icvf_kwargs)._critic_encoder
    #     reward_function = partial(embedding_goal_distance_reward, encoder=icvf_encoder, task_id_mapping=task_id_mapping)
    #     reward_preprocess_function = partial(preprocess_mean_embeddings, encoder=icvf_encoder, task_id_mapping=task_id_mapping)
    # elif variant.get('icvf_reward_agent'):
    #     print("Loading ICVF reward agent")
    #     # breakpoint()
    #     with open('/nfs/nfs1/users/derekguo/code/icvf/icvf_params.pkl', 'rb') as f:
    #         save_dict = pickle.load(f)
    #     icvf_config = save_dict['config']
    #     state_dict = save_dict['agent']
    #     state_dict['value']['extra_variables'] = dict()
    #     state_dict['target_value']['extra_variables'] = dict()

    #     icvf_env = DummyEnv(variant)
    #     encoder_def = encoders['resnetv2-50-1']()
    #     icvf_sample_obs = add_batch_dim(icvf_env.observation_space.sample())
    #     icvf_sample_obs['image'] = icvf_sample_obs.pop('pixels')
    #     icvf_agent = learner.create_learner(
    #         seed=77,
    #         observations=icvf_sample_obs,
    #         encoder_def=encoder_def,
    #         icvf_type='symmetric_norm',
    #         **icvf_config,
    #     )

    #     full_icvf = flax.serialization.from_state_dict(icvf_agent, state_dict)
    #     reward_function = partial(icvf_value_reward, icvf_agent=full_icvf)
    # if variant.get('no_reward_preprocess'):
    #     print("Skipping reward preprocessing")
    #     reward_preprocess_function = None
    if variant.get('reset_agent_restore_path', False):
        assert variant.algorithm == 'cql_encodersep_parallel'
        reset_agent = PixelCQLLearnerEncoderSepParallel(variant.seed, sample_obs,
                                            sample_action, **kwargs)
    else:
        reset_agent = None

    if hasattr(variant, "encoder_path") and variant.encoder_path is not None:
        assert variant.restore_path == '', "Can't load both checkpoint and pretrained encoder"

    if variant.restore_path != '':
        print('loading checkpoint...')
        agent.restore_checkpoint(variant.restore_path)
        if 'parallel' in variant.algorithm:
            # agent.replicate()
            pass
        if variant.normalize_actions:
            action_stats = load_action_stats('/'.join(variant.restore_path.split('/')[:-1]))[0]
            print('restored action stats.')

        if variant.get('reset_agent_restore_path', False):
            reset_agent.restore_checkpoint(variant.reset_agent_restore_path)
    else:
        action_stats = {'mean': np.zeros_like(env.action_space.low), 'std': np.ones_like(env.action_space.low)}

    if variant.from_states:
        replay_buffer_class = ReplayBuffer
    elif variant.algorithm in ['cql_parallel', 'cql_encodersep_parallel']:
        if variant.target_dataset != '':
            # replay_buffer_class = MemoryEfficientReplayBuffer
            replay_buffer_class = BetterReplayBuffer
        else:
            # replay_buffer_class = MemoryEfficientReplayBufferParallel
            replay_buffer_class = BetterReplayBufferParallel
    else:
        # replay_buffer_class = MemoryEfficientReplayBuffer
        replay_buffer_class = BetterReplayBuffer
    if not variant.debug and variant.algorithm in ['cql_encodersep_parallel', 'cql_parallel'] and variant.get('offline_only_restore_onlinedata', '') == '':
        mixing_buff_class = MixingReplayBufferParallel
    else:
        mixing_buff_class = MixingReplayBuffer

    if not variant.online_from_scratch:
        print('making train buffer:')
        split_pos_neg = variant.get('split_pos_neg', False)
        train_replay_buffer, data_count_dict, misc_dict = make_buffer_and_insert(
            env,
            replay_buffer_class,
            task_id_mapping,
            train_tasks,
            variant,
            split_pos_neg=split_pos_neg,
            reward_function=reward_function,
            reward_preprocess_function=None, #reward_preprocess_function,
        )
        print('making val buffer:')
        eval_replay_buffer, data_count_dict_val, _ = make_buffer_and_insert(
            env,
            replay_buffer_class,
            task_id_mapping,
            eval_tasks,
            variant,
            split_pos_neg=False,
            reward_function=reward_function,
            reward_preprocess_dict=misc_dict['reward_preprocess_dict'],
        )
        if 'target_dataset' in variant and variant.target_dataset != '':
            if variant.get('split_by_traj_target', False):
                split_pos_neg = True
                split_by_traj=True
                pos_neg_ratio = variant.split_by_traj_target_ratio
            else:
                split_by_traj=False
                pos_neg_ratio = 0.5
            target_replay_buffer_train, data_count_dict_target, target_misc_dict = make_buffer_and_insert(
                env,
                replay_buffer_class,
                task_id_mapping,
                target_train_tasks,
                variant,
                split_pos_neg=split_pos_neg,
                split_by_traj=split_by_traj,
                reward_function=reward_function,
                reward_preprocess_function=None, #reward_preprocess_function,
                num_traj_cutoff=variant.num_target_traj,
                pos_neg_ratio=pos_neg_ratio,
            )
            train_replay_buffer = mixing_buff_class([train_replay_buffer, target_replay_buffer_train], variant.target_mixing_ratio)

            target_replay_buffer_eval, _, _ = make_buffer_and_insert(
                env,
                replay_buffer_class,
                task_id_mapping,
                target_eval_tasks,
                variant,
                split_pos_neg=split_pos_neg,
                split_by_traj=split_by_traj,
                reward_function=reward_function,
                reward_preprocess_dict=None, #target_misc_dict['reward_preprocess_dict'],
                num_traj_cutoff=variant.num_target_traj,
                pos_neg_ratio=pos_neg_ratio,
            )
            eval_replay_buffer = mixing_buff_class([eval_replay_buffer, target_replay_buffer_eval], variant.target_mixing_ratio)
        else:
            data_count_dict_target = None

        action_stats = train_replay_buffer.compute_action_stats()
        print('dataset action stats ', action_stats)
        if variant.normalize_actions:
            train_replay_buffer.normalize_actions(action_stats)
            eval_replay_buffer.normalize_actions(action_stats)
            print('dataset action stats after norm', train_replay_buffer.compute_action_stats())
            env.set_action_stats(action_stats)
            eval_env.set_action_stats(action_stats)
            save_action_stats(action_stats, variant.outputdir)

        save_jsons(outputdir, variant, task_id_mapping, data_count_dict, data_count_dict_val, data_count_dict_target)

        train_replay_buffer.seed(variant.seed)
        eval_replay_buffer.seed(variant.seed)

        if variant.get('offline_only_restore_onlinebuffer', '') == '' and variant.get('offline_only_restore_onlinedata', '') == '':
            offline_training_loop(variant, agent, eval_env, train_replay_buffer, eval_replay_buffer, wandb_logger, perform_control_evals=False, task_id_mapping=task_id_mapping)


    replay_buffer = online_replay_buffer = replay_buffer_class(env.observation_space, env.action_space, int(5e4))
    save_jsons(outputdir, variant, task_id_mapping)

    if variant.restore_path != '':
        restore_folder = '/'.join(str.split(variant.restore_path, '/')[:-1])
        if 'replaybuffer.npy' in os.listdir(restore_folder):
            online_replay_buffer.restore(restore_folder + '/replaybuffer.npy')
            print("restored replay buffer! Confirm with c to proceed!")
            import pdb; pdb.set_trace()

    replay_buffer.seed(variant.seed)

    # disable mc constraint during online phase
    if agent.bound_q_with_mc:
        print("changing bound_q_with_mc to false during online finetuning")
        agent.bound_q_with_mc = False

    now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    from vptr.data.raw_saver import RawSaverJaxRL
    saver = RawSaverJaxRL(os.environ['DATA'] + '/online_datacollection/{}/{}'.format(variant.prefix, now),
                          env.unnormalize_actions)

    trajwise_alternating_training_loop(variant, agent, reset_agent, env, eval_env, online_replay_buffer, replay_buffer,
                                           wandb_logger, saver=saver, perform_control_evals=variant.perform_control_evals if 'perform_control_evals' in variant else True)


def make_buffer_and_insert(env, replay_buffer_class, task_id_mapping, tasks, variant, split_pos_neg=False, split_by_traj=False,
                           reward_function=None, reward_preprocess_function=None, reward_preprocess_dict=None,
                           num_traj_cutoff=None, pos_neg_ratio=0.5, **kwargs):
    pos_buffer_size = neg_buffer_size = buffer_size = 0
    data_count_dict = {}
    all_trajs = []
    for dataset_file in tasks:
        task_size, trajs = load_buffer(
            dataset_file,
            variant,
            ALIASING_DICT,
            multi_viewpoint=variant.multi_viewpoint,
            data_count_dict=data_count_dict,
            split_pos_neg=split_pos_neg,
            split_by_traj=split_by_traj,
            num_traj_cutoff=num_traj_cutoff
        )
        all_trajs.append(trajs)
        if split_pos_neg:
            pos_size, neg_size = task_size
            pos_buffer_size += pos_size
            neg_buffer_size += neg_size
        else:
            buffer_size += task_size
    misc_dict = dict(
        reward_preprocess_dict=dict()
    )
    partial_reward_function = None
    if reward_function is not None:
        if reward_preprocess_dict is None and reward_preprocess_function is not None:
            reward_preprocess_dict = reward_preprocess_function(all_trajs=all_trajs)
        misc_dict['reward_preprocess_dict'] = reward_preprocess_dict
        partial_reward_function = partial(reward_function, preprocess_dict=reward_preprocess_dict)
    if split_pos_neg:
        print('pos size ', pos_buffer_size)
        print('neg size ', neg_buffer_size)
        pos_buffer = replay_buffer_class(env.observation_space, env.action_space, pos_buffer_size, **kwargs)
        neg_buffer = replay_buffer_class(env.observation_space, env.action_space, neg_buffer_size, **kwargs)
        buffer = MixingReplayBuffer([pos_buffer, neg_buffer], pos_neg_ratio)
    else:
        print('size ', buffer_size)
        buffer = replay_buffer_class(env.observation_space, env.action_space, buffer_size, **kwargs)
    print('inserting data...')
    while all_trajs:
        trajs = all_trajs.pop(0)
        num_views = 3 if variant.multi_viewpoint else 1
        for i in range(num_views):
            insert_data_real(
                variant,
                buffer,
                trajs,
                variant.reward_type,
                task_id_mapping=task_id_mapping,
                env=env, image_key=f'images{i}',
                target_point=TARGET_POINT,
                split_pos_neg=split_pos_neg,
                split_by_traj=split_by_traj,
                reward_function=partial_reward_function,
            )
        del trajs

    return buffer, data_count_dict, misc_dict

def save_jsons(outputdir, variant, task_id_mapping=None, data_count_dict=None, data_count_dict_val=None, data_count_dict_target=None):
    if not os.path.exists(outputdir) and "gs://" not in outputdir:
        os.makedirs(outputdir)
    print('saving config to ', outputdir)

    
    def save_ordered_dict(output_dir, name, dict):
        if "gs://" in output_dir:
            with tf.io.gfile.GFile(tf.io.gfile.join(output_dir, "{}.json".format(name)), "w") as f:
                variant = OrderedDict(sorted(OrderedDict(dict).items(), key=lambda x: x))
                json.dump(variant, f, indent=4)
        else:
            with open(os.path.join(output_dir, "{}.json".format(name)), 'w') as f:
                variant = OrderedDict(sorted(OrderedDict(dict).items(), key=lambda x: x))
                json.dump(variant, f, indent=4)

    optional_jsons = {
        'task_index': task_id_mapping,
        'data_count_dict': data_count_dict,
        'data_count_dict_val': data_count_dict_val,
        'data_count_dict_target': data_count_dict_target,
    }
    for filename, optional_json in optional_jsons.items():
        if optional_json is not None:
            save_ordered_dict(outputdir, filename, optional_json)
    print('data count dict train', data_count_dict)
    print('data count dict val', data_count_dict_val)
    save_ordered_dict(outputdir, 'config', variant)

def save_action_stats(action_stats, path):
    if "gs://" in path:
        with tf.io.gfile.GFile(tf.io.gfile.join(path, "action_stats.npy"), "w") as f:
            np.save(f, [action_stats])
    else:
        np.save(os.path.join(path, '/action_stats.npy'), [action_stats])

def load_action_stats(path):
    if "gs://" in path:
        with tf.io.gfile.GFile(tf.io.gfile.join(path, "action_stats.npy"), "rb") as f:
            return np.load(f, allow_pickle=True)
    else:
        return np.load(path + '/action_stats.npy', allow_pickle=True)
