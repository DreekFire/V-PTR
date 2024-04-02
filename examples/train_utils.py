from pathlib import Path
repo_dir = Path(__file__).parent.parent

import copy
import pickle

import time
from tqdm import tqdm
import os
import numpy as np
import jax.numpy as jnp
import wandb
import collections
from vptr.utils.visualization_utils import visualize_image_actions
from vptr.wrappers.reaching_reward_wrapper import compute_distance_reward
from vptr.utils.visualization_utils import visualize_states_rewards
from vptr.data.dataset import MixingReplayBuffer, MixingReplayBufferParallel
from vptr.data.dataset import PropertyReplayBuffer
import tensorflow as tf
import re

embed_dicts = {} # np.load(repo_dir/'notebooks/final_embeddings/normalized_embeddings_dicts.npy', allow_pickle=True).item()
embed_dicts['paraphrase-minilm'] = np.load(repo_dir/'notebooks/final_embeddings/paraphrase_embeddings.npy', allow_pickle=True).item()['paraphrase-MiniLM-L3']
with (repo_dir/'notebooks/final_embeddings/embed_dict.pkl').open('rb') as pkl:
    embed_dicts['rule-based'] = pickle.load(pkl)
with (repo_dir/'notebooks/final_embeddings/grif_embeddings/text_embed_dict.pkl').open('rb') as pkl:
    embed_dicts['grif-text'] = pickle.load(pkl)
with (repo_dir/'notebooks/final_embeddings/grif_embeddings/grif_v1_remap_embed_dict.pkl').open('rb') as pkl:
    embed_dicts['grif-text-v1-remap'] = pickle.load(pkl)

def offline_training_loop(variant, agent, eval_env, replay_buffer, eval_replay_buffer=None, wandb_logger=None, perform_control_evals=True, task_id_mapping=None):
    if eval_replay_buffer is None:
        eval_replay_buffer = replay_buffer
    
    if variant.offline_finetuning_start != -1:
        changed_buffer_to_finetuning = False
        if isinstance(replay_buffer, MixingReplayBuffer) or isinstance(replay_buffer, MixingReplayBufferParallel):
            replay_buffer.set_mixing_ratio(1.0) 
    replay_buffer_iterator = replay_buffer.get_iterator(variant.batch_size)
    if eval_replay_buffer is not None:
        eval_replay_buffer_iterator = eval_replay_buffer.get_iterator(variant.batch_size)

    bc_end = variant.get('bc_end', 0)
    if hasattr(agent, 'use_bc_actor'):
        agent.use_bc_actor(True)

    for i in tqdm(range(1, min(variant.online_start, variant.max_steps) + 1),smoothing=0.1,):
        if i == bc_end + 1 and hasattr(agent, 'use_bc_actor'):
            print('ending BC training')
            agent.use_bc_actor(False)
        if variant.get('streaming_interval', False):
            if i % variant.streaming_interval == 0 or i == 1:
                # replay_buffer.replay_buffers[1].add_streaming_data(variant.max_streaming_iter)
                replay_buffer.replay_buffers[1].add_streaming_data_1traj()
        
        t0 = time.time()
        batch = next(replay_buffer_iterator)
        tget_data = time.time() - t0
        t1 = time.time()

        update_info = agent.update(batch)
        tupdate = time.time() - t1

        if variant.offline_finetuning_start != -1:
            # Update the buffer when finetuning offline
            if not changed_buffer_to_finetuning and i >= variant.offline_finetuning_start and (
                    isinstance(replay_buffer, MixingReplayBuffer) or isinstance(replay_buffer, MixingReplayBufferParallel)):
                replay_buffer.set_mixing_ratio(variant.target_mixing_ratio)
                del replay_buffer_iterator
                replay_buffer_iterator = replay_buffer.get_iterator(variant.batch_size)
                changed_buffer_to_finetuning = True
        if i % variant.eval_interval == 0:
            if hasattr(agent, 'unreplicate'):
                agent.unreplicate()
            wandb_logger.log({'t_get_data': tget_data}, step=i)
            wandb_logger.log({'t_update': tupdate}, step=i)
            # if 'pixels' in update_info and i % (variant.eval_interval*10) == 0:
            if 'pixels' in update_info and i % variant.eval_interval == 0:
                image = visualize_image_actions(update_info.pop('pixels'), batch['actions'], update_info.pop('pred_actions_mean'))
                wandb_logger.log({'training/image_actions': wandb.Image(image)}, step=i)
            if perform_control_evals:
                perform_control_eval(agent, eval_env, i, variant, wandb_logger)
            agent.perform_eval(variant, i, wandb_logger, replay_buffer, prefix='training_media')
            agent.perform_eval(variant, i, wandb_logger, eval_replay_buffer, prefix='validation_media')
            if hasattr(agent, 'replicate'):
                agent.replicate()

        if i % variant.log_interval == 0:
            for k, v in update_info.items():
                if v.ndim == 0:
                    wandb_logger.log({f'training/{k}': v}, step=i)
                elif v.ndim <= 2:
                    wandb_logger.log_histogram(f'training/{k}', v, i)
            val_batch = next(eval_replay_buffer_iterator)
            val_update_info = agent.update(val_batch, no_update=True)
            for k, v in val_update_info.items():
                if v.ndim == 0:
                    wandb_logger.log({f'validation/{k}': v}, step=i)
                elif v.ndim <= 2:
                    wandb_logger.log_histogram(f'validation/{k}', v, i)

        if variant.checkpoint_interval != -1: # and i >= variant.offline_finetuning_start:
            if i % variant.checkpoint_interval == 0:
                if hasattr(agent, 'unreplicate'):
                    agent.unreplicate()
                agent.save_checkpoint(variant.outputdir, i, variant.checkpoint_interval)
                if hasattr(agent, 'replicate'):
                    agent.replicate()


def trajwise_alternating_training_loop(variant, agent, reset_agent, env, eval_env, online_replay_buffer, replay_buffer, wandb_logger,
                                       perform_control_evals=True, saver=None):
    replay_buffer_iterator = replay_buffer.get_iterator(variant.batch_size)

    traj_id = 0

    performed_first_eval = False

    i = variant.online_start + 1
    with tqdm(total=variant.max_steps + 1) as pbar:
        while i < variant.max_steps + 1:
            if variant.alternate_stochastic_data_collect:
                deterministic_rollout = bool(traj_id % 2)
            else:
                deterministic_rollout = not variant.stochastic_data_collect

            if hasattr(agent, 'unreplicate'):
                agent.unreplicate()
                reset_agent.unreplicate()

            run_reset(variant, env, reset_agent)
            # assume task_id 0 is closing, 1 is opening
            traj = collect_traj_timed(variant, agent, env, task_id=1, deterministic=deterministic_rollout)
            traj_id += 1
            if hasattr(agent, 'replicate'):
                agent.replicate()
                reset_agent.replicate()

            add_online_data_to_buffer(variant, traj, online_replay_buffer)
            if saver is not None:
                saver.save(traj)
            print('collecting traj len online buffer', len(online_replay_buffer))

            if len(online_replay_buffer) > variant.start_online_updates:
                for _ in range(len(traj)*variant.multi_grad_step):
                    if i % variant.eval_interval == 0 or not performed_first_eval:
                        if hasattr(agent, 'unreplicate'):
                            agent.unreplicate()
                            reset_agent.unreplicate()
                        if perform_control_evals:
                            perform_control_eval(agent, eval_env, i, variant, wandb_logger, reset_agent=reset_agent)
                        agent.perform_eval(variant, i, wandb_logger, replay_buffer)
                        performed_first_eval = True
                        if hasattr(agent, 'replicate'):
                            agent.replicate()
                            reset_agent.replicate()

                    # online perform update once we have some amount of online trajs
                    batch = next(replay_buffer_iterator)
                    update_info = agent.update(batch)
                    pbar.update()
                    i += 1

                    if i % variant.log_interval == 0:
                        for k, v in update_info.items():
                            if v.ndim == 0:
                                wandb_logger.log({f'training/{k}': v}, step=i)
                            elif v.ndim <= 2:
                                wandb_logger.log_histogram(f'training/{k}', v, i)
                        wandb_logger.log({'replay_buffer_size': len(online_replay_buffer)}, i)

                    if variant.checkpoint_interval != -1:
                        if i % variant.checkpoint_interval == 0:
                            print('saving checkpoint to {}'.format(variant.outputdir))
                            agent.save_checkpoint(variant.outputdir, i, variant.checkpoint_interval)
                            if hasattr(variant, 'save_replay_buffer') and variant.save_replay_buffer:
                                print('saving replay buffer to ', variant.outputdir + '/replaybuffer.npy')
                                online_replay_buffer.save(variant.outputdir + '/replaybuffer.npy')

def add_online_data_to_buffer(variant, traj, online_replay_buffer):
    if variant.only_add_success:
        if traj[-1]['reward'] < 1e-3:
            print('trajecotry discarded because unsuccessful')
            return

    online_replay_buffer.increment_traj_counter()
    for t, step in enumerate(traj):
        obs = step['observation']
        next_obs = step['next_observation']
        if not variant.add_states and 'state' in obs:
            obs.pop('state')
        if not variant.add_states and 'state' in next_obs:
            next_obs.pop('state')
        online_replay_buffer.insert(
            dict(observations=obs,
                 actions=step['action'],
                 next_actions=traj[t + 1]['action'] if t < len(traj) - 1 else step['action'],
                 rewards=step['reward'],
                 masks=step['mask'],
                 dones=step['done'],
                 next_observations=next_obs,
                 trajectory_id=online_replay_buffer._traj_counter
                 ))

def success_criterion(rewards):
    if np.any(np.array(rewards) > 0):
        return 1
    else:
        return 0

def run_multiple_trajs(variant, agent,  env, num_trajs, deterministic=True, saver=None, reset_agent=None):
    obs = []

    for i in range(num_trajs):
        print('##############################################')
        print('traj', i)
        if reset_agent is not None:
            run_reset(variant, env, reset_agent)
        # assume task_id 0 is closing, 1 is opening
        if variant.lang_embedding:
            traj = collect_traj_timed(variant, agent, env, embedding=variant.embedding, deterministic=deterministic)
        else:
            traj = collect_traj_timed(variant, agent, env, task_id=variant.eval_task_id, deterministic=deterministic)
        rewards = [step['reward'] for step in traj]

        obs.append([step['observation'] for step in traj])
        if saver is not None:
            # if isinstance(saver, RawSaver):
            #     traj_image = copy.deepcopy(traj)
            #     for i in range(len(traj)):
            #         traj_image[i]['images'] = traj_image[i].pop('pixels')
            # else:
            traj_image = traj
            saver.save(traj_image)

        if variant.get('visualize_run', False):
            agent.show_value_reward_visualization(traj)

    eval_info = {
        'obs': obs,
    }
    return eval_info

def run_reset(variant, env, reset_agent):
    assert variant.get('reset_agent_restore_path', False)
    # assume task_id 0 is closing, 1 is opening
    task_id = env.select_task_from_reward_function()
    while task_id == 0:
        print('resetting with reset agent')
        collect_traj_timed(variant, reset_agent, env, task_id=0, deterministic=True)
        task_id = env.select_task_from_reward_function()

def collect_traj_timed(variant, agent, env, task_id=None, embedding=None, deterministic=True):
    obs, done = env.reset(), False

    full_obs, next_full_obs = None, None
    if 'full_obs' in obs:
        full_obs = obs.pop('full_obs')

    env.start()  # this sets the correct moving time for the robot
    last_tstep = time.time()

    step_duration = 0.2

    print('using task id ', task_id)
    print('deterministic  ', deterministic)
    print('policy std  ', variant.policy_std)

    breakpoint()
    
    traj = []
            
    last_gripper = 1
    while not done:
        if time.time() > last_tstep + step_duration:
            if (time.time() - last_tstep) > step_duration * 1.05:
                print('###########################')
                print('Warning, loop takes too long: {}s!!!'.format(time.time() - last_tstep))
                print('###########################')
            if (time.time() - last_tstep) < step_duration * 0.95:
                print('###########################')
                print('Warning, loop too short: {}s!!!'.format(time.time() - last_tstep))
                print('###########################')
            last_tstep = time.time()
            
            if embedding is not None:
                obs['language'] = embedding
            else:
                obs['task_id'] = np.zeros(variant.num_tasks, np.float32)[None]
                obs['task_id'][:, task_id] = 1.
                if hasattr(env, "set_task_id"):
                    env.set_task_id(task_id)
            if variant.from_states:
                obs_filtered = copy.deepcopy(obs)
                if 'pixels' in obs_filtered:
                    obs_filtered.pop('pixels')
            elif variant.reward_type == 'dense':  # for reaching task we don't want to use the state
                obs_filtered = copy.deepcopy(obs)
                if 'state' in obs_filtered:
                    obs_filtered.pop('state')
            else:
                obs_filtered = obs

            if deterministic:
                # example_obs = next(obs_generator)
                # ex_obs, action = next(replay_generator, (None, None))
                # if action is None:
                #     break
                action = agent.eval_actions(obs_filtered)
                action[..., -1] = action[..., -1] / variant.smooth_gripper
                # action = agent_gd_actions(agent, obs_filtered).reshape(1, 7)
                print('Q live', get_q_value(action, obs_filtered, agent._critic_encoder, agent._critic_decoder)[0, 0], obs['state'].squeeze())
                # print('Q replay', get_q_value(action, ex_obs, agent._critic_encoder, agent._critic_decoder)[0, 0], obs['state'].squeeze())
                print('Action live', action)
                # print('Action replay', agent.eval_actions(ex_obs))
                # print('Action recorded', action.flatten())
                print('Gripper Open', action.flatten()[-1] > 0)
            else:
                if variant.additive_noise != -1:
                    print('adding additive noise, std', variant.additive_noise)
                    action = np.array(agent.eval_actions(obs_filtered))
                    action += np.random.normal(np.zeros(7), np.ones(7)*variant.additive_noise)
                    assert variant.rescale_actions
                    action = np.clip(action, -1, 1)
                else:
                    action = agent.sample_actions(obs_filtered)
                action[..., -1] = action[..., -1] / variant.smooth_gripper
            new_action = action.copy()
            new_action[0, -1] = np.clip(action[0, -1], last_gripper - 0.3, last_gripper + 1)
            last_gripper = new_action[0, -1]
            tstamp_return_obs = last_tstep + step_duration

            next_obs, reward, done, info = env.step({'action':new_action, 'tstamp_return_obs':tstamp_return_obs})

            info['task_id'] = task_id

            if 'full_obs' in next_obs:
                next_full_obs = next_obs.pop('full_obs')
            traj.append({
                'observation': obs,
                'full_observation': full_obs,
                'action': action,
                'reward': reward,
                'next_observation': next_obs,
                'next_full_observation': next_full_obs,
                'done': done,
                'info': info
            })
            obs = next_obs
            full_obs = next_full_obs
            
    return post_process_traj(traj, variant)

def post_process_traj(traj, variant):
    if variant.sparsify_reward:
        from vptr.extra_envs.widowx_real_env import TERMINATION_REW_THRESH, NUM_STEPS_HIGHREWARD
        thresh = TERMINATION_REW_THRESH[variant.taskid2string[traj[0]['info']['task_id']]]
        rewards = np.array([step['reward'] for step in traj])
        new_rewards = np.zeros_like(rewards)
        if np.all(rewards[-NUM_STEPS_HIGHREWARD:] > thresh):
            new_rewards[-NUM_STEPS_HIGHREWARD:] = 1.
        mask = 1. - new_rewards
        for i, step in enumerate(traj):
            step['reward'] = new_rewards[i]
            step['mask'] = mask[i]
    else:
        for step in traj:
            if not step['done'] or 'TimeLimit.truncated' in step['info'] or 'Error.truncated' in step['info']:
                step['mask'] = 1.0
            else:
                step['mask'] = 0.

    for step in traj:
        step['reward'] = step['reward'] * variant.reward_scale + variant.reward_shift
    return traj


def make_multiple_value_reward_visualizations(agent, variant, i, replay_buffer, wandb_logger, prefix=None):
    def log_with_prefix(data, *args, **kwargs):
        if prefix is not None:
            data = {f'{prefix}/{k}': v for k, v in data.items()}
        wandb_logger.log(data, *args, **kwargs)

    trajs = replay_buffer.get_random_trajs(3)
    if not hasattr(variant, 'target_dataset') or variant.target_dataset == '':  # if not using target dataset
        if hasattr(replay_buffer, 'replay_buffers') and  hasattr(replay_buffer.replay_buffers[1], 'replay_buffers'): # if doing offline pretraining with pos_neg saved npy target data
                images = agent.make_value_reward_visualization(variant, trajs[0])
                log_with_prefix({'reward_value_images_offline_bridge': wandb.Image(images)}, step=i)
                images = agent.make_value_reward_visualization(variant, trajs[1][0])
                log_with_prefix({'reward_value_images_offline_target_pos': wandb.Image(images)}, step=i)
                images = agent.make_value_reward_visualization(variant, trajs[1][1])
                log_with_prefix({'reward_value_images_offline_target_neg': wandb.Image(images)}, step=i)
        elif hasattr(replay_buffer, 'replay_buffers') and not hasattr(replay_buffer.replay_buffers[1], 'replay_buffers'): # if doing online learning
                images = agent.make_value_reward_visualization(variant, trajs[0])
                log_with_prefix({'reward_value_images_offline': wandb.Image(images)}, step=i)
                images = agent.make_value_reward_visualization(variant, trajs[1])
                log_with_prefix({'reward_value_images_online': wandb.Image(images)}, step=i)
        else: # if doing offline learning
            images = agent.make_value_reward_visualization(variant, trajs)
            log_with_prefix({'reward_value_images': wandb.Image(images)}, step=i)
    else: # if using target dataset
        if hasattr(replay_buffer.replay_buffers[1], 'replay_buffers') and not hasattr(replay_buffer.replay_buffers[0], 'replay_buffers')  :  # if finetuning on online data which is split between pos and neg
            images = agent.make_value_reward_visualization(variant, trajs[0])
            log_with_prefix({'reward_value_images_offline_bridge': wandb.Image(images)}, step=i)
            images = agent.make_value_reward_visualization(variant, trajs[1][0])
            log_with_prefix({'reward_value_images_online_pos': wandb.Image(images)}, step=i)
            images = agent.make_value_reward_visualization(variant, trajs[1][1])
            log_with_prefix({'reward_value_images_online_neg': wandb.Image(images)}, step=i)
        elif hasattr(replay_buffer.replay_buffers[1], 'replay_buffers') or isinstance(replay_buffer.replay_buffers[0], PropertyReplayBuffer):  # if doing offline or online learning with an online buffer that is split between pos and neg
            try:
                images = agent.make_value_reward_visualization(variant, trajs[0][0])
                log_with_prefix({'reward_value_images_offline_bridge': wandb.Image(images)}, step=i)
                images = agent.make_value_reward_visualization(variant, trajs[0][1])
                log_with_prefix({'reward_value_images_offline_target': wandb.Image(images)}, step=i)
                images = agent.make_value_reward_visualization(variant, trajs[1][0])
                log_with_prefix({'reward_value_images_online_pos': wandb.Image(images)}, step=i)
                images = agent.make_value_reward_visualization(variant, trajs[1][1])
                log_with_prefix({'reward_value_images_online_neg': wandb.Image(images)}, step=i)
            except:
                import pdb; pdb.set_trace()
        elif hasattr(replay_buffer.replay_buffers[0], 'replay_buffers'):  # if doing online learning
            images = agent.make_value_reward_visualization(variant, trajs[0][0])
            log_with_prefix({'reward_value_images_offline_bridge': wandb.Image(images)}, step=i)
            images = agent.make_value_reward_visualization(variant, trajs[0][1])
            log_with_prefix({'reward_value_images_offline_target': wandb.Image(images)}, step=i)
            images = agent.make_value_reward_visualization(variant, trajs[1])
            log_with_prefix({'reward_value_images_online': wandb.Image(images)}, step=i)
        else: # if doing offline learning
            images = agent.make_value_reward_visualization(variant, trajs[0])
            log_with_prefix({'reward_value_images_bridge': wandb.Image(images)}, step=i)
            images = agent.make_value_reward_visualization(variant, trajs[1])
            log_with_prefix({'reward_value_images_target': wandb.Image(images)}, step=i)
            if variant.lang_embedding is not None:
                images = agent.make_language_counterfactual_visualization(trajs[0])
                log_with_prefix({'counterfactual_value_images_bridge': wandb.Image(images)}, step=i)


def perform_control_eval(agent, eval_env, i, variant, wandb_logger, saver=None, reset_agent=None):
    if variant.from_states:
        if hasattr(eval_env, 'enable_render'):
            eval_env.enable_render()
    eval_info = run_multiple_trajs(variant, agent, eval_env,
                                   num_trajs=variant.eval_episodes, deterministic=not variant.stochastic_evals,
                                   saver=saver, reset_agent=reset_agent)
    print('eval runs done.')
    if variant.from_states:
        if hasattr(eval_env, 'enable_render'):
            eval_env.disable_render()
    trajs_obs = eval_info.pop('obs')
    if 'pixels' in trajs_obs[0][0]:
        videos = []
        for obs in trajs_obs:
            images = [ts['pixels'] for ts in obs]
            video = np.concatenate(images, 0)
            if len(video.shape) == 5:
                video = video[..., -1] # visualizing only the last frame of the stack when using framestacking
            videos.append(video.transpose(0, 3, 1, 2))
        videos = concat_videos(videos)
        wandb_logger.log({'eval_video': wandb.Video(videos, fps=8)}, step=i)

    if 'state' in obs[0] and variant.reward_type == 'dense':
        states = np.stack([ts['state'] for ts in obs])
        states_image = visualize_states_rewards(states, eval_info['rewards'], eval_env.target)
        wandb_logger.log({'state_traj_image': wandb.Image(states_image)}, step=i)

    for k, v in eval_info.items():
        if v.ndim == 0:
            wandb_logger.log({f'evaluation/{k}': v}, step=i)

    for k in eval_info.keys():
        if 'return' or 'length' in k:
            print("{} {}".format(k, eval_info[k]))


def concat_videos(videos):
    max_len = max([vid.shape[0] for vid in videos])
    _, C, H, W = videos[0].shape
    new_videos = []
    for vid in videos:
        if vid.shape[0] < max_len:
            vid = np.concatenate([vid, np.zeros([max_len - vid.shape[0], C, H, W])])
        new_videos.append(vid)
    return np.concatenate(new_videos, 3)


def run_evals_only(variant, agent, reset_agent, eval_env, wandb_logger):
    if variant.save_evals_prefix != '':
        from vptr.data.raw_saver import RawSaverJaxRL
        from vptr.utils.wandb_logger import create_exp_name
        expname = create_exp_name(variant.prefix, seed=variant.seed)
        outputdir = os.environ['EXP'] + variant.save_evals_prefix + expname
        print('saving eval rollouts to ', outputdir)
        saver = RawSaverJaxRL(outputdir, eval_env.unnormalize_actions)
    else:
        saver = None
    perform_control_eval(agent, eval_env, 0, variant, wandb_logger, saver=saver, reset_agent=reset_agent)

def is_positive_sample(traj_end, i, variant, task_name):
    return i >= traj_end - variant.num_final_reward_steps

def is_positive_traj(traj):
    return traj['rewards'][-1, 0] == 10

def is_positive_traj_timestep(traj, i):
    return traj['rewards'][i, 0] == 10

def load_buffer(dataset_file, variant, task_aliasing_dict=None, multi_viewpoint=False, data_count_dict=None, split_pos_neg=False, split_by_traj=False, num_traj_cutoff=None):
    
    print('loading buffer data from ', dataset_file)

    index_shift = 'bridge_data_v2' in dataset_file
    task_name = str.split(dataset_file, '/')[-3 - index_shift]
    env_name = str.split(dataset_file, '/')[-4 - index_shift]
    if task_aliasing_dict and task_name in task_aliasing_dict:
        task_name = task_aliasing_dict[task_name]
    if "gs://" in dataset_file:
        with tf.io.gfile.GFile(dataset_file, "rb") as f:
            trajs = np.load(f, allow_pickle=True)
    else:
        trajs = np.load(dataset_file, allow_pickle=True)

    if num_traj_cutoff is not None and num_traj_cutoff != -1:
        print('cutting off after ', num_traj_cutoff)
        if num_traj_cutoff is not None:
            np.random.shuffle(trajs)
            trajs = trajs[:num_traj_cutoff]

    if data_count_dict is not None:
        if env_name not in data_count_dict:
            data_count_dict[env_name] = {}
        if task_name in data_count_dict[env_name]:
            data_count_dict[env_name][task_name] += len(trajs)
        else:
            data_count_dict[env_name][task_name] = len(trajs)
    if len(trajs) == 0:
        return 0, trajs
    pos_num_transitions = 0
    neg_num_transitions = 0
    num_transitions = 0

    # Count number of viewpoints
    if multi_viewpoint:
        viewpoints = trajs[0]['observations'][0].keys()
        viewpoints = [viewpoint for viewpoint in viewpoints if viewpoint.startswith('images')]
        num_viewpoints = len(viewpoints)
        print('num viewpoints', num_viewpoints)
    else:
        num_viewpoints = 1

    for traj in trajs:
        if "rewards" in traj.keys() and isinstance(traj["rewards"], list):
            traj["rewards"] = np.array(traj["rewards"]).reshape(-1, 1)
        for i in range(len(traj['observations'])):
            if split_by_traj:
                if is_positive_traj(traj):
                    pos_num_transitions += num_viewpoints
                else:
                    neg_num_transitions += num_viewpoints
            elif split_pos_neg:
                if is_positive_sample(traj, i, variant, task_name):
                    pos_num_transitions += num_viewpoints
                else:
                    neg_num_transitions += num_viewpoints
            else:
                num_transitions += num_viewpoints
        # Not using memory efficient buffer anymore
        # num_transitions += 1  # needed because of memory efficient replay buffer
        # pos_num_transitions += 1  # needed because of memory efficient replay buffer
        # neg_num_transitions += 1  # needed because of memory efficient replay buffer
        traj['task_description'] = task_name
    if split_pos_neg:
        return (pos_num_transitions, neg_num_transitions), trajs
    return num_transitions, trajs


def _reshape_image(obs):
    if len(obs.shape) == 1:
        obs = np.reshape(obs, (3, 128, 128))
        return np.transpose(obs, (1, 2, 0))
    elif len(obs.shape) == 3:
        return obs
    else:
        raise ValueError

@tf.function(jit_compile=True)
def _binarize_gripper_actions(actions):
    """Converts gripper actions from continous to binary values (0 and 1).

    We exploit that fact that most of the time, the gripper is fully open (near
    1.0) or fully closed (near 0.0). As it transitions between the two, it
    sometimes passes through a few intermediate values. We relabel those
    intermediate values based on the state that is reached _after_ those
    intermediate values.

    In the edge case that the trajectory ends with an intermediate value, we
    give up on binarizing and relabel that chunk of intermediate values as
    the last action in the trajectory.

    The scan implements the following code:

    new_actions = np.empty_like(actions)
    carry = actions[-1]
    for i in reversed(range(actions.shape[0])):
        if in_between_mask[i]:
            carry = carry
        else:
            carry = float(open_mask[i])
        new_actions[i] = carry
    """
    open_mask = actions > 0.95
    closed_mask = actions < 0.05
    in_between_mask = tf.logical_not(tf.logical_or(open_mask, closed_mask))

    is_open_float = tf.cast(open_mask, tf.float32)

    def scan_fn(carry, i):
        return tf.cond(
            in_between_mask[i],
            lambda: tf.cast(carry, tf.float32),
            lambda: is_open_float[i],
        )

    new_actions = tf.scan(
        scan_fn, tf.range(tf.shape(actions)[0]), actions[-1], reverse=True
    )
    return new_actions

def insert_data_real(variant, replay_buffer, trajs, reward_type='final_one', task_id_mapping=None,
                     env=None, image_key='images0', target_point=None, split_pos_neg=False, split_by_traj=False,
                     reward_function=None):
    if split_pos_neg:
        # print("SPLIT POSITIVE AND NEGATIVE SAMPLES")
        assert isinstance(replay_buffer, MixingReplayBuffer)
        pos_buffer, neg_buffer = replay_buffer.replay_buffers

    if split_by_traj:
        num_traj_pos = 0
        num_traj_neg = 0

    override_rew_with_gripper = variant.get('override_reward_with_gripper')
    
    for idx, traj in enumerate(trajs):
        # if idx % 8 == 0 and reward_function is not None:
        #     batch_rewards = reward_function(trajs=trajs[idx:idx+8])
        #     batch_mask = np.ones(batch_rewards.shape)
        #     batch_mask[..., -1] = 0
        #     breakpoint()
        if variant.get('lang_embedding'):
            if variant.lang_embedding.endswith("-v1-remap"):
                lang = traj['task_description'].replace('_', ' ')
            else:
                # Embed language
                lang = traj['language'][0]
            if lang is None:
                lang = ''
            lang = lang.lower()
            if '\nconfidence' in lang:
                lang = lang[:lang.index('\nconfidence')]
            lang = lang.strip()
            # print("embedding type: %s, language label: %s" % (variant.lang_embedding, lang))
            embedding = None
            if re.search("[A-Z0-9,\.\!\-]", lang):
                print('special character found in v1 langs: ', lang)
            if (not variant.filter_lang) or (lang and not re.search("[^a-z A-Z0-9,\.\!\-']", lang)):
                embedding = embed_dicts[variant.lang_embedding].get(lang)
            if embedding is None:
                print("embedding not found for '%s'" % lang)
                continue
            task = lang
        else:
            task = traj['task_description']
        
        if variant.relabel_actions:
            observations, next_observations = traj['observations'], traj['next_observations']
            movement_actions = [n_obs['state'][:6]-obs['state'][:6] for obs, n_obs in zip(observations, next_observations)]
            continuous_gripper_actions = np.float32(traj['actions'])[:, 6]
            if variant.binarize_gripper:
                gripper_actions = _binarize_gripper_actions(continuous_gripper_actions)
            else:
                gripper_actions = continuous_gripper_actions
            combined_actions=np.concatenate(
                [movement_actions, gripper_actions[:, None]],
                axis=1,
            )
            traj['actions'] = list(combined_actions)

        if variant.frame_stack == 1:
            action_queuesize = 1
        else:
            action_queuesize = variant.frame_stack - 1
        prev_actions = collections.deque(maxlen=action_queuesize)
        current_states = collections.deque(maxlen=variant.frame_stack)

        for i in range(action_queuesize):
            prev_action = np.zeros_like(traj['actions'][0])
            prev_actions.append(prev_action)

        for i in range(variant.frame_stack):
            state = traj['observations'][0]['state']
            current_states.append(state)
            
        if split_by_traj:
            positive_traj = is_positive_traj(traj)
            if positive_traj:
                num_traj_pos += 1
            else:
                num_traj_neg += 1
        
        traj_end = len(traj['observations'])
        # first process rewards, masks and mc_returns
        masks = []
        rewards = []
        if reward_function is not None:
            rewards = reward_function(traj=traj)
            masks = np.ones(rewards.shape)
            masks[..., -1] = 0
            # rewards = batch_rewards[idx % 8]
            # masks = batch_mask[idx % 8]
        else:
            if override_rew_with_gripper:
                last_open = 0
                first_close = 0
                prev_closed = False
                for i, act in enumerate(traj['actions']):
                    gripper_open = act[-1] > 0.9
                    if prev_closed and gripper_open:
                        last_open = i
                    elif not prev_closed and i > 0 and not gripper_open:
                        first_close = i
                    prev_closed = not gripper_open
                last_open = min(last_open, traj_end - 3)

            override_current_traj = override_rew_with_gripper and first_close > 0 and last_open > 0 and not prev_closed
            if override_current_traj:
                traj_end = last_open + 3

            for i in range(traj_end):
                if override_current_traj:
                    is_positive = i >= last_open
                elif not split_by_traj:
                    is_positive = is_positive_sample(traj_end, i, variant, task_name=traj['task_description'])
                else:
                    is_positive = is_positive_traj_timestep(traj, i)

                if  reward_type == 'dense':
                    reward = compute_distance_reward(traj['observations'][i]['state'][:3], target_point, reward_type)
                else:
                    if is_positive:
                        reward = 1
                    else:
                        reward = 0
                orig_rew = reward
                reward = reward * variant.reward_scale + variant.reward_shift

                if variant.term_as_rew:
                    mask = (1-orig_rew)
                elif reward_type == 'dense' or not variant.use_terminals:
                    # for the eef-reaching experiment, let's always do bootstrapping
                    mask = 1.0
                else:
                    # I am guessing this is only true assuming demos?
                    if i == len(traj['observations']) - 1:
                        mask = 0.0
                    else:
                        mask = 1.0  

                if reward_type == "original":
                    reward = traj["rewards"][i]
                    mask = 0 if reward == 10 else 1
                
                rewards.append(reward)
                masks.append(mask)
        # calculate reward to go
        monte_carlo_return = calc_return_to_go(variant, masks, jnp.array(rewards).squeeze())

        for i in range(traj_end):
            obs = dict()
            if not variant.from_states:
                obs['pixels'] = traj['observations'][i][image_key]
                obs['pixels'] = _reshape_image(obs['pixels'])[..., np.newaxis]
            if variant.add_states:
                obs['state'] = traj['observations'][i]['state']
            if variant.add_prev_actions:
                obs['prev_action'] = np.stack(prev_actions, axis=-1)

            action_i = traj['actions'][i]
            action_iplus1 = traj['actions'][i + 1] if i < len(traj['actions']) - 1 else action_i
            if variant.rescale_actions:
                action_i = env.rescale_actions(action_i)
                action_iplus1 = env.rescale_actions(action_iplus1)
                action_i[..., -1] = action_i[..., -1] * variant.smooth_gripper
                action_iplus1[..., -1] = action_iplus1[..., -1] * variant.smooth_gripper
            elif variant.smooth_gripper != 1:
                action_i[..., -1] = action_i[..., -1] * variant.smooth_gripper + (1 - variant.smooth_gripper) / 2
                action_iplus1[..., -1] = action_iplus1[..., -1] * variant.smooth_gripper + (1 - variant.smooth_gripper) / 2

            # if np.isnan(action_i).sum() > 0:
            #     print(action_i)
            #     print("fuga")
            #     import pdb; pdb.set_trace()
            prev_actions.append(action_i)

            current_state = traj['next_observations'][i]['state']
            current_states.append(current_state)  # do not delay state, therefore use i instead of i

            next_obs = dict()

            if not variant.from_states:
                next_obs['pixels'] = traj['next_observations'][i][image_key]
                next_obs['pixels'] = _reshape_image(next_obs['pixels'])[..., np.newaxis]
                # if i == 0:
                #     obs['pixels'] = np.tile(obs['pixels'], [1, 1, 1, variant.frame_stack])
            if variant.add_states:
                next_obs['state'] = traj['next_observations'][i]['state'] #np.stack(current_states, axis=-1)
            if variant.add_prev_actions:
                next_obs['prev_action'] = np.stack(prev_actions, axis=-1)

            if variant.get('lang_embedding') is None:
                if task_id_mapping is not None:
                    num_tasks = len(task_id_mapping.keys())
                    task_id_vec = np.zeros(num_tasks, np.float32)[None]
                    task_id_vec[:, task_id_mapping[traj['task_description']]] = 1
                    obs['task_id'] = task_id_vec
                    next_obs['task_id'] = task_id_vec
            else:
                obs['language'] = next_obs['language'] = embedding
            if split_pos_neg:
                if positive_traj:
                    trajectory_id=pos_buffer._traj_counter
                else:
                    trajectory_id=neg_buffer._traj_counter
            else:
                trajectory_id=replay_buffer._traj_counter

            insert_dict = dict(
                 observations=obs,
                 actions=action_i,
                 next_actions=action_iplus1,
                 rewards=rewards[i],
                 next_observations=next_obs,
                 masks=masks[i],
                 dones=bool(i == traj_end - 1),
                 trajectory_id=trajectory_id,
                 mc_returns=monte_carlo_return[i],
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
                pos_buffer.add_task(task)
            else:
                neg_buffer.increment_traj_counter()
                pos_buffer.add_task(task)
        else:
            replay_buffer.increment_traj_counter()
            replay_buffer.add_task(task)
        
    if split_by_traj:
        print('num traj pos', num_traj_pos)
        print('num traj neg', num_traj_neg)


RETURN_TO_GO_DICT = dict()

def calc_return_to_go(variant, masks, rewards):
    # rewards_str = str(rewards.tolist())
    if False: # rewards_str in RETURN_TO_GO_DICT.keys():
        pass # reward_to_go = RETURN_TO_GO_DICT[rewards_str]
    else:
        reward_to_go = [0]*len(rewards)
        gamma = 1
        prev_return = 0
        len_r = len(rewards)
        for i in range(len(rewards)):
            reward_to_go[-i-1] = rewards[-i-1] + gamma * prev_return * masks[-i-1]
            gamma = variant.discount
            prev_return = reward_to_go[-i-1]
        reward_to_go = np.array(reward_to_go)
        # RETURN_TO_GO_DICT[rewards_str] = reward_to_go
    return reward_to_go

# Experimental code for argmax policy
import jax
import jax.numpy as jnp
@jax.jit
def get_q_value(actions, obs_dict, critic_encoder, critic_decoder):
    if critic_encoder.batch_stats is not None:
        embed_obs, _ = critic_encoder.apply_fn({'params': critic_encoder.params, 'batch_stats': critic_encoder.batch_stats}, obs_dict, mutable=['batch_stats'])
    else:    
        embed_obs = critic_encoder.apply_fn({'params': critic_encoder.params}, obs_dict)
    if critic_decoder.batch_stats is not None:
        q_pred, _ = critic_decoder.apply_fn({'params': critic_decoder.params, 'batch_stats': critic_decoder.batch_stats}, embed_obs, actions, mutable=['batch_stats'])
    else:    
        q_pred = critic_decoder.apply_fn({'params': critic_decoder.params}, embed_obs, actions, training=False)
        
    return q_pred

def add_batch_dim(x):
    return jax.tree_map(lambda x: x[None], x)

@jax.jit
def sample_from_agent_gd(critic_encoder, critic_decoder, obs_dict, init_action):
    action = init_action
    max_qs = []
    obs_dict = jax.tree_map(lambda x: x[None], obs_dict)
    max_qs.append(get_q_value(action[None], obs_dict, critic_encoder, critic_decoder)[0, 0])
    
    mins = [-0.05, -0.05, -0.05, -0.25, -0.25, -0.25, 0.]
    maxes = [0.05, 0.05, 0.05, 0.25, 0.25, 0.25, 1.0]
    mins = [-1] * 7
    maxes = [1] * 7
    # mins = [0, 0, 0, 0, 0, 0, -1]
    # maxes = [0, 0, 0, 0, 0, 0, 1]
    diffs = [(maxes[i] - mins[i]) for i in range(7)]

    def get_q_grid(actions):
        def q(a):
            a = a[None]
            return get_q_value(a, obs_dict, critic_encoder, critic_decoder)[0, 0]
        return jax.vmap(q)(actions)

    # for i in range(7):
    #     X = jnp.linspace(mins[i], maxes[i], 20)
    #     action_tiled = jnp.tile(action, (len(X), 1))
    #     action_tiled = action_tiled.at[:, i].set(X)
    #     qs = get_q_grid(action_tiled)
    #     max_qs.append(jnp.max(qs))
    #     action = action_tiled[jnp.argmax(qs)]
    
    for i in range(7):
        X = jnp.linspace(-1 * diffs[i] / 10, diffs[i] / 10, 20)
        action_tiled = jnp.tile(action, (len(X), 1))
        action_tiled = action_tiled.at[:, i].add(X)
        qs = get_q_grid(action_tiled)
        max_qs.append(jnp.max(qs))
        action = action_tiled[jnp.argmax(qs)]
    return action, max_qs

def agent_gd_actions(agent, obs_dict):
    if obs_dict['pixels'].shape[0] == 1:
        obs_dict = jax.tree_map(lambda x: x[0], obs_dict)
    init_action = agent.eval_actions(add_batch_dim(obs_dict))[0]
    new_action, max_qs = sample_from_agent_gd(agent._critic_encoder, agent._critic_decoder, obs_dict, init_action)
    print(max_qs[0], max_qs[-1])
    print(init_action, new_action)
    return jnp.clip(new_action, -1, 1)
