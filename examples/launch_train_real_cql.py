import argparse
import sys
import imp
from vptr.utils.general_utils import AttrDict

from examples.train_pixels_real import main
from vptr.utils.launch_util import parse_training_args
import os
import json


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', default=42, help='Random seed.', type=int)
    parser.add_argument('--launch_group_id', default='', help='group id used to group runs on wandb.')
    parser.add_argument('--eval_episodes', default=10,
                        help='Number of episodes used for evaluation.', type=int)
    parser.add_argument('--log_interval', default=1000, help='Logging interval.', type=int)
    parser.add_argument('--eval_interval', default=5000, help='Eval interval.', type=int)
    parser.add_argument('--checkpoint_interval', default=20000, help='checkpoint interval.', type=int)
    parser.add_argument('--batch_size', default=64, help='Mini batch size.', type=int)
    parser.add_argument('--online_start', default=int(5e5), help='Number of training steps after which to start training online.', type=int)
    parser.add_argument('--max_steps', default=int(5e5), help='Number of training steps.', type=int)
    parser.add_argument('--tqdm', default=1, help='Use tqdm progress bar.', type=int)
    parser.add_argument('--save_video', action='store_true', help='Save videos during evaluation.')
    parser.add_argument('--use_negatives', action='store_true', help='Use negative_data')
    parser.add_argument('--reward_scale', default=11.0, help='Scale for the reward', type=float)
    parser.add_argument('--reward_shift', default=-1, help='Shift for the reward', type=float)
    parser.add_argument('--reward_type', default='final_one', help='reward type')

    parser.add_argument('--frame_stack', default=1, help='Number of frames stacked', type=int)
    parser.add_argument('--add_states', default=1, help='whether to add low-dim states to the obervations',  type=int)
    parser.add_argument('--add_prev_actions', default=0, help='whether to add low-dim previous actions to the obervations', type=int)

    parser.add_argument('--dataset', default='online_reaching_pixels', help='name of dataset')
    parser.add_argument('--target_dataset', default='', help='name of dataset', type=str)

    parser.add_argument('--online_mixing_ratio', default=0.5,
                        help='fraction of batch composed of old data to be used, the remainder part is newly collected data',
                        type=float)
    parser.add_argument('--target_mixing_ratio', default=0.9,
                        help='fraction of batch composed of bridge data, the remainder is target data',
                        type=float)
    parser.add_argument('--num_target_traj', default=-1,
                        help='num trajectories used for the target task',
                        type=int)
    parser.add_argument("--multi_viewpoint", default=1, help="whether to use multiple camreas", type=int)
    parser.add_argument('--negatives_nofinal_bootstrap', action='store_true', help='apply bootstrapping at last time step of negatives')

    parser.add_argument('--trajwise_alternating', default=1,
                        help='alternate between training and data collection after each trajectory', type=int)
    parser.add_argument('--restore', action='store_true', help='whether to restore weights')
    parser.add_argument('--restore_path', default='', help='folder inside $EXP where weights are stored')
    parser.add_argument('--absolute_restore_path', default='', help='Absolute file path to the weights file inside $EXP folder')
    parser.add_argument('--placeholder_task', default='', help='Task to replace in restored task id mapping')
    parser.add_argument('--only_add_success', action='store_true', help='only add successful traj to buffer')

    parser.add_argument('--wandb_project', default='cql_real', help='wandb project')
    parser.add_argument('--wandb_user', default='frederik', help='wandb user config (this is not the actual wanddb username)')
    parser.add_argument("--use_terminals", default=1, help="whether to use terminals", type=int)

    parser.add_argument('--from_states', action='store_true', help='only use states, no images')
    parser.add_argument('--lang_embedding', default=None, help='use a language embedding instead of task ids')
    parser.add_argument('--start_online_updates', default=1000, help='number of steps to collect before starting online updates', type=int)
    parser.add_argument('--online_from_scratch', action='store_true', help='train online from scratch.')

    parser.add_argument('--stochastic_data_collect', default=1, help='sample from stochastic policy for data collection.', type=int)

    parser.add_argument('--algorithm', default='cql_encodersep', help='type of algorithm')

    parser.add_argument('--team', default='', help='team to use for wandb')
    parser.add_argument('--prefix', default='', help='prefix to use for wandb')
    parser.add_argument('--config', default='examples/configs/offline_pixels_default_real.py', help='File path to the training hyperparameter configuration.')
    parser.add_argument("--azure", action='store_true', help="run on azure")
    parser.add_argument("--offline_only_restore_onlinebuffer", default='', help="location of saved replay buffer as .npy file", type=str)
    parser.add_argument("--offline_only_restore_onlinedata", default='', help="location of saved online data, extracted as .npy file", type=str)
    parser.add_argument("--online_pos_neg_ratio", default=-1, help="ratio of positives and negatives for online data", type=float)
    parser.add_argument("--split_by_traj_target", action='store_true', help="ratio of positives and negatives for online data")
    parser.add_argument("--split_by_traj_target_ratio", default=0.5, help="ratio of positives and negatives for online data", type=float)

    parser.add_argument('--relabel_actions', default=0, help='relabel actions to difference between robot states')
    parser.add_argument('--binarize_gripper', default=0, help='Binarize gripper actions')
    parser.add_argument('--rescale_actions', default=1, help='rescale actions to so that action-bounds are within +-1', type=int)
    parser.add_argument('--normalize_actions', default=0, help='rescale actions to so that action-bounds are within +-1', type=int)
    parser.add_argument('--start_transform', default='openmicrowave', help='start transform to use', type=str)
    
    #environment
    parser.add_argument('--episode_timelimit', default=40, help='prefix to use', type=int)

    parser.add_argument('--restore_reward_path', default='', help='File path to the weights file of the reward function inside $EXP folder')
    parser.add_argument('--file_system', default='', help='local or azure-blobfuse')

    parser.add_argument('--num_eval_tasks', default=-1, help='nubmer of eval tasks, if -1 infer from dataset', type=int)
    parser.add_argument('--annotate_with_classifier', default=0,
                        help='whether to annotate bridge data with classifier rewads', type=int)

    parser.add_argument('--num_final_reward_steps', default=1, help='number of final reward timesteps', type=int)
    parser.add_argument('--term_as_rew', type=int, default=1)

    parser.add_argument('--offline_finetuning_start', default=-1, help='when to start offline finetuning', type=int)
    
    parser.add_argument('--cql_alpha_online_finetuning', default=-1.0, help='alpha for finetuning', type=float)
    parser.add_argument('--streaming_interval', help='stream the online data per how many gradient steps', type=int)
    parser.add_argument('--max_streaming_iter', help='how many times to stream the data', type=int)
    parser.add_argument('--debug', type=int, default=0, help='debug mode')
    parser.add_argument('--online_cql_alpha', type=float, help='cql_alpha for online, if None then will be same as cql_alpha')
    parser.add_argument('--freeze_encoders', action='store_true', help='whether to stop gradients from propagating into the encoders.')
    parser.add_argument('--encoder_key', default='agent/value/params/encoder', help='Key for encoder params in pretrained encder', type=str)
    parser.add_argument('--num_demo', type=int, help='num_demo for tk1_targetdomain_openmicro_3')
    parser.add_argument('--tpu_port', type=int, default=8476, help='define a unique port num for each tpu process to run multipule jobs in single instance')
    parser.add_argument('--bound_q_with_mc', action='store_true', help='target_q = max(MC, target_q)')

    parser.add_argument('--bc_end', default=0, help='iterations of BC training to run for the actor', type=int)
    parser.add_argument('--filter_lang', action='store_true', help='reject language labels which contain special characters')
    parser.add_argument('--override_reward_with_gripper', action='store_true', help='begin giving reward after the gripper opens for the final time')
    parser.add_argument('--smooth_gripper', default=1.0, help='multiplies the gripper action to prevent pre-tanh activations approaching infinity. requires relabel_actions', type=float)
    
    parser.add_argument('--vip_reward_encoder', default=None, help='path to VIP encoder checkpoint to use VIP embedding distance as a reward function')
    parser.add_argument('--icvf_reward_encoder', default=None, help='path to ICVF encoder checkpoint to use ICVF embedding distance as a reward function')
    parser.add_argument('--icvf_reward_agent', default=None, help='path to full ICVF agent checkpoint to use ICVF value function as a reward function')
    parser.add_argument('--no_reward_preprocess', action='store_true', help='do not preprocess data for rewards (use individual trajs instead of mean)')
    parser.add_argument('--goal_conditioned', action='store_true', help='condition on last few frames in trajectory rather than task')
    parser.add_argument('--goal_encoder_type', default='same', help='encoder for embedding goal image. Set to "same" to use same type as state encoder')
    parser.add_argument('--pretrained_goal_encoder', default=None, help='path to pretrained encoder for embedding goal image')
    # algorithm args:
    train_args_dict = dict(
        actor_lr= 1e-4,
        critic_lr= 3e-4,
        temp_lr= 3e-4,
        decay_steps= None,
        hidden_dims= (1024, 1024, 1024),
        cnn_features= (32, 32, 32, 32),
        cnn_strides= (2, 1, 1, 1),
        cnn_padding= 'VALID',
        latent_dim=1024,
        discount= 0.96,
        cql_alpha= 0.0,
        tau= 0.005,
        backup_entropy= False,
        target_entropy= None,
        critic_reduction= 'min',
        dropout_rate= None,
        init_temperature= 1.0,
        max_q_backup= True,
        pretrained_encoder='',
        policy_encoder_type='resnet_small',
        encoder_type='resnet_small',
        encoder_resize_dim=128,
        encoder_norm= 'batch',
        dr3_coefficient= 0.0,
        use_spatial_softmax=False,
        softmax_temperature=-1,
        use_spatial_learned_embeddings=True,
        share_encoders=False,
        action_jitter_scale=0.0,
        use_bottleneck=True,
        use_language_bottleneck=False,
        use_action_sep=False,
        use_basis_projection=False,
        basis_projection_coefficient=0.0,
        use_film_conditioning=False,
        use_multiplicative_cond=False,
        target_entropy_factor=1.0,
        policy_use_multiplicative_cond=False,
        use_language_sep_critic=False,
        use_pixel_sep_critic=False,
        include_state_critic=False,
        include_state_actor=True,
        use_gaussian_policy=False,
        use_mixture_policy=False,
        use_bc_policy=False,
        reward_regression=False,
        mc_regression=False,
        min_q_version=3,
        std_scale_for_gaussian_policy=0.05,
        q_dropout_rate=0.0,
        freeze_batch_stats=False,
    )

    variant, args = parse_training_args(train_args_dict, parser)
    variant['train_kwargs']['debug'] = args.debug
    variant['train_kwargs']['freeze_encoders'] = args.freeze_encoders
    variant['train_kwargs']['bound_q_with_mc'] = args.bound_q_with_mc

    
    filesys = 'AZURE_' if args.file_system == 'azure' else ''
    prefix = '/data/spt_data/experiments/' if args.azure else os.environ[filesys + 'EXP']
    if args.restore_reward_path:
        local_reward_restore_path = os.environ[filesys + 'EXP'] + '/' + args.restore_reward_path
        full_reward_restore_path = prefix + '/' + args.restore_reward_path
        config_file_reward = '/'.join(local_reward_restore_path.split('/')[:-1]) + '/config.json'
        with open(config_file_reward) as config_file:
            variant_reward = json.load(config_file)
        variant_reward = AttrDict(variant_reward)
        variant.variant_reward = variant_reward
    else:
        full_reward_restore_path = ''

    variant.restore_reward_path = full_reward_restore_path
    if variant.policy_encoder_type == 'same':
        variant.policy_encoder_type = variant.encoder_type
    
    if variant.use_gaussian_policy:
        variant.rescale_actions = 0
        variant.normalize_actions = 1

    if not args.azure:
        main(variant)
        sys.exit()
    else:
        from doodad.wrappers.easy_launch import sweep_function, save_doodad_config
        def train(doodad_config, default_params):
            main(variant)
            save_doodad_config(doodad_config)

        params_to_sweep = {}
        mode = 'azure'
        sweep_function(
            train,
            params_to_sweep,
            default_params={},
            log_path=args.prefix,
            mode=mode,
            use_gpu=True,
            num_gpu=1,
        )
