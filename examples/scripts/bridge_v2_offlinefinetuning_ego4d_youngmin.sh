proj_name=icvf_bridge_v2

#!/bin/bash
export WANDB_EXP=/nfs/nfs2/users/youngminpark/vptr/experiment_output/wandb/$proj_name/
export EXP=/nfs/nfs2/users/youngminpark/vptr/experiment_output/
export DATA=gs://rail-tpus-mitsuhiko-central2/bridge_data_all_numpy_lang_224
export PYTHONPATH=$PYTHONPATH:.

export TPU_VISIBLE_DEVICES=0,1
export TPU_CHIPS_PER_HOST_BOUNDS=1,2,1
export TPU_HOST_BOUNDS=1,1,1
export TPU_MESH_CONTROLLER_ADDRESS=localhost:8476
export TPU_MESH_CONTROLLER_PORT=8476

tpu_id=0
tpu_port=$(( $tpu_id+8820 ))

encoder=resnetv2-50-1
prefix=vptr_cucumber_test

# dataset=all_pickplace_v1_except_tk6
# dataset=all_openclose_v1_except_tk1
dataset=bridge_v1_test_small
# target_dataset=tk1_targetdomain_openmicro_combined_0523
target_dataset=tk6_targetdomain_cucumberpot_combined_0515
# target_dataset=test_v1
target_mixing_ratio=0.9
alpha=5

sudo chmod 777 -R /tmp/tpu_logs
sudo rm /tmp/libtpu_lockfile

# for target_mixing_ratio in 0.7 0.9
# do
# for alpha in 10 5
# do
# TPU_VISIBLE_DEVICES=${tpu_id}
# --lang_embedding grif-text \
# --restore_path /nfs/nfs1/users/derekguo/jaxrl2_private/experiment_output/ego4d_v1_grif_2023_08_30_06_55_41_0000--s-42/checkpoint200000 \
# --restore_path /nfs/nfs1/users/chet/jaxrl2_private/experiment_output/vip_v1_croissant_less_demos_combined_2023_08_04_21_38_50_0000--s-42/checkpoint200000 \
# --restore_path /nfs/nfs1/users/derekguo/jaxrl2_private/experiment_output/scratch_grif_lcbc_croissant_2023_09_11_04_58_50_0000--s-42/checkpoint200000 \
python3 examples/launch_train_real_cql.py \
--prefix $prefix \
--wandb_project $proj_name \
--algorithm cql_encodersep_parallel \
--cql_alpha $alpha \
--encoder_type $encoder \
--policy_encoder_type $encoder \
--encoder_resize_dim 224 \
--pretrained_encoder examples/checkpoints/icvf/icvf_224 \
--dataset $dataset \
--target_dataset $target_dataset \
--target_mixing_ratio $target_mixing_ratio \
--batch_size 64 \
--multi_viewpoint 0 \
--add_prev_actions 0 \
--use_action_sep 1 \
--use_basis_projection 0 \
--discount 0.96 \
--max_q_backup 1 \
--basis_projection_coefficient 0.0 \
--use_multiplicative_cond 0 \
--num_final_reward_steps 3 \
--term_as_rew 1 \
--encoder_norm group \
--use_spatial_learned_embeddings 1 \
--target_entropy_factor 1.0 \
--policy_use_multiplicative_cond 0 \
--use_pixel_sep_critic 1 \
--min_q_version 3 \
--q_dropout_rate 0.0 \
--offline_finetuning_start 200000 \
--checkpoint_interval 50000 \
--eval_interval 5000 \
--max_steps 400000 \
--online_start 400000 \
--bound_q_with_mc \
--relabel_actions 1 \
--tpu_port $tpu_port \
--actor_lr 1e-4 \
--critic_lr 3e-4 \
--temp_lr 3e-4 \
--team youngminpark
# --hidden_dims 256 256 256 \
# --latent_dim 256 \