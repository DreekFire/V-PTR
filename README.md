# Robotic Offline RL from Internet Videos via Value-Function Pre-Training #

### Installation ###

pip install from requirements.txt

### Data ###

The location of the data is specified in examples/configs/bridge_all_dataset.py. This file contains functions that return lists of paths to train and validation trajectories, stored in numpy format. Each path will be prepended with the DATA environment variable defined in the launch script. It is currently set up for the bridge v1 dataset stored on our buckets.

dataset_config_real.py contains an ALIASING_DICT which maps the bottommost folder name (directly containing the "train" and "val" directories) to a task name. If not using language conditioning, then each unique task name is mapped to a task ID.

### Language Conditioning ###

Language conditioning was done by passing the task specifications through an off-the-shelf language model, and storing the dictionary which maps task specifications to embedding vectors using pickle. The ones we used are present in notebooks/final_embeddings

### Training ###

An example training script used to produce our results is available in bridge_v2_offlinefinetuning_ego4d.sh

The key parameters used during our experiments were:

pretrained_encoder: a checkpoint of an ICVF model produced by the first phase of training decsribed in our paper. This is produced by training an ICVF on the Ego4D database using the github.com/dibyaghosh/icvf_video repository. Leave empty to train from scratch (equivalent to PTR).

dataset: the multi-task robot dataset for the second phase of training. We use a set of tasks that does not contain our target task but is performed in the same environment.

target_dataset: the single-task target dataset for the final phase of training.
