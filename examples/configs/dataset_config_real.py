import os

train_dataset_single_task = [os.environ['DATA'] + '/robonetv2/toykitchen_numpy_shifted/bridge_data_numpy_shifted_split_uint8_5-26/toykitchen1/put_broccoli_in_pot_cardboardfence/train/out.npy']
eval_dataset_single_task = [os.environ['DATA'] + '/robonetv2/toykitchen_numpy_shifted/bridge_data_numpy_shifted_split_uint8_5-26/toykitchen1/put_broccoli_in_pot_cardboardfence/val/out.npy']


train_dataset_single_task_openmicro = ['robonetv2/toykitchen_numpy_shifted/bridge_data_numpy_shifted_split_uint8_5-26/toykitchen5/open_microwave/train/out.npy']
eval_dataset_single_task_openmicro = ['robonetv2/toykitchen_numpy_shifted/bridge_data_numpy_shifted_split_uint8_5-26/toykitchen5/open_microwave/val/out.npy']


train_tasks = val_tasks = [
    'put_broccoli_in_pot_cardboardfence',
    'put_carrot_on_plate_cardboardfence',
    'put_broccoli_in_pot_or_pan',
    'put_broccoli_in_bowl',
    'put_carrot_on_plate',
    'put_sushi_on_plate',
    'put_corn_into_bowl',
    'put_sweet_potato_in_pan_which_is_on_stove',
    'put_sweet_potato_in_pan_which_is_on_stove_distractors',
    'put_sweet_potato_in_pot_which_is_in_sink_distractors',

    'take_broccoli_out_of_pan_cardboardfence',
    'take_carrot_off_plate_cardboardfence',
    'take_broccoli_out_of_pan',
    'take_can_out_of_pan',
    'take_carrot_off_plate',
    'take_lid_off_pot_or_pan',
]

# train_dataset_11_task = [f'robonetv2/toykitchen_numpy_shifted/bridge_data_numpy_shifted_split_uint8/toykitchen1/{task}/train/out.npy' for task in train_tasks]
train_dataset_11_task = [f'robonetv2/toykitchen_numpy_shifted/bridge_data_numpy_shifted_split_uint8_5-26/toykitchen1/{task}/train/out.npy' for task in train_tasks]

# eval_dataset_11_task = [f'robonetv2/toykitchen_numpy_shifted/bridge_data_numpy_shifted_split_uint8/toykitchen1/{task}/val/out.npy' for task in val_tasks]
eval_dataset_11_task = [f'robonetv2/toykitchen_numpy_shifted/bridge_data_numpy_shifted_split_uint8_5-26/toykitchen1/{task}/val/out.npy' for task in val_tasks]

train_tasks_tk5 = val_tasks_tk5 = [
    'close_fridge',
    'close_microwave',
    'open_fridge',
    'open_microwave',
    'open_cabinet',
    'open_low_fridge',
    'open_oven',
    'close_cabinet',
    'close_low_fridge',
    'close_oven',
]

train_tasks_tk6 = val_tasks_tk6 = [
    'close_microwave',
    'open_microwave',
    'open_oven',
    'close_oven',
    'open_fridge',
    'close_fridge',
]


train_dataset_openclose = [f'robonetv2/toykitchen_numpy_shifted/bridge_data_numpy_shifted_split_uint8_5-26/toykitchen5/{task}/train/out.npy' for task in train_tasks_tk5]
train_dataset_openclose.extend([f'robonetv2/toykitchen_numpy_shifted/bridge_data_numpy_shifted_split_uint8_5-26/toykitchen1/{task}/train/out.npy' for task in train_tasks_tk6])
train_dataset_openclose.extend([f'robonetv2/toykitchen_numpy_shifted/bridge_data_numpy_shifted_split_uint8_5-26/toykitchen6/{task}/train/out.npy' for task in train_tasks_tk6])
eval_dataset_openclose = [f'robonetv2/toykitchen_numpy_shifted/bridge_data_numpy_shifted_split_uint8_5-26/toykitchen5/{task}/val/out.npy' for task in train_tasks_tk5]
eval_dataset_openclose.extend([f'robonetv2/toykitchen_numpy_shifted/bridge_data_numpy_shifted_split_uint8_5-26/toykitchen1/{task}/val/out.npy' for task in train_tasks_tk6])
eval_dataset_openclose.extend([f'robonetv2/toykitchen_numpy_shifted/bridge_data_numpy_shifted_split_uint8_5-26/toykitchen6/{task}/val/out.npy' for task in train_tasks_tk6])

ALIASING_DICT = {
    "flip_pot_upright_in_sink_distractors": "flip_pot_upright_which_is_in_sink",
    "put_eggplant_into_pan": "put_eggplant_in_pot_or_pan",
    "put_eggplant_into_pot_or_pan": "put_eggplant_in_pot_or_pan",
    # "faucet_front_to_left": "turn_faucet_front_to_left",
    "put_cup_from_counter_or_drying_rack_into_sink": "put_cup_from_anywhere_into_sink",
    "put_green_squash_into_pot_or_pan": "put_green_squash_in_pot_or_pan",
    "turn_lever_vertical_to-front": "turn_lever_vertical_to_front",
    "turn_lever_vertical_to_front_distractors": "turn_lever_vertical_to_front",
    "put_pan_from_sink_into_drying_rack": "put_pot_or_pan_from_sink_into_drying_rack",
    "put_corn_in_pan_which_is_on_stove_distractors": "put_corn_into_pot_or_pan",
    "put_corn_in_pan_which-is_on_stove_distractors": "put_corn_into_pot_or_pan",
    "put_corn_in_pot_which_is_in_sink_distractors": "put_corn_into_pot_or_pan",
    "take_broccoli_out_of_pan": "take_broccoli_out_of_pot_or_pan",
    "put_pepper_in_pan": "put_pepper_in_pot_or_pan",
    "put_sweet_potato_in_pot_which_is_in_sink_distractors": "put_sweet_potato_in_pot",
    "put_sweet_potato_in_pan_which_is_on_stove_distractors": "put_sweet_potato_in_pan_which_is_on_stove",
    "put_pan_in_sink": "put_pot_or_pan_in_sink",
    "put_pot_in_sink": "put_pot_or_pan_in_sink",
    "put_pan_from_stove_to_sink": "put_pot_or_pan_in_sink",
    "put_pot_on_stove_which_is_near_stove_distractors": "put_pot_or_pan_on_stove",
    "put_pan_on_stove_from_sink": "put_pot_or_pan_on_stove",

    "put_broccoli_in_pot_cardboardfence": "put_broccoli_in_pot_or_pan",
    "put_carrot_on_plate_cardboardfence": "put_carrot_on_plate",
    "take_broccoli_out_of_pan_cardboardfence": "take_broccoli_out_of_pot_or_pan",
    "take_carrot_off_plate_cardboardfence": "take_carrot_off_plate",

    'open_cabinet': 'open_door',
    'open_oven': 'open_door',
    'open_low_fridge': 'open_door',
    'open_fridge': 'open_door',
    'open_microwave': 'open_door',
    'open_microwave_2': 'open_door',
    'open_microwave_2_5pos_20neg': 'open_door',
    'open_microwave_2_1pos_20neg': 'open_door',
    'open_microwave_2_few_demo': 'open_door',
    'open_microwave_2_few_demo_with_neg': 'open_door',
    'open_microwave_3': 'open_door',
    'open_microwave_3_with_neg': 'open_door',
    '1014_all': 'open_door',
    'open_microwave_4': 'open_door',
    'open_microwave_5': 'open_door',
    'open_microwave_6': 'open_door',
    'open_1117': 'open_door',
    'open_microwave_8': 'open_door',
    'open_microwave_idling': 'open_door',
    'open_microwave_0419': 'open_door',
    'open_microwave_0425': 'open_door',
    'open_microwave_idling_0425': 'open_door',

    'close_oven': 'close_door',
    'close_fridge': 'close_door',
    'close_cabinet': 'close_door',
    'close_low_fridge': 'close_door',
    'close_microwave': 'close_door',
    'close_microwave_2': 'close_door',
    'close_microwave_3': 'close_door',
    'close_microwave_4': 'close_door',
    'close_microwave_6': 'close_door',
    'close_microwave_8': 'close_door',
    'open_microwave_0523':'open_door',
    'open_microwave_distractors_0523':'open_door',
    'close_microwave_0523':'close_door',
    'close_microwave_distractors_0523':'close_door',

    "faucet_front_to_left": "turn_faucet_left",
    "turn_faucet_front_to_left": "turn_faucet_left",
    "turn_faucet_left_56": "turn_faucet_left",
    "turn_faucet_right_55": "turn_faucet_right",
    "move_faucet_front_to_left": "turn_faucet_left",
    "turn_faucet_front_to_right": "turn_faucet_right",
    "turn_faucet_left":"turn_faucet_left",
    "turn_faucet_right":"turn_faucet_right",
    
    'put_cucumber_in_orange_pot_0515':'put_cucumber_in_orange_pot',
    'put_knife_in_orange_pan_0515':'put_knife_in_orange_pan',
    'put_sweet_potato_on_plate_0515':'put_sweet_potato_on_plate',
    'put_cucumber_in_orange_pot_distractors_0515':'put_cucumber_in_orange_pot',
    'put_knife_in_orange_pan_distractors_0515':'put_knife_in_orange_pan',
    'put_sweet_potato_on_plate_distractors_0515':'put_sweet_potato_on_plate',
    'take_croissant_out_of_colander_0515':'take_croissant_out_of_colander',
    'take_croissant_out_of_colander_distractors_0515':'take_croissant_out_of_colander',

    # Toggle comments when using grif-text-v1-remap embedding
    
    # 'take_banana_out_of_colander_0530':'take_banana_out_of_colander',
    # 'take_blueberries_out_of_colander_0530':'take_blueberries_out_of_colander',
    # 'take_carrot_out_of_colander_0530':'take_carrot_out_of_colander',
    # 'take_challah_out_of_colander_0530':'take_challah_out_of_colander',
    # 'take_croissant_out_of_colander_0530':'take_croissant_out_of_colander',
    # 'take_greengrapes_out_of_colander_0530':'take_green_grapes_out_of_colander',

    'take_banana_out_of_colander_0530':'take_croissant_out_of_colander',
    'take_blueberries_out_of_colander_0530':'take_croissant_out_of_colander',
    'take_carrot_out_of_colander_0530':'take_croissant_out_of_colander',
    'take_challah_out_of_colander_0530':'take_croissant_out_of_colander',
    'take_croissant_out_of_colander_0530':'take_croissant_out_of_colander',
    'take_greengrapes_out_of_colander_0530':'take_croissant_out_of_colander',

    # 'pick_up_red_srewdriver': 'pick_up_red_screwdriver',

    # 'close_large4fbox_flaps': 'close_large_box_flaps',
    # 'close_small4fbox_flaps': 'close_small_box_flaps',
    # 'open_large4fbox_flaps': 'open_large_box_flaps',
    # 'pick_up_bowl_and_put_in_small4fbox': 'pick_up_bowl_and_put_in_small_box',
    # 'twist_knob_start_vertical_clockwise90': 'twist_knob_clockwise',
    # 'close_brown1fbox_flap': 'close_brown_box_flap',
    # 'close_white1fbox_flap': 'close_white_box_flap',
    # 'open_brown1fbox_flap': 'open_brown_box_flap',
    # 'open_small4fbox_flaps': 'open_small_box_flaps',
    # 'open_white1fbox_flap': 'open_white_box_flap',
    # 'put_banana_in_pot_cardboard_fence': 'put_banana_in_pot',
    # 'put_bowl_on_plate_cardboard_fence': 'put_bowl_on_plate',
    # 'put_carrot_in_pot_cardboard_fence': 'put_carrot_in_pot',
    # 'put_knife_in_pot_cardboard_fence': 'put_knife_in_pot',
    # 'put_knife_on_cutting_board_cardboard_fence': 'put_knife_on_cutting_board',
    # 'put_lid_on_pot_cardboardfence': 'put_lid_on_pot',
    # 'put_pear_in_bowl_cardboardfence': 'put_pear_in_bowl',
    # 'put_potato_in_pot_cardboard_fence': 'put_potato_in_pot',
    # 'put_sushi_in_pot_cardboard_fence': 'put_sushi_in_pot',
    # 'take_bowl_off_plate_cardboard_fence': 'take_bowl_off_plate',
    # 'take_carrot_out_of_pot_cardboard_fence': 'take_carrot_out_of_pot',
    # 'take_lid_off_pot_cardboardfence': 'take_lid_off_pot',
    # 'take_sushi_out_of_pot_cardboard_fence': 'take_sushi_out_of_pot',
    # 'topple_basil_bottle_cardboard_fence': 'topple_basil_bottle',
    # 'topple_hot_sauce_bottle_cardboard_fence': 'topple_hot_sauce_bottle',
    # 'topple_metal_pot_cardboard_fence': 'topple_metal_pot',
    # 'upright_basil_bottle_cardboard_fence': 'upright_basil_bottle',
    # 'upright_hot_sauce_bottle_cardboard_fence': 'upright_hot_sauce_bottle',
    # 'upright_metal_pot_cardboard_fence': 'upright_metal_pot',
    # 'close_microwave_0515': 'close_door',
    # 'close_microwave_distractors_0515': 'close_door',
    # 'microwave_closed_positive': 'close_door',
    # 'microwave_open_positive': 'open_door',
    # 'open_microvave_distractors_0515': 'open_door',
    # 'open_microwave_0515': 'open_door',
    # 'cloth_0803': 'fold_or_unfold_cloth',
    # 'open_the_drawer_0805': 'open_drawer',
    # 'put_cucumber_in_orange_pot_distractors': 'put_cucumber_in_orange_pot',
    # 'put_cucumber_in_orange_pot_simple': 'put_cucumber_in_orange_pot',
    # 'put_sweet_potato_on_plate_distractors': 'put_sweet_potato_on_plate',
    # 'put_sweet_potato_on_plate_simple': 'put_sweet_potato_on_plate',
    # 'sweep_into_pile_0805': 'sweep_into_pile',
    # 'take_croissant_out_of_colander_simple': 'take_croissant_out_of_colander',

    # 'put_banana_in_pot_cardboard_fence': 'put_banana_in_pot',
    # 'put_bowl_on_plate_cardboard_fence': 'put_bowl_on_plate',
    # 'put_carrot_in_pot_cardboard_fence': 'put_carrot_in_pot',
    # 'put_knife_in_pot_cardboard_fence': 'put_knife_in_pot',
    # 'put_knife_on_cutting_board_cardboard_fence': 'put_knife_on_cutting_board',
    # 'put_lid_on_pot_cardboardfence': 'put_lid_on_pot',
    # 'put_pear_in_bowl_cardboardfence': 'put_pear_in_bowl',
    # 'put_potato_in_pot_cardboard_fence': 'put_potato_in_pot',
    # 'put_sushi_in_pot_cardboard_fence': 'put_sushi_in_pot',
    # 'take_bowl_off_plate_cardboard_fence': 'take_bowl_off_plate',
    # 'take_carrot_out_of_pot_cardboard_fence': 'take_carrot_out_of_pot',
    # 'take_lid_off_pot_cardboardfence': 'take_lid_off_pot',
    # 'take_sushi_out_of_pot_cardboard_fence': 'take_sushi_out_of_pot',
    # 'topple_basil_bottle_cardboard_fence': 'topple_basil_bottle',
    # 'topple_hot_sauce_bottle_cardboard_fence': 'topple_hot_sauce_bottle',
    # 'topple_metal_pot_cardboard_fence': 'topple_metal_pot',
    # 'upright_basil_bottle_cardboard_fence': 'upright_basil_bottle',
    # 'upright_hot_sauce_bottle_cardboard_fence': 'upright_hot_sauce_bottle',
    # 'upright_metal_pot_cardboard_fence': 'upright_metal_pot',

    # 'put_beet_in_pot_sink': 'put_beet_in_pot_in_sink',
    # 'put_blueberries_on_plate_sink': 'put_blueberries_on_plate_in_sink',
    # 'put_corn_in_bowl_sink': 'put_corn_in_bowl_in_sink',
    # 'put_spatula_on_plate_sink': 'put_spatula_on_plate_in_sink',
    # 'put_spoon_in_bowl_sink': 'put_spoon_in_bowl_in_sink',
    # 'take_beet_from_pot_sink': 'take_beet_from_pot_in_sink',
    # 'take_blueberries_off_plate_sink': 'take_blueberries_off_plate_in_sink',
    # 'take_corn_out_of_bowl_sink': 'take_corn_out_of_bowl_in_sink',
    # 'take_spatula_off_plate_sink': 'take_spatula_off_plate_in_sink',
    # 'take_spoon_out_of_bowl_sink': 'take_spoon_out_of_bowl_in_sink',
}

# data from online reaching
online_reaching_pixels_first100 = ['/robonetv2/online_datacollection/extract/online_reaching_first100/toykitchen1/pix_policy_std0.3_initscale0.0_collectdata/train/out.npy']
online_reaching_pixels_val_first100 = ['/robonetv2/online_datacollection/extract/online_reaching_first100/toykitchen1/pix_policy_std0.3_initscale0.0_collectdata/val/out.npy']
online_reaching_pixels= ['/robonetv2/online_datacollection/extract/online_reaching/toykitchen1/pix_policy_std0.3_initscale0.0_collectdata/train/out.npy']
online_reaching_pixels_val= ['/robonetv2/online_datacollection/extract/online_reaching/toykitchen1/pix_policy_std0.3_initscale0.0_collectdata/val/out.npy']

