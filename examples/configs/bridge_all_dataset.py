from typing import Iterator, List, Union, Optional, Iterable
import fnmatch

import os
from absl import logging
import tensorflow as tf

def glob_to_path_list(
    glob_strs: Union[str, List[str]], prefix: str = '', include: Iterable[str]=('',), exclude: Iterable[str] = ()
):
    """Converts a glob string or list of glob strings to a list of paths."""
    if isinstance(glob_strs, str):
        glob_strs = [glob_strs]
    path_list = []
    for glob_str in glob_strs:
        full_path = f"{prefix}/{glob_str}" if prefix else glob_str
        paths = tf.io.gfile.glob(full_path)
        filtered_paths = []
        for path in paths:
            if not any(i in path for i in include):
                continue
            # if not any(fnmatch.fnmatch(path, e) for e in exclude):
            if not any(e in path for e in exclude):
                filtered_paths.append(path)
            else:
                logging.info(f"Excluding {path}")
        #assert len(filtered_paths) > 0, f"{glob_str} came up empty"
        path_list += filtered_paths
    return path_list

def _get_tasks(task_patterns, include=('',), exclude=(), small=False):
    task_paths = glob_to_path_list(task_patterns, prefix=os.environ['DATA'], include=include, exclude=exclude)
    for task_path in task_paths:
        task_name = str.split(task_path, '/')[-2:]
        print(task_name)
    print(len(task_paths))

    train_globs, val_globs = [], []
    for task_path in task_paths:
        if 'bridge_data_v2' in task_path and not small:
            train_globs.append(f'{task_path}/*/train/out.npy')
            val_globs.append(f'{task_path}/*/val/out.npy')
        else:
            train_globs.append(f'{task_path}/train/out.npy')
            val_globs.append(f'{task_path}/val/out.npy')

    train_paths, val_paths = glob_to_path_list(train_globs), glob_to_path_list(val_globs)
    return train_paths, val_paths

def empty():
    return [], []

def bridge_v2_test():
    task_patterns = ['bridge_data_v2/datacol1_toykitchen6/*']
    return _get_tasks(task_patterns)

def bridge_v2_test_small():
    task_patterns = ['bridge_data_v2/datacol1_toykitchen6/pnp_sweep/00']
    return _get_tasks(task_patterns, small=True)

def bridge_v1_test_small():
    task_patterns = ['bridge_data_v1/berkeley/toykitchen1/close_microwave']
    return _get_tasks(task_patterns)

def bridge_v2_hard():
    task_patterns = ['bridge_data_v2/*/*stack*', 'bridge_data_v2/*/*fold*', 'bridge_data_v2/*/*sweep*']
    return _get_tasks(task_patterns)

def bridge_v2_blocks():
    task_patterns = ['bridge_data_v2/*/*stack*']
    return _get_tasks(task_patterns)
    
def get_all_pickplace():
    task_patterns =  ['bridge_data_v1/berkeley/*/*', 'bridge_data_v2/*/*']
    exclude_strs = [
        'zip', 'test', 'box', 'fold_cloth_in_half', 'put_cap_on_container', 'open_book', 'basil_bottle',
        'test', 'box', 'basket', 'knob', 'open', 'close', 'flip', 'lever', 'topple', 'pour', 'drying_rack'
    ]
    return _get_tasks(task_patterns, exclude=exclude_strs)

def get_all_pickplace_exclude_tk6():
    task_patterns =  ['bridge_data_v1/berkeley/*/*', 'bridge_data_v2/*/*']
    exclude_strs = ['zip', 'test', 'box', 'fold_cloth_in_half', 'put_cap_on_container', 'open_book', 'basil_bottle',
                    'test', 'box', 'basket', 'knob', 'open', 'close', 'flip', 'lever', 'topple', 'pour', 'drying_rack',
                    'tool_chest', 'laundry_machine', 'put_brush_into_pot_or_pan', 'put_cup_into_pot_or_pan']
    exclude_strs += ['toykitchen6']

    return _get_tasks(task_patterns, exclude=exclude_strs)


def get_all_pickplace_v1_exclude_tk6():
    task_patterns =  ['bridge_data_v1/berkeley/*/*']
    exclude_strs = ['zip', 'test', 'box', 'fold_cloth_in_half', 'put_cap_on_container', 'open_book', 'basil_bottle',
                    'test', 'box', 'basket', 'knob', 'open', 'close', 'flip', 'lever', 'topple', 'pour', 'drying_rack',
                    'tool_chest', 'laundry_machine', 'put_brush_into_pot_or_pan', 'put_cup_into_pot_or_pan',
                    'put_knife_in_pot_or_pan', 'turn_faucet_right', 'turn_faucet_front_to_right', 'put_cup_from_anywhere_into_sink', 'put_spoon_in_pot',
                     'lmdb', 'colander', 'put_spatula_on_cutting_board']
    exclude_strs += ['toykitchen6']

    return _get_tasks(task_patterns, exclude=exclude_strs)

def get_all_openclose_v1_exclude_tk1():
    task_patterns =  ['bridge_data_v1/berkeley/*/*']

    include_strs = ['open', 'close']
    exclude_strs = ['book', 'pick_up_closest_rainbow_Allen_key_set', 'box', 'toykitchen1']

    return _get_tasks(task_patterns, include=include_strs, exclude=exclude_strs)

def get_all_openclose_v1_exclude_tk6():
    task_patterns =  ['bridge_data_v1/berkeley/*/*']

    include_strs = ['open', 'close']
    exclude_strs = ['book', 'pick_up_closest_rainbow_Allen_key_set', 'box', 'toykitchen6']

    return _get_tasks(task_patterns, include=include_strs, exclude=exclude_strs)

# collected on 2023.05.15
def tk6_targetdomain_cucumberpot_0515():
    task_patterns = ['toykitchen_fixed_cam_numpy/*/put_cucumber_in_orange_pot_0515']
    return _get_tasks(task_patterns)
    
# collected on 2023.05.15
def tk6_targetdomain_cucumberpot_distractors_0515():
    task_patterns = ['toykitchen_fixed_cam_numpy/*/put_cucumber_in_orange_pot_distractors_0515']
    return _get_tasks(task_patterns)

# collected on 2023.05.15
def tk6_targetdomain_cucumberpot_combined_0515():
    task_patterns = [
        'toykitchen_fixed_cam_numpy/*/put_cucumber_in_orange_pot_0515',
        'toykitchen_fixed_cam_numpy/*/put_cucumber_in_orange_pot_distractors_0515'
    ]
    return _get_tasks(task_patterns)

# collected on 2023.05.15
def tk6_targetdomain_knifepan_0515():
    task_patterns = ['toykitchen_fixed_cam_numpy/*/put_knife_in_orange_pan_0515']
    return _get_tasks(task_patterns)

# collected on 2023.05.15
def tk6_targetdomain_knifepan_distractors_0515():
    task_patterns = ['toykitchen_fixed_cam_numpy/*/put_knife_in_orange_pan_distractors_0515']
    return _get_tasks(task_patterns)

# collected on 2023.05.15
def tk6_targetdomain_knifepan_combined_0515():
    task_patterns = [
        'toykitchen_fixed_cam_numpy/*/put_knife_in_orange_pan_0515',
        'toykitchen_fixed_cam_numpy/*/put_knife_in_orange_pan_distractors_0515'
    ]
    return _get_tasks(task_patterns)

# collected on 2023.05.15
def tk6_targetdomain_sweetpotatoplate_0515():
    task_patterns = ['toykitchen_fixed_cam_numpy/*/put_sweet_potato_on_plate_0515']
    return _get_tasks(task_patterns)

# collected on 2023.05.15
def tk6_targetdomain_sweetpotatoplate_distractors_0515():
    task_patterns = ['toykitchen_fixed_cam_numpy/*/put_sweet_potato_on_plate_distractors_0515']
    return _get_tasks(task_patterns)

# collected on 2023.05.15
def tk6_targetdomain_sweetpotatoplate_combined_0515():
    task_patterns = [
        'toykitchen_fixed_cam_numpy/*/put_sweet_potato_on_plate_0515',
        'toykitchen_fixed_cam_numpy/*/put_sweet_potato_on_plate_distractors_0515'
    ]
    return _get_tasks(task_patterns)

# collected on 2023.05.15
def tk6_targetdomain_croissantcolander_0515():
    task_patterns = ['toykitchen_fixed_cam_numpy/*/take_croissant_out_of_colander_0515']
    return _get_tasks(task_patterns)

# collected on 2023.05.15
def tk6_targetdomain_croissantcolander_distractors_0515():
    task_patterns = ['toykitchen_fixed_cam_numpy/*/take_croissant_out_of_colander_distractors_0515']
    return _get_tasks(task_patterns)

# collected on 2023.05.30
def tk6_targetdomain_croissantcolander_0530():
    task_patterns = ['toykitchen_fixed_cam_numpy/*/take_croissant_out_of_colander_0530']
    return _get_tasks(task_patterns)

def tk6_targetdomain_croissantcolander_combined_0515():
    task_patterns = [
        'toykitchen_fixed_cam_numpy/*/take_croissant_out_of_colander_0515',
        'toykitchen_fixed_cam_numpy/*/take_croissant_out_of_colander_distractors_0515'
    ]
    return _get_tasks(task_patterns)

def tk6_targetdomain_croissantcolander_all():
    task_patterns = [
        'toykitchen_fixed_cam_numpy/*/take_croissant_out_of_colander_0515',
        'toykitchen_fixed_cam_numpy/*/take_croissant_out_of_colander_0530',
        'toykitchen_fixed_cam_numpy/*/take_croissant_out_of_colander_distractors_0515',
    ]
    return _get_tasks(task_patterns)

def tk6_targetdomain_colander_all():
    task_patterns = [
        'toykitchen_fixed_cam_numpy/*/take_banana_out_of_colander_0530',
        'toykitchen_fixed_cam_numpy/*/take_blueberries_out_of_colander_0530',
        'toykitchen_fixed_cam_numpy/*/take_carrot_out_of_colander_0530',
        'toykitchen_fixed_cam_numpy/*/take_challah_out_of_colander_0530',
        'toykitchen_fixed_cam_numpy/*/take_croissant_out_of_colander_0515',
        'toykitchen_fixed_cam_numpy/*/take_croissant_out_of_colander_0530',
        'toykitchen_fixed_cam_numpy/*/take_croissant_out_of_colander_distractors_0515',
        'toykitchen_fixed_cam_numpy/*/take_greengrapes_out_of_colander_0530'
    ]
    return _get_tasks(task_patterns)

# collected on 2023.05.23
def tk1_targetdomain_openmicro_0523():
    task_patterns = ['toykitchen_fixed_cam_numpy/*/open_microwave_0523']
    return _get_tasks(task_patterns)

# collected on 2023.05.23
def tk1_targetdomain_openmicro_distractors_0523(num_demo=None):
    task_patterns = ['toykitchen_fixed_cam_numpy/*/open_microwave_distractors_0523']
    return _get_tasks(task_patterns)

# collected on 2023.05.23
def tk1_targetdomain_openmicro_combined_0523(num_demo=None):
    task_patterns = [
        'toykitchen_fixed_cam_numpy/*/open_microwave_0523',
        'toykitchen_fixed_cam_numpy/*/open_microwave_distractors_0523'
    ]
    return _get_tasks(task_patterns)

# collected on 2023.05.23
def tk1_targetdomain_closemicro_0523(num_demo=None):
    task_patterns = ['toykitchen_fixed_cam_numpy/*/close_microwave_0523']
    return _get_tasks(task_patterns)

# collected on 2023.05.23
def tk1_targetdomain_closemicro_distractors_0523(num_demo=None):
    task_patterns = ['toykitchen_fixed_cam_numpy/*/close_microwave_distractors_0523']
    return _get_tasks(task_patterns)

# collected on 2023.05.23
def tk1_targetdomain_closemicro_combined_0523(num_demo=None):
    task_patterns = [
        'toykitchen_fixed_cam_numpy/*/close_microwave_0523',
        'toykitchen_fixed_cam_numpy/*/close_microwave_distractors_0523'
    ]
    return _get_tasks(task_patterns)

def tk6_targetdomain_anycolander_0515():
    task_patterns = [
        'toykitchen_fixed_cam_numpy/*/take_banana_out_of_colander_0530',
        'toykitchen_fixed_cam_numpy/*/take_blueberries_out_of_colander_0530',
        'toykitchen_fixed_cam_numpy/*/take_carrot_out_of_colander_0530',
        'toykitchen_fixed_cam_numpy/*/take_challah_out_of_colander_0530',
        'toykitchen_fixed_cam_numpy/*/take_croissant_out_of_colander_0530',
        'toykitchen_fixed_cam_numpy/*/take_greengrapes_out_of_colander_0530'
    ]
    return _get_tasks(task_patterns)

def tk6_targetdomain_cloth_0803():
    task_patterns = [
        'toykitchen_fixed_cam_numpy/toykitchen6/cloth_0803',
    ]
    return _get_tasks(task_patterns)

def tk6_targetdomain_sweep_0805():
    task_patterns = [
        'toykitchen_fixed_cam_numpy/toykitchen6/sweep_into_pile_0805',
    ]
    return _get_tasks(task_patterns)

def tk6_targetdomain_opendrawer_0805():
    task_patterns = [
        'toykitchen_fixed_cam_numpy/toykitchen6/open_the_drawer_0805',
    ]
    return _get_tasks(task_patterns)

def all_tk6():
    task_patterns = [
        'bridge_data_v2/datacol1_toykitchen6/*',
        'bridge_data_v1/berkeley/toykitchen6/*',
    ]
    return _get_tasks(task_patterns)

dataset_fns = {
    'all_pickplace': get_all_pickplace,
    'all_pickplace_except_tk6': get_all_pickplace_exclude_tk6,
    'all_pickplace_v1_except_tk6': get_all_pickplace_v1_exclude_tk6,
    'all_openclose_v1_except_tk1': get_all_openclose_v1_exclude_tk1,
    'all_openclose_v1_exclude_tk6': get_all_openclose_v1_exclude_tk6,
    'bridge_v1_test_small': bridge_v1_test_small,
    'bridge_v2_test': bridge_v2_test,
    'bridge_v2_test_small': bridge_v2_test_small,
    'bridge_v2_hard': bridge_v2_hard,
    'bridge_v2_blocks': bridge_v2_blocks,
    'empty': empty,
}

target_dataset_fns = {
    'tk6_targetdomain_cucumberpot_0515': tk6_targetdomain_cucumberpot_0515, # this is collected on 2023/05/15
    'tk6_targetdomain_cucumberpot_distractors_0515': tk6_targetdomain_cucumberpot_distractors_0515, # this is collected on 2023/05/15
    'tk6_targetdomain_cucumberpot_combined_0515': tk6_targetdomain_cucumberpot_combined_0515, # this is collected on 2023/05/15
    'tk6_targetdomain_knifepan_0515': tk6_targetdomain_knifepan_0515, # this is collected on 2023/05/15
    'tk6_targetdomain_knifepan_distractors_0515': tk6_targetdomain_knifepan_distractors_0515, # this is collected on 2023/05/15
    'tk6_targetdomain_knifepan_combined_0515': tk6_targetdomain_knifepan_combined_0515, # this is collected on 2023/05/15
    'tk6_targetdomain_sweetpotatoplate_0515': tk6_targetdomain_sweetpotatoplate_0515, # this is collected on 2023/05/15
    'tk6_targetdomain_sweetpotatoplate_distractors_0515': tk6_targetdomain_sweetpotatoplate_distractors_0515, # this is collected on 2023/05/15
    'tk6_targetdomain_sweetpotatoplate_combined_0515': tk6_targetdomain_sweetpotatoplate_combined_0515, # this is collected on 2023/05/15
    'tk6_targetdomain_croissantcolander_0515': tk6_targetdomain_croissantcolander_0515, # this is collected on 2023/05/15
    'tk6_targetdomain_croissantcolander_distractors_0515': tk6_targetdomain_croissantcolander_distractors_0515, # this is collected on 2023/05/15
    'tk6_targetdomain_croissantcolander_combined_0515':tk6_targetdomain_croissantcolander_combined_0515, # this is collected on 2023/05/15
    'tk6_targetdomain_anycolander_0515':tk6_targetdomain_anycolander_0515, # this is collected on 2023/05/15
    'tk6_targetdomain_croissantcolander_0530':tk6_targetdomain_croissantcolander_0530,
    'tk6_targetdomain_anycolander_0515':tk6_targetdomain_anycolander_0515,
    'tk6_targetdomain_croissantcolander_all':tk6_targetdomain_croissantcolander_all,
    'tk6_targetdomain_colander_all':tk6_targetdomain_colander_all,
    'tk6_targetdomain_cloth_0803':tk6_targetdomain_cloth_0803,
    'tk6_targetdomain_sweep_0805':tk6_targetdomain_sweep_0805,
    'tk6_targetdomain_opendrawer_0805':tk6_targetdomain_opendrawer_0805,
    'test':bridge_v2_test_small,
    'test_v1':bridge_v1_test_small,
    'all_tk6':all_tk6,
    'tk1_targetdomain_openmicro_0523':tk1_targetdomain_openmicro_0523,
    'tk1_targetdomain_openmicro_distractors_0523':tk1_targetdomain_openmicro_distractors_0523,
    'tk1_targetdomain_openmicro_combined_0523':tk1_targetdomain_openmicro_combined_0523,
    'tk1_targetdomain_closemicro_0523':tk1_targetdomain_closemicro_0523,
    'tk1_targetdomain_closemicro_distractors_0523':tk1_targetdomain_closemicro_distractors_0523,
    'tk1_targetdomain_closemicro_combined_0523':tk1_targetdomain_closemicro_combined_0523,
}

if __name__ == '__main__':
    get_all_pickplace_exclude_tk6()