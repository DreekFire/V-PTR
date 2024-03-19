import pdb

import numpy as np
import os
from widowx_envs.widowx.widowx_env import WidowXEnv
from gym.spaces import Dict
from gym.spaces import Box
import time
from vptr.utils.visualization_utils import sigmoid
import rospy
import random
import pickle as pkl

from widowx_envs.utils.exceptions import Environment_Exception


from examples.train_pixels_real import TARGET_POINT

import glob

traj_group_open = glob.glob(os.environ['DATA'] + '/robonetv2/toykitchen_fixed_cam_microwave_open_close/microwave/open_microwave/*/raw/traj_group*/traj*')
# traj_group_open = glob.glob(os.environ['DATA'] + '/robonetv2/tk1_microwave_targetdemos/toykitchen1/open_1109/*/raw/traj_group*/traj*')


traj_group_close = glob.glob(os.environ['DATA'] + '/robonetv2/toykitchen_fixed_cam_microwave_open_close/microwave/close_microwave/*/raw/traj_group0/traj0')
# traj_group_close = glob.glob(os.environ['DATA'] + '/robonetv2/tk1_microwave_targetdemos/toykitchen1/close/*/raw/traj_group*/traj*')

# traj_group_croissant_pot = glob.glob(os.environ['DATA'] + '/robonetv2/toykitchen_fixed_cam/toykitchen6/take_croissant_out_of_pot/*/raw/traj_group*/traj*')


traj_group_sushi = glob.glob(os.environ['DATA'] + '/robonetv2/toykitchen_fixed_cam/toykitchen2/put_sushi_in_pot_cardboard_fence/*/raw/traj_group*/traj*')
traj_group_sushi_out = glob.glob(os.environ['DATA'] + '/robonetv2/toykitchen_fixed_cam/toykitchen2/take_sushi_out_of_pot_cardboard_fence/*/raw/traj_group*/traj*')
traj_group_bowlplate = glob.glob(os.environ['DATA'] + '/robonetv2/toykitchen_fixed_cam/toykitchen2/put_bowl_on_plate_cardboard_fence/*/raw/traj_group*/traj*')
traj_group_cornbowl = glob.glob(os.environ['DATA'] + '/robonetv2/toykitchen_fixed_cam/toykitchen6/put_corn_in_bowl_sink/*/raw/traj_group*/traj*')
traj_group_cup_plate = glob.glob(os.environ['DATA'] + '/robonetv2/toykitchen_fixed_cam/toykitchen6/put_cup_on_plate/*/raw/traj_group*/traj*')
traj_group_cup_off_plate = glob.glob(os.environ['DATA'] + '/robonetv2/toykitchen_fixed_cam/toykitchen6/take_cup_off_plate/*/raw/traj_group*/traj*')
traj_group_corn_out_bowl = glob.glob(os.environ['DATA'] + '/robonetv2/toykitchen_fixed_cam/toykitchen6/take_corn_out_of_bowl_sink/*/raw/traj_group*/traj*')
traj_group_knife_pot = glob.glob(os.environ['DATA'] + '/robonetv2/toykitchen_fixed_cam_targetdomain/berkeley/toykitchen6/put_knife_into_pot/2022-06-15_13-58-35/raw/traj_group*/traj*')
traj_group_croissant_pot = glob.glob(os.environ['DATA'] + '/robonetv2/toykitchen_fixed_cam/toykitchen6/take_croissant_out_of_colander_0515/2023-05-19_17-14-17/raw/traj_group*/traj*')
traj_group_sweet_potato_plate = glob.glob(os.environ['DATA'] + '/robonetv2/toykitchen_fixed_cam/toykitchen6/put_sweet_potato_on_plate_0515/2023-05-19_18-05-27/raw/traj_group*/traj*')

traj_group_tk6_task1 = glob.glob(os.environ['DATA'] + '/robonetv2/toykitchen_fixed_cam_targetdomain/berkeley/toykitchen6/put_knife_into_pot/*/raw/traj_group*/traj*')
traj_group_tk6_task2 = glob.glob(os.environ['DATA'] + '/robonetv2/toykitchen_fixed_cam_targetdomain/berkeley/toykitchen6/take_croissant_out_of_pot/*/raw/traj_group*/traj*')
traj_group_tk6_task3 = glob.glob(os.environ['DATA'] + '/robonetv2/toykitchen_fixed_cam_targetdomain/berkeley/toykitchen6/take_pear_from_plate/*/raw/traj_group*/traj*')
traj_group_tk6_task4 = glob.glob(os.environ['DATA'] + '/robonetv2/toykitchen_fixed_cam_targetdomain/berkeley/toykitchen6/put_sweet_potato_in_bowl/*/raw/traj_group*/traj*')
traj_group_tk6_task5 = glob.glob(os.environ['DATA'] + '/robonetv2/toykitchen_fixed_cam_targetdomain/berkeley/toykitchen6/put_lime_in_pan_sink/*/raw/traj_group*/traj*')
traj_group_tk6_task6 = glob.glob(os.environ['DATA'] + '/robonetv2/toykitchen_fixed_cam_targetdomain/berkeley/toykitchen6/put_sweet_potato_on_plate/*/raw/traj_group*/traj*')
traj_group_tk6_task7 = glob.glob(os.environ['DATA'] + '/robonetv2/toykitchen_fixed_cam_targetdomain/berkeley/toykitchen6/put_drumstick_on_plate/*/raw/traj_group*/traj*')
traj_group_tk6_task8 = glob.glob(os.environ['DATA'] + '/robonetv2/toykitchen_fixed_cam_targetdomain/berkeley/toykitchen6/put_cucumber_in_orange_pot/*/raw/traj_group*/traj*')
traj_group_tk6_task9 = glob.glob(os.environ['DATA'] + '/robonetv2/toykitchen_fixed_cam_targetdomain/berkeley/toykitchen6/put_carrot_in_pan/*/raw/traj_group*/traj*')
traj_group_tk6_task10 = glob.glob(os.environ['DATA'] + '/robonetv2/toykitchen_fixed_cam_targetdomain/berkeley/toykitchen6/put_big_corn_in_big_pot/*/raw/traj_group*/traj*')

traj_group_anikait_tk6_task1 = glob.glob(os.environ['DATA'] + '/robonetv2/toykitchen_fixed_cam_targetdom_anikait/toykitchen6/croissant_in_pan/*/raw/traj_group*/traj*')
traj_group_anikait_tk6_task2 = glob.glob(os.environ['DATA'] + '/robonetv2/toykitchen_fixed_cam_targetdom_anikait/toykitchen6/cucumber_in_pot/*/raw/traj_group*/traj*')
traj_group_anikait_tk6_task3 = glob.glob(os.environ['DATA'] + '/robonetv2/toykitchen_fixed_cam_targetdom_anikait/toykitchen6/knife_in_pan/*/raw/traj_group*/traj*')
traj_group_anikait_tk6_task4 = glob.glob(os.environ['DATA'] + '/robonetv2/toykitchen_fixed_cam_targetdom_anikait/toykitchen6/sushi_in_pot/*/raw/traj_group*/traj*')

traj_group_tk6_put_cucumber_pot_elevated = glob.glob(os.environ['DATA'] + '/robonetv2/toykitchen_fixed_cam_targetdomain/berkeley/toykitchen6_elevated/put_cucumber_in_orange_pot/*/raw/traj_group*/traj*')
traj_group_tk6_put_cucumber_pot_elevated_rotated = glob.glob(os.environ['DATA'] + '/robonetv2/toykitchen_fixed_cam_targetdomain/berkeley/toykitchen6_rotated_elevated/put_cucumber_in_orange_pot/*/raw/traj_group*/traj*')
traj_group_tk6_take_croissant_elevated = glob.glob(os.environ['DATA'] + '/robonetv2/toykitchen_fixed_cam_targetdomain/berkeley/toykitchen6_elevated/take_croissant_out_of_pot//*/raw/traj_group*/traj*')
traj_group_tk6_take_croissant_rotated = glob.glob(os.environ['DATA'] + '/robonetv2/toykitchen_fixed_cam_targetdomain/berkeley/toykitchen6_rotated/take_croissant_out_of_pot//*/raw/traj_group*/traj*')

start_transforms = dict(
    right_rear = [os.environ['DATA'] + '/robonetv2/toykitchen_fixed_cam_targetdomain/berkeley/toykitchen6/take_croissant_out_of_pot/2022-06-15_13-38-29/raw/traj_group0/traj0', 0],
    right_front = [os.environ['DATA'] + '/robonetv2/toykitchen_fixed_cam_targetdomain/berkeley/toykitchen6/take_croissant_out_of_pot/2022-06-15_13-38-29/raw/traj_group0/traj12', 0],
    middle = [os.environ['DATA'] + '/robonetv2/toykitchen_fixed_cam_targetdomain/berkeley/toykitchen6/take_croissant_out_of_pot/2022-06-15_13-38-29/raw/traj_group0/traj10', 0],
    left_front = [os.environ['DATA'] + '/robonetv2/toykitchen_fixed_cam_targetdomain/berkeley/toykitchen6/take_croissant_out_of_pot/2022-06-14_22-57-10/raw/traj_group0/traj12', 0],
    left_rear = [os.environ['DATA'] + '/robonetv2/toykitchen_fixed_cam/toykitchen6/take_croissant_out_of_colander_0515/2023-05-19_17-14-17/raw/traj_group0/traj1', 26],
    # left_rear = [os.environ['DATA'] + '/robonetv2/toykitchen_fixed_cam_targetdomain/berkeley/toykitchen6/take_croissant_out_of_pot/2022-06-14_22-57-10/raw/traj_group0/traj12', 34],
    
    # right_front = [os.environ['DATA'] + '/robonetv2/toykitchen_fixed_cam_initial_testconfig/rss_benchmark/toykitchen1/put_broccoli_in_pot/2022-01-22_11-06-56/raw/traj_group0/traj0', 150],
    # middle = [os.environ['DATA'] + '/robonetv2/toykitchen_fixed_cam_initial_testconfig/rss_benchmark/toykitchen1/put_broccoli_in_pot/2022-01-22_11-06-56/raw/traj_group0/traj1', 290],
    # left_front = [os.environ['DATA'] + '/robonetv2/toykitchen_fixed_cam_initial_testconfig/rss_benchmark/toykitchen1/put_broccoli_in_pot/2022-01-22_11-06-56/raw/traj_group0/traj2', 200],
    # left_rear = [os.environ['DATA'] + '/robonetv2/toykitchen_fixed_cam_initial_testconfig/rss_benchmark/toykitchen1/put_broccoli_in_pot/2022-01-22_17-54-21/raw/traj_group0/traj0', 290],
    # right_rear = [os.environ['DATA'] + '/robonetv2/toykitchen_fixed_cam_initial_testconfig/rss_benchmark/toykitchen1/put_broccoli_in_pot/2022-01-22_17-54-21/raw/traj_group0/traj1', 290],

    openmicrowave = [os.environ['DATA'] + '/robonetv2/toykitchen_fixed_cam/toykitchen1/open_microwave/2021-12-01_16-14-39/raw/traj_group0/traj0', 7],
    
    # 7, 9, 8, 10, 2, 5

    openmicrowave1 = [os.environ['DATA'] + '/robonetv2/toykitchen_fixed_cam/toykitchen1/open_microwave/2021-12-01_16-14-39/raw/traj_group0/traj0', 4],
    openmicrowave2 = [os.environ['DATA'] + '/robonetv2/toykitchen_fixed_cam/toykitchen1/open_microwave/2021-12-01_16-14-39/raw/traj_group0/traj1', 1],
    openmicrowave3 = [os.environ['DATA'] + '/robonetv2/toykitchen_fixed_cam/toykitchen1/open_microwave/2021-12-01_16-14-39/raw/traj_group0/traj2', 2],
    openmicrowave4 = [os.environ['DATA'] + '/robonetv2/toykitchen_fixed_cam/toykitchen1/open_microwave/2021-12-01_16-14-39/raw/traj_group0traj23', 3],
    openmicrowave5 = [os.environ['DATA'] + '/robonetv2/toykitchen_fixed_cam/toykitchen1/open_microwave/2021-12-01_16-14-39/raw/traj_group0/traj4', 5],
    openmicrowave6 = [os.environ['DATA'] + '/robonetv2/toykitchen_fixed_cam/toykitchen1/open_microwave/2021-12-01_16-14-39/raw/traj_group0/traj5', 7],
    openmicrowave7 = [os.environ['DATA'] + '/robonetv2/toykitchen_fixed_cam/toykitchen1/open_microwave/2021-12-01_16-14-39/raw/traj_group0/traj6', 1],
    openmicrowave8 = [os.environ['DATA'] + '/robonetv2/toykitchen_fixed_cam/toykitchen1/open_microwave/2021-12-01_16-14-39/raw/traj_group0/traj7', 6],
    openmicrowave9 = [os.environ['DATA'] + '/robonetv2/toykitchen_fixed_cam/toykitchen1/open_microwave/2021-12-01_16-14-39/raw/traj_group0/traj8', 5],
    openmicrowave10 = [os.environ['DATA'] + '/robonetv2/toykitchen_fixed_cam/toykitchen1/open_microwave/2021-12-01_16-14-39/raw/traj_group0/traj9', 5],

    openmicrowave11 = [os.environ['DATA'] + '/robonetv2/toykitchen_fixed_cam/toykitchen1/open_microwave/2021-12-01_16-14-39/raw/traj_group0/traj6', 6],

    openmicrowave12 = [os.environ['DATA'] + '/robonetv2/toykitchen_fixed_cam/toykitchen1/open_microwave/2021-12-01_16-14-39/raw/traj_group0/traj0', 6],
    
    openmicrowave13 = [os.environ['DATA'] + '/robonetv2/toykitchen_fixed_cam/toykitchen1/open_microwave/2021-12-01_16-14-39/raw/traj_group0/traj1', 6],
    
    
    closemicrowave = [os.environ['DATA'] + '/robonetv2/toykitchen_fixed_cam/toykitchen1/close_microwave/2021-12-02_12-14-59/raw/traj_group0/traj1', 0],
    openmicrowave_sampled = [traj_group_open, 6],
    closemicrowave_sampled = [traj_group_close,  (10,16)],
    # closemicrowave_sampled = [os.environ['DATA'] + '/robonetv2/toykitchen_fixed_cam_microwave_open_close/microwave/close_microwave/2022-09-09_15-18-34/raw/traj_group0/traj0', (10, 16)],
    # openclosemicrowave_sampled = {3: [traj_group_open, 10], 1:[traj_group_close, (10, 16)]},  # use without aliasing
    openclosemicrowave_sampled = {1: [traj_group_open, 4], 0:[traj_group_close, (10,16)]},  # with aliasing
    # openclosemicrowave_sampled = {1: [traj_group_open, 7], 0: [os.environ['DATA'] + '/robonetv2/toykitchen_fixed_cam/toykitchen1/close_microwave/2021-12-02_12-14-59/raw/traj_group0/traj1', 0]},  # with aliasing

    
    reaching = [os.environ['DATA'] + '/robonetv2/online_datacollection/online_reaching/berkeley/toykitchen1/pix_policy_std0.3_initscale0.0_collectdata/2022-04-25_18-10-03/raw/traj_group0/traj0', 0],
    toykitchen2_sushi_cardboard_sampled = [traj_group_sushi, (5, 10)],
    # toykitchen2_sushi_cardboard_sampled = [traj_group_sushi, (0, 5)],
    toykitchen2_sushi_out_cardboard_sampled = [traj_group_sushi_out, (0, 5)],
    toykitchen2_bowlplate_cardboard_sampled= [traj_group_bowlplate, (0, 5)],
    toykitchen6_cornbowl_cardboard_sampled= [traj_group_cornbowl, (5, 10)],
    # toykitchen6_cornbowl_cardboard_sampled= [traj_group_cornbowl, (0, 5)],
    toykitchen6_cupplate_cardboard_sampled= [traj_group_cup_plate, (0, 5)],
    toykitchen6_cup_off_plate_cardboard_sampled= [traj_group_cup_off_plate, (8, 13)],
    # toykitchen6_corn_out_bowl_cardboard_sampled= [traj_group_corn_out_bowl, (0, 3)],
    toykitchen6_corn_out_bowl_cardboard_sampled= [traj_group_corn_out_bowl, (3, 6)],
    toykitchen6_knife_pot_cardboard_sampled= [traj_group_knife_pot, (0, 5)],
    toykitchen6_croissant_pot_sampled= [traj_group_croissant_pot, (0, 5)],
    toykitchen6_sweet_potato_plate_sampled = [traj_group_sweet_potato_plate, (0, 5)],
    
    anikait_croissant=[traj_group_anikait_tk6_task1,(0,5)],
    anikait_cucumber=[traj_group_anikait_tk6_task2,(0,5)],
    anikait_knife=[traj_group_anikait_tk6_task3, (0, 5)],
    anikait_sushi=[traj_group_anikait_tk6_task4, (0, 5)],

    # tk6_task1=[traj_group_tk6_task1, (0,5)],
    # tk6_task2=[traj_group_tk6_task2, (0,5)],
    # tk6_task3=[traj_group_tk6_task3, (0,5)],
    # tk6_task4=[traj_group_tk6_task4, (0,5)],
    # tk6_task5=[traj_group_tk6_task5, (0,5)],
    # tk6_task6=[traj_group_tk6_task6, (0,5)],
    # tk6_task7=[traj_group_tk6_task7, (0,5)],
    # tk6_task8=[traj_group_tk6_task8, (0,5)],
    # tk6_task9=[traj_group_tk6_task9, (0,5)],
    # tk6_task10=[traj_group_tk6_task10, (0,5)],

    tk6_task1=[traj_group_tk6_task1, (5,10)],
    tk6_task2=[traj_group_tk6_task2, (5,10)],
    tk6_task3=[traj_group_tk6_task3, (5,10)],
    tk6_task4=[traj_group_tk6_task4, (5,10)],
    tk6_task5=[traj_group_tk6_task5, (5,10)],
    tk6_task6=[traj_group_tk6_task6, (5,10)],
    tk6_task7=[traj_group_tk6_task7, (5,10)],
    tk6_task8=[traj_group_tk6_task8, (5,10)],
    tk6_task9=[traj_group_tk6_task9, (5,10)],
    tk6_task10=[traj_group_tk6_task10, (5,10)],
    tk6_put_cucumber_pot_elevated=[traj_group_tk6_put_cucumber_pot_elevated, (5,10)],
    tk6_put_cucumber_pot_elevated_rotated=[traj_group_tk6_put_cucumber_pot_elevated_rotated, (5,10)],
    tk6_take_croissant_elevated=[traj_group_tk6_take_croissant_elevated, (5,10)],
    tk6_take_croissant_rotated=[traj_group_tk6_take_croissant_rotated, (5,10)],
)

def get_env_params(variant):
    env_params = {
        'fix_zangle': True,  # do not apply random rotations to start state
        'move_duration': 0.2,
        'adaptive_wait': False,
        'move_to_rand_start_freq': variant.move_to_rand_start_freq if 'move_to_rand_start_freq' in variant else 1,
        # 'override_workspace_boundaries': [[0.17, -0.08, 0.06, -1.57, 0], [0.35, 0.08, 0.1, 1.57, 0]],
        # broad action boundaries for reaching
        # 'override_workspace_boundaries': [[0.100, - 0.1820, 0.0, -1.57, 0], [0.40, 0.143, 0.24, 1.57, 0]],
        # broad action boundaries for door opening
        'override_workspace_boundaries': [[0.100, -0.25, 0.0, -1.57, 0], [0.41, 0.143, 0.33, 1.57, 0]],

        'action_clipping': 'xyz',
        'catch_environment_except': True,
        'target_point': TARGET_POINT,
        'add_states': variant.add_states,
        'from_states': variant.from_states,
        'reward_type': variant.reward_type,
        'start_transform': None if variant.start_transform == '' else start_transforms[variant.start_transform],
        'randomize_initpos': 'full_area'
    }
    print("getting widowx_real_env env params: ", env_params)
    return env_params


class JaxRLWidowXEnv(WidowXEnv):
    def __init__(self, env_params=None, task_id=None, num_tasks=None, domain_id=None, num_domains=None, fixed_image_size=128,
                 control_viewpoint=0 # used for reward function
                 ):
        super().__init__(env_params)
        self.image_size = fixed_image_size
        self.task_id = task_id
        self.num_tasks = num_tasks
        obs_dict = {}
        if not self._hp.from_states:
            obs_dict['pixels'] = Box(low=0, high=255, shape=(128, 128, 3), dtype=np.uint8)
        if self._hp.add_states:
            obs_dict['state'] = Box(low=-100000, high=100000, shape=(7,), dtype=np.float32)
        if self._hp.add_task_id:
            obs_dict['task_id'] = Box(low=0, high=1, shape=(num_tasks,), dtype=np.float32)
        if num_domains:
            print("ADDING DOMAIN ID")
            obs_dict['domain_id'] = Box(low=0, high=1, shape=(num_domains,), dtype=np.float32)
        # if num_domains:
        #     import pdb; pdb.set_trace()
        #     print("ADDING DOMAIN ID")
        #     obs_dict['domain_id'] = Box(low=0, high=1, shape=(num_domains,), dtype=np.float32)
        self.observation_space = Dict(obs_dict)
        self.move_except = False
        self.control_viewpoint = control_viewpoint
        self.spec = None
        self.requires_timed = True
        self.do_render = True
        self.traj_counter = 0


    def _default_hparams(self):
        from widowx_envs.utils.multicam_server_rospkg.src.topic_utils import IMTopic
        default_dict = {
            'gripper_attached': 'custom',
            'skip_move_to_neutral': True,
            'camera_topics': [IMTopic('/cam0/image_raw')],
            'image_crop_xywh': None,  # can be a tuple like (0, 0, 100, 100)
            'add_states': False,
            'add_task_id':False,
            'from_states': False,
            'reward_type': None
        }
        parent_params = super()._default_hparams()
        parent_params.update(default_dict)
        return parent_params

    def reset(self, itraj=None, reset_state=None):
        if itraj is None:
            itraj = self.traj_counter
        self.traj_counter += 1
        return super().reset(itraj, reset_state)


    def _get_processed_image(self, image=None):
        from skimage.transform import resize
        downsampled_trimmed_image = resize(image, (self.image_size, self.image_size), anti_aliasing=True, preserve_range=True).astype(np.uint8)
        return downsampled_trimmed_image

    def step(self, action):
        obs = super().step(action['action'].squeeze(), action['tstamp_return_obs'], blocking=False)
        reward = 0
        done = obs['full_obs']['env_done']  # done can come from VR buttons
        info = {}
        if self.move_except:
            done = True
            info['Error.truncated'] = True
            # self.move_to_startstate()
        return obs, reward, done, info

    def disable_render(self):
        self.do_render = False

    def enable_render(self):
        self.do_render = True

    def _get_obs(self):
        full_obs = super()._get_obs()
        obs = {}
        if self.do_render:
            processed_images = np.stack([self._get_processed_image(im) for im in full_obs['images']], axis=0)
            obs['pixels'] = processed_images
        obs['full_obs'] = full_obs
        if self._hp.add_states:
            obs['state'] = self.get_full_state()[None]
        return obs

    def set_task_id(self, task_id):
        self.task_id = task_id

    def get_task_id_vec(self, task_id):
        task_id_vec = None
        if (task_id is not None) and self.num_tasks:
            task_id_vec = np.zeros(self.num_tasks, dtype=np.float32)[None]
            task_id_vec[:, task_id] = 1.0
        return task_id_vec

    def move_to_startstate(self, start_state=None):
        # sel_task = self.select_task_from_reward_function()
        paths, tstep = self._hp.start_transform

        successful = False
        itrial = 0
        while not successful:
            print('move to startstate trial ', itrial)
            itrial += 1
            if itrial > 10:
                import pdb; pdb.set_trace()
            try:
                if not isinstance(paths, str):
                    print(f'sampling random start state from {len(paths)} paths ...')
                    sel_path = random.choice(paths)
                    if isinstance(tstep, int):
                        sel_tstep = np.random.randint(0, tstep)
                    elif isinstance(tstep, tuple) and len(tstep) == 2:
                        sel_tstep = np.random.randint(tstep[0], tstep[1])
                    else:
                        raise ValueError('Incorrect tstep index')
                    print('loading starttransform from {} at step {}'.format(sel_path, sel_tstep))
                else:
                    sel_path = paths
                    sel_tstep = tstep
                transform = pkl.load(open(sel_path + '/obs_dict.pkl', 'rb'))['eef_transform'][sel_tstep]
                self._controller.move_to_eep(transform, duration=0.8)
                successful = True
            except Environment_Exception:
                self.move_to_neutral()
        
        import ipdb; ipdb.set_trace()

    def select_task_from_reward_function(self):
        obs = self._get_obs()
        obs_reward = obs.copy()
        obs_reward.pop('full_obs')
        obs_reward['pixels'] = np.expand_dims(obs_reward['pixels'], len(obs_reward['pixels'].shape))

        rewards = []
        for task_id in self.all_tasks:
            obs_reward['task_id'] = self.get_task_id_vec(task_id)
            if 'state' in obs_reward:  # reward classifiers never use proprioceptive states
                obs_reward.pop('state')
            reward = sigmoid(self.reward_function.eval_actions(obs_reward))
            rewards.append(reward)
            print('reward for task {} {}'.format(self.taskid2string[task_id], reward))

        sel_task = self.all_tasks[np.argmin(rewards)]
        print('selected task {} {}'.format(sel_task, self.taskid2string[sel_task]))
        return sel_task

class ImageReachingJaxRLWidowXEnv(JaxRLWidowXEnv):
    def _get_obs(self):
        full_obs = WidowXEnv._get_obs(self)
        obs = {}
        processed_images = np.stack([self._get_processed_image(im) for im in full_obs['images']], axis=0)
        obs['pixels'] = processed_images
        obs['full_obs'] = full_obs
        obs['state'] = self.get_full_state()[None]
        return obs

class BridgeDataJaxRLWidowXRewardAdapter(JaxRLWidowXEnv):
    def __init__(self, env_params=None, reward_function=None, task_id=None, num_tasks=None, fixed_image_size=128,
                 control_viewpoint=0, all_tasks=None, task_id_mapping=None # used for reward function
                 ):

        super().__init__(env_params, task_id, num_tasks, fixed_image_size, control_viewpoint)
        self.reward_function = reward_function
        self.all_tasks = all_tasks
        self.task_string2id = task_id_mapping
        self.taskid2string = {v: k for k, v in self.task_string2id.items()}

    def set_reward_function(self, reward_function):
        self.reward_function = reward_function

    def set_task_id(self, task_id):
        self.task_id = task_id

    def step(self, action):
        obs, reward, done, info = super().step(action)

        if self.reward_function:
            reward = self.infer_reward_for_task(obs, self.task_id)
            # for tsk in self.all_tasks:
            #     self.infer_reward_for_task(obs, tsk)

        return obs, reward, done, info

    def infer_reward_for_task(self, obs, task_id):
        task_id_vec = self.get_task_id_vec(task_id)
        obs_reward = obs.copy()
        obs_reward.pop('full_obs')
        obs_reward['task_id'] = task_id_vec
        obs['task_id'] = task_id_vec
        obs_reward['pixels'] = np.expand_dims(obs_reward['pixels'], len(obs_reward['pixels'].shape))
        if 'state' in obs_reward:  # reward classifiers never use proprioceptive states
            obs_reward.pop('state')
        # print('obs reward shape', obs_reward['pixels'].shape)
        t0 = time.time()
        reward = self.reward_function.eval_actions(obs_reward)
        reward = sigmoid(reward)
        # print('infer rewrard took ', time.time() - t0)
        print("Predicted reward for task {} {}".format(self.taskid2string[task_id], reward))
        reward = np.asarray(reward)[0]
        return reward


class VR_JaxRLWidowXEnv(JaxRLWidowXEnv):
    def __init__(self, env_params=None, task_id=None, num_tasks=None, fixed_image_size=128,
                 control_viewpoint=0 # used for reward function
                 ):

        super().__init__(env_params, task_id, num_tasks, fixed_image_size, control_viewpoint)

        from oculus_reader import OculusReader
        self.oculus_reader = OculusReader()

    def get_vr_buttons(self):
        poses, buttons = self.oculus_reader.get_transformations_and_buttons()
        if 'RG' in buttons:
            buttons['handle'] = buttons['RG']
        else:
            buttons['handle'] = False
        return buttons

    def _default_hparams(self):
        default_dict = {
            'num_task_stages': 1,
            'make_oculus_reader': True

        }
        parent_params = super(VR_JaxRLWidowXEnv, self)._default_hparams()
        parent_params.update(default_dict)
        return parent_params

    def step(self, action):
        """
        :param action:  endeffector velocities
        :return:  observations
        """
        obs, reward, done, info = super(VR_JaxRLWidowXEnv, self).step(action)
        if self.get_vr_buttons()['B']:
            done = True
        return obs, reward, done, info

    def reset(self, itraj=None, reset_state=None):
        obs = super(VR_JaxRLWidowXEnv, self).reset(itraj=itraj)
        start_key = 'handle'
        print('waiting for {} button press to start recording. Press B to go to neutral.'.format(start_key))
        buttons = self.get_vr_buttons()
        while not buttons[start_key]:
            buttons = self.get_vr_buttons()
            rospy.sleep(0.01)
            if 'B' in buttons and buttons['B']:
                self.move_to_neutral()
                print("moved to neutral. waiting for {} button press to start recording.".format(start_key))
        return self._get_obs()

    def ask_confirmation(self):
        print('current endeffector pos', self.get_full_state()[:3])
        print('current joint angles pos', self._controller.get_joint_angles())
        print('Was the trajectory okay? Press A to confirm and RJ to discard')
        buttons = self.get_vr_buttons()
        while not buttons['A'] and not buttons['RJ']:
            buttons = self.get_vr_buttons()
            rospy.sleep(0.01)

        if buttons['RJ']:
            print('trajectory discarded!')
            return False
        if buttons['A']:
            print('trajectory accepted!')
            return True

class VR_JaxRLWidowXEnv_DAgger(VR_JaxRLWidowXEnv):
    def ask_confirmation(self):
        print('Was the trajectory okay? Press A to save a successful trajectory, trigger to save an unsuccessful trajectory, and RJ to discard')
        buttons = self.get_vr_buttons()

        while buttons['A'] or buttons['RJ'] or buttons['RTr']:
            buttons = self.get_vr_buttons()
            rospy.sleep(0.01)

        while not buttons['A'] and not buttons['RJ'] and not buttons['RTr']:
            buttons = self.get_vr_buttons()
            rospy.sleep(0.01)

        if buttons['RJ']:
            print('trajectory discarded!')
            return False
        elif buttons['A']:
            print('successful trajectory accepted!')
            return 'Success'
        elif buttons['RTr']:
            print('unsuccessful trajectory accepted!')
            return 'Failure'

class BridgeDataJaxRLVRWidowXReward(BridgeDataJaxRLWidowXRewardAdapter, VR_JaxRLWidowXEnv_DAgger):
    def __init__(self, env_params=None, reward_function=None, task_id=None, num_tasks=None, fixed_image_size=128, all_tasks=None, task_id_mapping=None):
        super().__init__(env_params=env_params, reward_function=reward_function, task_id=task_id, num_tasks=num_tasks, fixed_image_size=fixed_image_size,
                         all_tasks=all_tasks, task_id_mapping=task_id_mapping)




###################################################
###################################################
# import pdb

# import numpy as np
# import os
# from widowx_envs.widowx.widowx_env import WidowXEnv
# from gym.spaces import Dict
# from gym.spaces import Box
# import time
# from vptr.utils.visualization_utils import sigmoid
# import rospy
# import random
# import pickle as pkl

# from widowx_envs.utils.exceptions import Environment_Exception


# from examples.train_pixels_real import TARGET_POINT

# import glob

# traj_group_open = glob.glob(os.environ['DATA'] + '/robonetv2/toykitchen_fixed_cam/toykitchen1/open_microwave/*/raw/traj_group*/traj*')
# traj_group_close = glob.glob(os.environ['DATA'] + '/robonetv2/toykitchen_fixed_cam/toykitchen1/close_microwave/*/raw/traj_group*/traj*')
# traj_group_sushi = glob.glob(os.environ['DATA'] + '/robonetv2/toykitchen_fixed_cam/toykitchen2/put_sushi_in_pot_cardboard_fence/*/raw/traj_group*/traj*')
# traj_group_sushi_out = glob.glob(os.environ['DATA'] + '/robonetv2/toykitchen_fixed_cam/toykitchen2/take_sushi_out_of_pot_cardboard_fence/*/raw/traj_group*/traj*')
# traj_group_bowlplate = glob.glob(os.environ['DATA'] + '/robonetv2/toykitchen_fixed_cam/toykitchen2/put_bowl_on_plate_cardboard_fence/*/raw/traj_group*/traj*')
# traj_group_cornbowl = glob.glob(os.environ['DATA'] + '/robonetv2/toykitchen_fixed_cam/toykitchen6/put_corn_in_bowl_sink/*/raw/traj_group*/traj*')
# traj_group_cup_plate = glob.glob(os.environ['DATA'] + '/robonetv2/toykitchen_fixed_cam/toykitchen6/put_cup_on_plate/*/raw/traj_group*/traj*')
# traj_group_cup_off_plate = glob.glob(os.environ['DATA'] + '/robonetv2/toykitchen_fixed_cam/toykitchen6/take_cup_off_plate/*/raw/traj_group*/traj*')
# traj_group_corn_out_bowl = glob.glob(os.environ['DATA'] + '/robonetv2/toykitchen_fixed_cam/toykitchen6/take_corn_out_of_bowl_sink/*/raw/traj_group*/traj*')
# traj_group_knife_pot = glob.glob(os.environ['DATA'] + '/robonetv2/toykitchen_fixed_cam_targetdomain/berkeley/toykitchen6/put_knife_into_pot/2022-06-15_13-58-35/raw/traj_group*/traj*')
# traj_group_croissant_pot = glob.glob(os.environ['DATA'] + '/robonetv2/toykitchen_fixed_cam_targetdomain/berkeley/toykitchen6/take_croissant_out_of_pot/2022-06-15_13-38-29/raw/traj_group*/traj*')

# traj_group_tk6_task1 = glob.glob(os.environ['DATA'] + '/robonetv2/toykitchen_fixed_cam_targetdomain/berkeley/toykitchen6/put_knife_into_pot/*/raw/traj_group*/traj*')
# traj_group_tk6_task2 = glob.glob(os.environ['DATA'] + '/robonetv2/toykitchen_fixed_cam_targetdomain/berkeley/toykitchen6/take_croissant_out_of_pot/*/raw/traj_group*/traj*')
# traj_group_tk6_task3 = glob.glob(os.environ['DATA'] + '/robonetv2/toykitchen_fixed_cam_targetdomain/berkeley/toykitchen6/take_pear_from_plate/*/raw/traj_group*/traj*')
# traj_group_tk6_task4 = glob.glob(os.environ['DATA'] + '/robonetv2/toykitchen_fixed_cam_targetdomain/berkeley/toykitchen6/put_sweet_potato_in_bowl/*/raw/traj_group*/traj*')
# traj_group_tk6_task5 = glob.glob(os.environ['DATA'] + '/robonetv2/toykitchen_fixed_cam_targetdomain/berkeley/toykitchen6/put_lime_in_pan_sink/*/raw/traj_group*/traj*')
# traj_group_tk6_task6 = glob.glob(os.environ['DATA'] + '/robonetv2/toykitchen_fixed_cam_targetdomain/berkeley/toykitchen6/put_sweet_potato_on_plate/*/raw/traj_group*/traj*')
# traj_group_tk6_task7 = glob.glob(os.environ['DATA'] + '/robonetv2/toykitchen_fixed_cam_targetdomain/berkeley/toykitchen6/put_drumstick_on_plate/*/raw/traj_group*/traj*')
# traj_group_tk6_task8 = glob.glob(os.environ['DATA'] + '/robonetv2/toykitchen_fixed_cam_targetdomain/berkeley/toykitchen6/put_cucumber_in_orange_pot/*/raw/traj_group*/traj*')
# traj_group_tk6_task9 = glob.glob(os.environ['DATA'] + '/robonetv2/toykitchen_fixed_cam_targetdomain/berkeley/toykitchen6/put_carrot_in_pan/*/raw/traj_group*/traj*')
# traj_group_tk6_task10 = glob.glob(os.environ['DATA'] + '/robonetv2/toykitchen_fixed_cam_targetdomain/berkeley/toykitchen6/put_big_corn_in_big_pot/*/raw/traj_group*/traj*')

# traj_group_tk6_put_cucumber_pot_elevated = glob.glob(os.environ['DATA'] + '/robonetv2/toykitchen_fixed_cam_targetdomain/berkeley/toykitchen6_elevated/put_cucumber_in_orange_pot/*/raw/traj_group*/traj*')
# traj_group_tk6_put_cucumber_pot_elevated_rotated = glob.glob(os.environ['DATA'] + '/robonetv2/toykitchen_fixed_cam_targetdomain/berkeley/toykitchen6_rotated_elevated/put_cucumber_in_orange_pot/*/raw/traj_group*/traj*')
# traj_group_tk6_take_croissant_elevated = glob.glob(os.environ['DATA'] + '/robonetv2/toykitchen_fixed_cam_targetdomain/berkeley/toykitchen6_elevated/take_croissant_out_of_pot//*/raw/traj_group*/traj*')
# traj_group_tk6_take_croissant_rotated = glob.glob(os.environ['DATA'] + '/robonetv2/toykitchen_fixed_cam_targetdomain/berkeley/toykitchen6_rotated/take_croissant_out_of_pot//*/raw/traj_group*/traj*')


# start_transforms = dict(
#     right_front = [os.environ['DATA'] + '/robonetv2/toykitchen_fixed_cam_initial_testconfig/rss_benchmark/toykitchen1/put_broccoli_in_pot/2022-01-22_11-06-56/raw/traj_group0/traj0', 150],
#     middle = [os.environ['DATA'] + '/robonetv2/toykitchen_fixed_cam_initial_testconfig/rss_benchmark/toykitchen1/put_broccoli_in_pot/2022-01-22_11-06-56/raw/traj_group0/traj1', 290],
#     left_front = [os.environ['DATA'] + '/robonetv2/toykitchen_fixed_cam_initial_testconfig/rss_benchmark/toykitchen1/put_broccoli_in_pot/2022-01-22_11-06-56/raw/traj_group0/traj2', 200],
#     left_rear = [os.environ['DATA'] + '/robonetv2/toykitchen_fixed_cam_initial_testconfig/rss_benchmark/toykitchen1/put_broccoli_in_pot/2022-01-22_17-54-21/raw/traj_group0/traj0', 290],
#     right_rear = [os.environ['DATA'] + '/robonetv2/toykitchen_fixed_cam_initial_testconfig/rss_benchmark/toykitchen1/put_broccoli_in_pot/2022-01-22_17-54-21/raw/traj_group0/traj1', 290],

#     openmicrowave = [os.environ['DATA'] + '/robonetv2/toykitchen_fixed_cam/toykitchen1/open_microwave/2021-12-01_16-14-39/raw/traj_group0/traj0', 7],
#     closemicrowave = [os.environ['DATA'] + '/robonetv2/toykitchen_fixed_cam/toykitchen1/close_microwave/2021-12-02_12-14-59/raw/traj_group0/traj1', 0],
#     openmicrowave_sampled = [traj_group_open, 10],
#     # openclosemicrowave_sampled = {3: [traj_group_open, 10], 1:[traj_group_close, (10, 16)]},  # use without aliasing
#     openclosemicrowave_sampled = {1: [traj_group_open, 10], 0:[traj_group_close, (10, 16)]},  # with aliasing
#     reaching = [os.environ['DATA'] + '/robonetv2/online_datacollection/online_reaching/berkeley/toykitchen1/pix_policy_std0.3_initscale0.0_collectdata/2022-04-25_18-10-03/raw/traj_group0/traj0', 0],
#     toykitchen2_sushi_cardboard_sampled = [traj_group_sushi, (5, 10)],
#     # toykitchen2_sushi_cardboard_sampled = [traj_group_sushi, (0, 5)],
#     toykitchen2_sushi_out_cardboard_sampled = [traj_group_sushi_out, (0, 5)],
#     toykitchen2_bowlplate_cardboard_sampled= [traj_group_bowlplate, (0, 5)],
#     toykitchen6_cornbowl_cardboard_sampled= [traj_group_cornbowl, (5, 10)],
#     # toykitchen6_cornbowl_cardboard_sampled= [traj_group_cornbowl, (0, 5)],
#     toykitchen6_cupplate_cardboard_sampled= [traj_group_cup_plate, (0, 5)],
#     toykitchen6_cup_off_plate_cardboard_sampled= [traj_group_cup_off_plate, (8, 13)],
#     # toykitchen6_corn_out_bowl_cardboard_sampled= [traj_group_corn_out_bowl, (0, 3)],
#     toykitchen6_corn_out_bowl_cardboard_sampled= [traj_group_corn_out_bowl, (3, 6)],
#     toykitchen6_knife_pot_cardboard_sampled= [traj_group_knife_pot, (0, 5)],
#     toykitchen6_croissant_pot_cardboard_sampled= [traj_group_croissant_pot, (0, 5)],

#     # tk6_task1=[traj_group_tk6_task1, (0,5)],
#     # tk6_task2=[traj_group_tk6_task2, (0,5)],
#     # tk6_task3=[traj_group_tk6_task3, (0,5)],
#     # tk6_task4=[traj_group_tk6_task4, (0,5)],
#     # tk6_task5=[traj_group_tk6_task5, (0,5)],
#     # tk6_task6=[traj_group_tk6_task6, (0,5)],
#     # tk6_task7=[traj_group_tk6_task7, (0,5)],
#     # tk6_task8=[traj_group_tk6_task8, (0,5)],
#     # tk6_task9=[traj_group_tk6_task9, (0,5)],
#     # tk6_task10=[traj_group_tk6_task10, (0,5)],

#     tk6_task1=[traj_group_tk6_task1, (5,10)],
#     tk6_task2=[traj_group_tk6_task2, (5,10)],
#     tk6_task3=[traj_group_tk6_task3, (5,10)],
#     tk6_task4=[traj_group_tk6_task4, (5,10)],
#     tk6_task5=[traj_group_tk6_task5, (5,10)],
#     tk6_task6=[traj_group_tk6_task6, (5,10)],
#     tk6_task7=[traj_group_tk6_task7, (5,10)],
#     tk6_task8=[traj_group_tk6_task8, (5,10)],
#     tk6_task9=[traj_group_tk6_task9, (5,10)],
#     tk6_task10=[traj_group_tk6_task10, (5,10)],
#     tk6_put_cucumber_pot_elevated=[traj_group_tk6_put_cucumber_pot_elevated, (5,10)],
#     tk6_put_cucumber_pot_elevated_rotated=[traj_group_tk6_put_cucumber_pot_elevated_rotated, (5,10)],
#     tk6_take_croissant_elevated=[traj_group_tk6_take_croissant_elevated, (5,10)],
#     tk6_take_croissant_rotated=[traj_group_tk6_take_croissant_rotated, (5,10)],
# )

# def get_env_params(variant):
#     env_params = {
#         'fix_zangle': True,  # do not apply random rotations to start state
#         'move_duration': 0.2,
#         'adaptive_wait': True,
#         'move_to_rand_start_freq': variant.move_to_rand_start_freq if 'move_to_rand_start_freq' in variant else 1,
#         # 'override_workspace_boundaries': [[0.17, -0.08, 0.06, -1.57, 0], [0.35, 0.08, 0.1, 1.57, 0]],
#         # broad action boundaries for reaching
#         # 'override_workspace_boundaries': [[0.100, - 0.1820, 0.0, -1.57, 0], [0.40, 0.143, 0.24, 1.57, 0]],
#         # broad action boundaries for door opening
#         'override_workspace_boundaries': [[0.100, -0.25, 0.0, -1.57, 0], [0.41, 0.143, 0.33, 1.57, 0]],

#         'action_clipping': 'xyz',
#         'catch_environment_except': True,
#         'target_point': TARGET_POINT,
#         'add_states': variant.add_states,
#         'from_states': variant.from_states,
#         'reward_type': variant.reward_type,
#         'start_transform': None if variant.start_transform == '' else start_transforms[variant.start_transform],
#         'randomize_initpos': 'full_area'
#     }
#     return env_params

# # widowx200 eval env

# import numpy as np
# from widowx_envs.widowx.widowx_env import StateReachingWidowX, \
#     ImageReachingWidowX
# from widowx_envs.utils.params import (
#     WORKSPACE_BOUNDARIES,
#     PERIMETER_SWEEP_WORKSPACE_BOUNDARIES,
#     STARTPOS,
# )
# from widowx_envs.policies.scripted_reach import ReachPolicy
# from widowx_envs.utils.grasp_utils import execute_reach
# from widowx_envs.utils.exceptions import Environment_Exception, Effort_Exceeded_Exception
# from widowx_envs.utils.object_detection import (ObjectDetectorKmeans,
#     ObjectDetectorDL, ObjectDetectorManual, RewardLabelerDL)
# from widowx_envs.utils.reward_labeling_utils import (obj_in_container_classifier,
#     obj_in_container_classifier_rgb)
# from retry import retry
# import time


# class GraspWidowXEnv(ImageReachingWidowX):
#     def __init__(self, env_params=None, publish_images=True, fixed_image_size=128):
#         super(GraspWidowXEnv, self).__init__(
#             env_params=env_params, publish_images=publish_images,
#             fixed_image_size=fixed_image_size,
#         )

#         obs_dict = {}
#         delta_limit = 0.05
#         # max step size: 5cm for each axis
#         # for gripper action, -1 is fully closed, 1 is fully open,
#         # -0.5 < x < 0.5 holds current position

#         self.startpos = STARTPOS

#     def step(self, action):
#         # first three dimensions are delta x, y, z
#         # Last action is gripper action
#         # for gripper action, -1 is fully closed, 1 is fully open,
#         # -0.5 < x < 0.5 holds current position
#         movement_failed = False
#         efforts_exceeded = False
#         try:
#             obs = super(StateReachingWidowX, self).step(action)
#         except Environment_Exception:
#             obs = self._get_obs()
#             movement_failed = True
#         except Effort_Exceeded_Exception:
#             obs = self._get_obs()
#             efforts_exceeded = True
#         camera_error = self.is_camera_error()

#         reward = 0.
#         done = False

#         # if obs['state'][-1] > 0.7:
#         #     self.is_gripper_open = True
#         # else:
#         #     self.is_gripper_open = False
#         # obs = self._get_obs()

#         if self.publish_images:
#             self._publish_image(obs['image'])

#         time_diff = self._get_step_time_diff()
#         info = {'movement_failed': movement_failed, 'camera_error': camera_error, 'time_diff': time_diff, 'efforts_exceeded': efforts_exceeded}
#         return obs, reward, done, info

#     def reset(self, itraj=None, reset_state=None):
#         """
#         Resets the environment and returns initial observation
#         :return: obs dict (look at step(self, action) for documentation)
#         """
#         self._controller.open_gripper(True)
#         # self.is_gripper_open = True

#         if not self._hp.skip_move_to_neutral:
#             self._controller.move_to_neutral(duration=1.5)

#         # if itraj % self._hp.move_to_rand_start_freq == 0:
#         #     self.move_to_startstate()
#         zangle = 0.
#         currpos = self._get_obs()['state'][:3]
#         currangle = self._get_obs()['state'][-3]
#         self._controller.move_to_state(self.startpos, zangle, duration=1.5)
#         self._reset_previous_qpos()
#         self.last_step_time = None

#         # time.sleep(self._hp.wait_time)  # sleep is already called in self._reset_previous_qpos()
#         return self._get_obs()

#     def sweep_tray_perimeter(self):
#         self._hp.override_workspace_boundaries = PERIMETER_SWEEP_WORKSPACE_BOUNDARIES
#         self._setup_robot()
#         eps = 0.0

#         x_lo, x_hi = WORKSPACE_BOUNDARIES[0][0] + eps, WORKSPACE_BOUNDARIES[1][0] - eps
#         y_lo, y_hi = WORKSPACE_BOUNDARIES[0][1] + eps, WORKSPACE_BOUNDARIES[1][1] - eps
#         z = 0.04
#         corners = [
#             np.array([x_lo, y_hi, z]),
#             np.array([x_hi, y_hi, z]),
#             np.array([x_hi, y_lo, z]),
#             np.array([x_lo, y_lo, z]),
#         ]
#         corners = [self.rotate_point(corner) for corner in corners]
#         corners.append(corners[0])
#         # Something related to workspace boundaries?
#         reach_policy = ReachPolicy(self, reach_point=None)
#         for corner in corners:
#             obs = execute_reach(self, reach_policy, corner, noise=0.01)
#             # self._controller.move_to_state(corner, angle, duration=2)
#             # add some noise
#             # or use the reaching policy

#         self._hp.override_workspace_boundaries = WORKSPACE_BOUNDARIES
#         self._setup_robot()

#         self.reset()

#     @retry(tries=5)
#     def _get_obs(self):
#         full_obs = super(StateReachingWidowX, self)._get_obs()
#         image = full_obs['images'][0]
#         ee_coord = full_obs['full_state'][:3]
#         processed_image = self._get_processed_image(image)

#         obs = {'image': processed_image, 'state': self.get_full_state(),
#                'joints': full_obs['qpos'], 'ee_coord': ee_coord, 'wrist_angle': full_obs['wrist_angle']}

#         if self._hp.return_full_image:
#             obs['full_image'] = image
            
        
#         # obs['gripper'] = full_obs['state'][-1]  # this dimension is not being updated for now
#         return obs

#     def _default_hparams(self):
#         default_dict = {
#             'override_workspace_boundaries': WORKSPACE_BOUNDARIES,
#             'continuous_gripper': False
#         }
#         parent_params = super(GraspWidowXEnv, self)._default_hparams()
#         parent_params.update(default_dict)
#         return parent_params

#     def relabel_rewards(self, rollout, success):
#         new_rewards = np.zeros(rollout['rewards'].shape)
#         self.last_trajectory_successful = bool(success)

#         dropped = False
#         dropped_idx = None

#         for i in range(len(new_rewards) - 1, 2, -1):
#             if rollout['actions'][i][-1] > 0.5 and rollout['actions'][i-1][-1] < 0.5:
#                 dropped = True
#                 dropped_idx = i
#                 break

#         if dropped:
#             for i in range(dropped_idx, len(new_rewards)):
#                 new_rewards[i] = float(success)
#         rollout['rewards'] = new_rewards

#         for i in range(len(rollout['rewards'])):
#             rollout['env_infos'][i]['reward'] = rollout['rewards'][i][0]

#         return rollout



# class JaxRLWidowXEnv(WidowXEnv):
#     def __init__(self, env_params=None, task_id=None, num_tasks=None, domain_id=None, num_domains=None, fixed_image_size=128,
#                  control_viewpoint=0 # used for reward function
#                  ):

#         super().__init__(env_params)
#         self.image_size = fixed_image_size
#         self.task_id = task_id
#         self.num_tasks = num_tasks

#         obs_dict = {}
#         if not self._hp.from_states:
#             obs_dict['pixels'] = Box(low=0, high=255, shape=(128, 128, 3), dtype=np.uint8)
#         if self._hp.add_states:
#             obs_dict['state'] = Box(low=-100000, high=100000, shape=(7,), dtype=np.float32)
#         if self._hp.add_task_id:
#             obs_dict['task_id'] = Box(low=0, high=1, shape=(num_tasks,), dtype=np.float32)
#         # if num_domains:
#         #     import pdb; pdb.set_trace()
#         #     print("ADDING DOMAIN ID")
#         #     obs_dict['domain_id'] = Box(low=0, high=1, shape=(num_domains,), dtype=np.float32)
#         self.observation_space = Dict(obs_dict)
#         self.move_except = False
#         self.control_viewpoint = control_viewpoint
#         self.spec = None
#         self.requires_timed = True
#         self.do_render = True
#         self.traj_counter = 0

#     def _default_hparams(self):
#         from widowx_envs.utils.multicam_server_rospkg.src.topic_utils import IMTopic
#         default_dict = {
#             'gripper_attached': 'custom',
#             'skip_move_to_neutral': True,
#             'camera_topics': [IMTopic('/cam0/image_raw')],
#             'image_crop_xywh': None,  # can be a tuple like (0, 0, 100, 100)
#             'add_states': False,
#             'add_task_id':False,
#             'from_states': False,
#             'reward_type': None
#         }
#         parent_params = super()._default_hparams()
#         parent_params.update(default_dict)
#         return parent_params

#     def reset(self, itraj=None, reset_state=None):
#         if itraj is None:
#             itraj = self.traj_counter
#         self.traj_counter += 1
#         return super().reset(itraj, reset_state)


#     def _get_processed_image(self, image=None):
#         from skimage.transform import resize
#         downsampled_trimmed_image = resize(image, (self.image_size, self.image_size), anti_aliasing=True, preserve_range=True).astype(np.uint8)
#         return downsampled_trimmed_image

#     def step(self, action):
#         obs = super().step(action['action'].squeeze(), action['tstamp_return_obs'], blocking=False)
#         reward = 0
#         done = obs['full_obs']['env_done']  # done can come from VR buttons
#         info = {}
#         if self.move_except:
#             done = True
#             info['Error.truncated'] = True
#             # self.move_to_startstate()
#         return obs, reward, done, info

#     def disable_render(self):
#         self.do_render = False

#     def enable_render(self):
#         self.do_render = True

#     def _get_obs(self):
#         full_obs = super()._get_obs()
#         obs = {}
#         if self.do_render:
#             processed_images = np.stack([self._get_processed_image(im) for im in full_obs['images']], axis=0)
#             obs['pixels'] = processed_images
#         obs['full_obs'] = full_obs
#         if self._hp.add_states:
#             obs['state'] = self.get_full_state()[None]
#         return obs

#     def set_task_id(self, task_id):
#         self.task_id = task_id

#     def get_task_id_vec(self, task_id):
#         task_id_vec = None
#         if (task_id is not None) and self.num_tasks:
#             task_id_vec = np.zeros(self.num_tasks, dtype=np.float32)[None]
#             task_id_vec[:, task_id] = 1.0
#         return task_id_vec

#     def move_to_startstate(self, start_state=None):
#         # sel_task = self.select_task_from_reward_function()
#         paths, tstep = self._hp.start_transform

#         successful = False
#         itrial = 0
#         print('entering move to startstate loop.')
#         while not successful:
#             print('move to startstate trial ', itrial)
#             itrial += 1
#             if itrial > 10:
#                 import pdb; pdb.set_trace()
#             try:
#                 if not isinstance(paths, str):
#                     print(f'sampling random start state from {len(paths)} paths ...')
#                     sel_path = random.choice(paths)
#                     if isinstance(tstep, int):
#                         sel_tstep = np.random.randint(0, tstep)
#                     elif isinstance(tstep, tuple) and len(tstep) == 2:
#                         sel_tstep = np.random.randint(tstep[0], tstep[1])
#                     else:
#                         raise ValueError('Incorrect tstep index')
#                     print('loading starttransform from {} at step {}'.format(sel_path, sel_tstep))
#                 else:
#                     sel_path = paths
#                     sel_tstep = tstep
#                 transform = pkl.load(open(sel_path + '/obs_dict.pkl', 'rb'))['eef_transform'][sel_tstep]
#                 self.controller.move_to_starteep(transform, duration=0.8)
#                 successful = True
#             except Environment_Exception:
#                 self.move_to_neutral()
        
#         import pdb; pdb.set_trace()

#     def select_task_from_reward_function(self):
#         obs = self._get_obs()
#         obs_reward = obs.copy()
#         obs_reward.pop('full_obs')
#         obs_reward['pixels'] = np.expand_dims(obs_reward['pixels'], len(obs_reward['pixels'].shape))

#         rewards = []
#         for task_id in self.all_tasks:
#             obs_reward['task_id'] = self.get_task_id_vec(task_id)
#             if 'state' in obs_reward:  # reward classifiers never use proprioceptive states
#                 obs_reward.pop('state')
#             reward = sigmoid(self.reward_function.eval_actions(obs_reward))
#             rewards.append(reward)
#             print('reward for task {} {}'.format(self.taskid2string[task_id], reward))

#         sel_task = self.all_tasks[np.argmin(rewards)]
#         print('selected task {} {}'.format(sel_task, self.taskid2string[sel_task]))
#         return sel_task

# class ImageReachingJaxRLWidowXEnv(JaxRLWidowXEnv):
#     def _get_obs(self):
#         full_obs = WidowXEnv._get_obs(self)
#         obs = {}
#         processed_images = np.stack([self._get_processed_image(im) for im in full_obs['images']], axis=0)
#         obs['pixels'] = processed_images
#         obs['full_obs'] = full_obs
#         obs['state'] = self.get_full_state()[None]
#         return obs

# class BridgeDataJaxRLWidowXRewardAdapter(JaxRLWidowXEnv):
#     def __init__(self, env_params=None, reward_function=None, task_id=None, num_tasks=None, fixed_image_size=128,
#                  control_viewpoint=0, all_tasks=None, task_id_mapping=None # used for reward function
#                  ):

#         super().__init__(env_params, task_id, num_tasks, fixed_image_size, control_viewpoint)
#         self.reward_function = reward_function
#         self.all_tasks = all_tasks
#         self.task_string2id = task_id_mapping
#         self.taskid2string = {v: k for k, v in self.task_string2id.items()}

#     def set_reward_function(self, reward_function):
#         self.reward_function = reward_function

#     def set_task_id(self, task_id):
#         self.task_id = task_id

#     def step(self, action):
#         obs, reward, done, info = super().step(action)

#         if self.reward_function:
#             reward = self.infer_reward_for_task(obs, self.task_id)
#             # for tsk in self.all_tasks:
#             #     self.infer_reward_for_task(obs, tsk)

#         return obs, reward, done, info

#     def infer_reward_for_task(self, obs, task_id):
#         task_id_vec = self.get_task_id_vec(task_id)
#         obs_reward = obs.copy()
#         obs_reward.pop('full_obs')
#         obs_reward['task_id'] = task_id_vec
#         obs['task_id'] = task_id_vec
#         obs_reward['pixels'] = np.expand_dims(obs_reward['pixels'], len(obs_reward['pixels'].shape))
#         if 'state' in obs_reward:  # reward classifiers never use proprioceptive states
#             obs_reward.pop('state')
#         # print('obs reward shape', obs_reward['pixels'].shape)
#         t0 = time.time()
#         reward = self.reward_function.eval_actions(obs_reward)
#         reward = sigmoid(reward)
#         # print('infer rewrard took ', time.time() - t0)
#         print("Predicted reward for task {} {}".format(self.taskid2string[task_id], reward))
#         reward = np.asarray(reward)[0]
#         return reward


# class VR_JaxRLWidowXEnv(JaxRLWidowXEnv):
#     def __init__(self, env_params=None, task_id=None, num_tasks=None, fixed_image_size=128,
#                  control_viewpoint=0 # used for reward function
#                  ):

#         super().__init__(env_params, task_id, num_tasks, fixed_image_size, control_viewpoint)

#         from oculus_reader import OculusReader
#         self.oculus_reader = OculusReader()

#     def get_vr_buttons(self):
#         poses, buttons = self.oculus_reader.get_transformations_and_buttons()
#         if 'RG' in buttons:
#             buttons['handle'] = buttons['RG']
#         else:
#             buttons['handle'] = False
#         return buttons

#     def _default_hparams(self):
#         default_dict = {
#             'num_task_stages': 1,
#             'make_oculus_reader': True

#         }
#         parent_params = super(VR_JaxRLWidowXEnv, self)._default_hparams()
#         parent_params.update(default_dict)
#         return parent_params

#     def step(self, action):
#         """
#         :param action:  endeffector velocities
#         :return:  observations
#         """
#         obs, reward, done, info = super(VR_JaxRLWidowXEnv, self).step(action)
#         if self.get_vr_buttons()['B']:
#             done = True
#         return obs, reward, done, info

#     def reset(self, itraj=None, reset_state=None):
#         obs = super(VR_JaxRLWidowXEnv, self).reset(itraj=itraj)
#         start_key = 'handle'
#         print('waiting for {} button press to start recording. Press B to go to neutral.'.format(start_key))
#         buttons = self.get_vr_buttons()
#         while not buttons[start_key]:
#             buttons = self.get_vr_buttons()
#             rospy.sleep(0.01)
#             if 'B' in buttons and buttons['B']:
#                 self.move_to_neutral()
#                 print("moved to neutral. waiting for {} button press to start recording.".format(start_key))
#         return self._get_obs()

#     def ask_confirmation(self):
#         print('current endeffector pos', self.get_full_state()[:3])
#         print('current joint angles pos', self.controller.get_joint_angles())
#         print('Was the trajectory okay? Press A to confirm and RJ to discard')
#         buttons = self.get_vr_buttons()
#         while not buttons['A'] and not buttons['RJ']:
#             buttons = self.get_vr_buttons()
#             rospy.sleep(0.01)

#         if buttons['RJ']:
#             print('trajectory discarded!')
#             return False
#         if buttons['A']:
#             print('trajectory accepted!')
#             return True

# class VR_JaxRLWidowXEnv_DAgger(VR_JaxRLWidowXEnv):
#     def ask_confirmation(self):
#         print('Was the trajectory okay? Press A to save a successful trajectory, trigger to save an unsuccessful trajectory, and RJ to discard')
#         buttons = self.get_vr_buttons()

#         while buttons['A'] or buttons['RJ'] or buttons['RTr']:
#             buttons = self.get_vr_buttons()
#             rospy.sleep(0.01)

#         while not buttons['A'] and not buttons['RJ'] and not buttons['RTr']:
#             buttons = self.get_vr_buttons()
#             rospy.sleep(0.01)

#         if buttons['RJ']:
#             print('trajectory discarded!')
#             return False
#         elif buttons['A']:
#             print('successful trajectory accepted!')
#             return 'Success'
#         elif buttons['RTr']:
#             print('unsuccessful trajectory accepted!')
#             return 'Failure'

# class BridgeDataJaxRLVRWidowXReward(BridgeDataJaxRLWidowXRewardAdapter, VR_JaxRLWidowXEnv_DAgger):
#     def __init__(self, env_params=None, reward_function=None, task_id=None, num_tasks=None, fixed_image_size=128, all_tasks=None, task_id_mapping=None):
#         super().__init__(env_params=env_params, reward_function=reward_function, task_id=task_id, num_tasks=num_tasks, fixed_image_size=fixed_image_size,
#                          all_tasks=all_tasks, task_id_mapping=task_id_mapping)



