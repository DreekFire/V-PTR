# V-PTR Evaluation Protocol

This is the evaluation protocol we used in our lab. It is probably not relevant to you unless you are also working at RAIL.

## Physical Setup
### Toykitchen 1 (for microwave task)
- Line up back corners of kitchen inside thin yellow electric tape
- Set desk height to 35.0
### Toykitchen 6 (for pick and place tasks)
- Line up back corners of kitchen inside thick black gaffer tape
- Set desk height to 24.3
## Evaluation Setups
### Pick and Place
1. Iterate through the following start transforms at https://github.com/DreekFire/V-PTR/blob/4794668ef51d58aa58b26abe30d96c6500626da3/vptr/extra_envs/widowx_real_env.py#L64:
- `right_rear`
- `right_front`
- `left_front`
- `left_rear`
2. For each start transform, evaluate on the following manual setups:
- `under`: place target object directly underneath the gripper
- `shift_1`: shift the object a few inches either up, down, left, or right such that it is closer to the center of the box created by the start transforms (to avoid exiting environment bounds)
- `shift_2`: same as shift_1 but shift in the perpendicular direction that is still towards the center

  For long target objects (croissant, cucumber, knife, carrot, etc.), orient their length in the same direction as the gripper opening. For put in tasks, containers can be placed a few inches from target objects provided they are within environment bounds.
### Sweep Beans
1. Iterate only through the 
- `right_rear`
- `right_front`
2. Place the tray on the left side of toy kitchen 6, leaving about half an inch of clearance from the left. Distribute a handful of beans evenly throughout the tray.
3. For each start transform, evaluate on the following manual setups:
- `under`: place the sweeper directly underneath the gripper, aligned with the opening of the gripper
- `shift_1`: while maintaining the sweeper orientation, shift it along its long axis to the top of the tray
- `shift_2`: same as shift_1 but towards the bottom of the tray

### Microwave
Evaluate on each of the start transforms `openmicrowave1` through `openmicrowave12` at https://github.com/DreekFire/V-PTR/blob/4794668ef51d58aa58b26abe30d96c6500626da3/vptr/extra_envs/widowx_real_env.py#L81.
## Distractor Evaluations
Pick 2 distractor objects and place them in the scene so that they are visible but do not interfere with the robot completing the task. Shift or swap them around when changing manual setups. Swap them out for new objects for each start transform.
## Success Definition
### Put In Tasks
The robot succeeds if it lifts the object and places it inside the container.
### Take Out Tasks
The robot succeeds if it lifts the object out of the container and keeps it outside afterward. Knocking the object out of the container without lifting is a failure.
### Sweep Tasks
The robot succeeds if it moves more than half way across the tray while keeping the sweeper close enough to the tray to move the beans. The long axis of the sweeper should also remain somewhat perpendicular to the direction of motion.
### Open Microwave Task
The robot succeeds if opens the microwave door by pulling the handle (first step) and begins to push the door open from behind (second step).