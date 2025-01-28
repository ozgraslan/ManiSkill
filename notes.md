To run motion planning code (only saves successfull trajectories): 
1 process cpu backend seems to be the best.

```
python mani_skill/examples/motionplanning/panda_robotiq/run.py -e PutCarrotOnPlateInSceneSep-v1 -o rgb+segmentation -n 1024 --num-procs 1  --sim-backend cpu  --reward-mode dense --only-count-success
```

```
python mani_skill/examples/motionplanning/panda_robotiq/run.py -e PickCube-v1 -o rgb+segmentation -n 64 --num-procs 1  --sim-backend cpu --reward-mode dense --traj-name pick_cube --save-video --only-count-success --new-size 128
```

To run rgb based bc training:

```
python examples/baselines/bc/bc_rgb.py --env_id PutCarrotOnPlateInSceneSep-v1 --demo_path demos/PutCarrotOnPlateInSceneSep-v1/motionplanning/20250104_184007.h5 --num_demos 1024 --total_iters 10002 --control_mode pd_joint_pos --batch_size 64 --sim_backend gpu --track --save_freq 1000

```

To run rgb based evaluation:

```
python examples/baselines/bc/bc_rgb_eval.py --env_id PutCarrotOnPlateInSceneSep-v1 --demo_path demos/PutCarrotOnPlateInSceneSep-v1/motionplanning/20250104_184007.h5 --exp_name PutCarrotOnPlateInSceneSep-v1__bc_rgb__1__1736458513 --control_mode pd_joint_pos --num_eval_envs 1 --sim_backend gpu --num_eval_episodes 10 --checkpoint 9999 --control_mode pd_joint_pos

```

Notes:
- There is somethig wrong with is_grasping function in agents/robots/panda_robotiq_2f_85
    - Checked the panda implementation with motionplanning and it works correctly.
    - Changed some variable initialization and usage in is_grasping function of panda_robotiq to be same as panda implementation.
    - The problem was how l_contact_forces and r_contact_forces is computed. 
        - In panda implementation for computing even if the finger links are different, their direction to open the gripper is the same. Therefore in panda implementation negative of rdirection is taken.
        - In panda_robotiq implementation fingerpads' direction to open the gipper is already opposite so no need to take the negative. 
        - This might be due to how urdfs are implemented.

- Changed segment_rgb to mask_background.

- First results show mask background version working worse.

- Initializing tensorboard before wandb results in wandb not syncing the tensorboard logs.

- For partial object initialization (in multiple env intances when some envs need reset some do not), objects that are created using self.scene.create_actor_builder() do the partial resets correctly. In reset function of sapien_env.py self.scene._reset_mask is set to mask objects of scene instances which are not reseted. Then this mask is used in robots, actors, etc. to correctly partial reset these objects.

- Removed episode_stats and consequtive_grasp tensor variables from BaseBridge env.
    - We do not use consequtive_grasp anywhere, and we do not need to keep track of values in episode_stats. We can just compute and return them. No need to store them in a dictionary variable.

- In ppo_fast.py there is an error if in evaluation the number of evaluation steps are finished before the environment finishes executing. This results in no "final_info" being returned and since we want to print, error occurs. 
    - It can be solved by increasing num_eval_steps to a higher or equal number to env's maximum number of steps. Or extracting values of "success_once" and "return", eve if there is no "final_info".