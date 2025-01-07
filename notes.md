To run motion planning code (only saves successfull trajectories): 
1 process cpu backend seems to be the best.

```
python mani_skill/examples/motionplanning/panda_robotiq/run.py -e PutCarrotOnPlateInSceneSep-v1 -o rgb+segmentation -n 1024 --num-procs 1  --sim-backend cpu  --reward-mode dense --only-count-success
```

To run rgb based bc training:

```
python examples/baselines/bc/bc_rgb.py --env_id PutCarrotOnPlateInSceneSep-v1 --demo_path demos/PutCarrotOnPlateInSceneSep-v1/motionplanning/20250104_184007.h5 --num_demos 1024 --total_iters 100002 --control_mode pd_joint_pos --batch_size 64 --sim_backend gpu --track --save_freq 100000

```

To run rgb based evaluation:

```
python examples/baselines/bc/bc_rgb_eval.py --env_id PutCarrotOnPlateInSceneSep-v1 --demo_path demos/PutCarrotOnPlateInSceneSep-v1/motionplanning/20250104_184007.h5 --exp_name PutCarrotOnPlateInSceneSep-v1__bc_rgb__1__1736128977 --control_mode pd_joint_pos --num_eval_envs 1 --sim_backend gpu --track --num_eval_episodes 10
```

There is somethig wrong with is_grasping function in agents/robots/panda*

Changed segment_rgb to mask_background.

First results show mask background version working worse.
But it can be due to torch.compile? Disabled it and training again.