import gymnasium as gym
import numpy as np
import torch
import matplotlib.pyplot as plt
import torchvision.transforms as T
from mani_skill.envs.tasks import PickCubeEnv
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import camera_observations_to_images
from mani_skill.trajectory.dataset import ManiSkillTrajectoryDataset
from mani_skill.utils.wrappers.obs import MaskResizeRGBSegObsWrapper
from mani_skill.trajectory.utils.actions import conversion
from einops import rearrange


env: gym.Env = gym.make(
  "PickCube-v1",
  robot_uids="panda_robotiq",
  obs_mode="rgb+segmentation",
  render_mode="rgb_array",
  num_envs=1, # if num_envs > 1, GPU simulation backend is used.
  control_mode="pd_joint_delta_pos",
  sensor_configs=dict(shader_pack="default"),
  human_render_camera_configs=dict(shader_pack="default"),
  viewer_camera_configs=dict(shader_pack="default"),
)

print(env.agent.controller.controllers["arm"])
# env = MaskResizeRGBSegObsWrapper(env, masked_obj_list=["ground"], new_size=(128, 128), normalize=False)
print(env.action_space)
obs, _ = env.reset()
print(obs["agent"]["qpos"])

lst = []
rgb = obs["sensor_data"]["base_camera"]["rgb"]  
seg = obs["sensor_data"]["base_camera"]["segmentation"]
seg = (seg * torch.tensor([11, 61, 127], device=seg.device)).to(torch.uint8)
combined = np.concatenate([np.concatenate(rgb.cpu().numpy(), axis=1), np.concatenate(seg.cpu().numpy(), axis=1)], axis=0)
lst.append(combined)

action = np.zeros((1,8))
action[:,-1] = 0.6
print(action)
obs, _, _, _, _ = env.step(action)
# obs, _, _, _, _ = env.step(action)
# obs, _, _, _, _ = env.step(action)

print(obs["agent"]["qpos"])

rgb = obs["sensor_data"]["base_camera"]["rgb"]  
seg = obs["sensor_data"]["base_camera"]["segmentation"]
seg = (seg * torch.tensor([11, 61, 127], device=seg.device)).to(torch.uint8)
combined = np.concatenate([np.concatenate(rgb.cpu().numpy(), axis=1), np.concatenate(seg.cpu().numpy(), axis=1)], axis=0)
lst.append(combined)

# action = np.zeros((1,8))
# action[:,-1] = 1
# print(action)
# obs, _, _, _, _ = env.step(action)
# print(obs["agent"]["qpos"])

# rgb = obs["sensor_data"]["base_camera"]["rgb"]  
# seg = obs["sensor_data"]["base_camera"]["segmentation"]
# seg = (seg * torch.tensor([11, 61, 127], device=seg.device)).to(torch.uint8)
# combined = np.concatenate([np.concatenate(rgb.cpu().numpy(), axis=1), np.concatenate(seg.cpu().numpy(), axis=1)], axis=0)
# lst.append(combined)

plt.imshow(np.concatenate(lst, axis=1))
plt.show()

# print()
# for x in env.scene.actors.values():
#     print(x.name, x._objs[0].per_scene_id)
# dataset_file = "demos/PutCarrotOnPlateInSceneSep-v1/motionplanning/20241217_202852.h5"
# dataset = ManiSkillTrajectoryDataset(dataset_file=dataset_file, load_count=2, device="cpu")
# obs = dataset[:1]["obs"]
# print(obs["sensor_data"].keys())
# exit(0)
# images = camera_observations_to_images(obs["sensor_data"]["3rd_view_camera"])
# print(images["segmentation"].shape)
# images["rgb"] = rearrange( T.Resize((256, 256))(rearrange(images["rgb"], "b h w c -> b c h w")), "b c h w -> b h w c")
# images["segmentation"] = rearrange( T.Resize((256, 256))(rearrange(images["segmentation"], "b h w c -> b c h w")), "b c h w -> b h w c")

# rgb = np.concatenate(images["rgb"].cpu().numpy(), axis=1)
# segment = np.concatenate(images["segmentation"].cpu().numpy(), axis=1)

# seg = (seg * torch.tensor([11, 61, 127], device=seg.device)).to(torch.uint8)
# combined = np.concatenate([np.concatenate(rgb.cpu().numpy(), axis=1), np.concatenate(seg.cpu().numpy(), axis=1)], axis=0)
# plt.imshow(combined)
# plt.show()

# while True:
#     action = env.action_space.sample() # replace this with your policy inference
#     obs, reward, terminated, truncated, info = env.step(action)
#     env.render()
