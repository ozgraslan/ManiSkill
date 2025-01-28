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

from einops import rearrange


env: gym.Env = gym.make(
  "PickCube-v1",
  robot_uids="panda_robotiq",
  obs_mode="rgb+segmentation",
  render_mode="human",
  num_envs=2, # if num_envs > 1, GPU simulation backend is used.
)

# env = MaskResizeRGBSegObsWrapper(env, new_size=(128, 128))
obs, _ = env.reset()
print(obs)
rgb = obs["sensor_data"]["base_camera"]["rgb"]
seg = obs["sensor_data"]["base_camera"]["segmentation"]
# zero_img = torch.zeros_like(rgb)
# mask = torch.ones_like(seg)
# mask[
#     torch.isin(
#         seg,
#         torch.cat([env.robot_link_ids, env.target_object_actor_ids, torch.tensor([env.scene.actors["sep_flat_table"]._objs[0].per_scene_id], device=env.device)]),
#     )
# ] = 0
# rgb = rgb * (1 - mask) + zero_img * mask
# print(rgb.shape, mask.shape, zero_img.shape)

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

seg = (seg * torch.tensor([11, 61, 127], device=seg.device)).to(torch.uint8)
combined = np.concatenate([np.concatenate(rgb.cpu().numpy(), axis=1), np.concatenate(seg.cpu().numpy(), axis=1)], axis=0)
plt.imshow(combined)
plt.show()

# while True:
#     action = env.action_space.sample() # replace this with your policy inference
#     obs, reward, terminated, truncated, info = env.step(action)
#     env.render()
