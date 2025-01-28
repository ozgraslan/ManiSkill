import copy
from typing import Dict, Tuple, Optional

import gymnasium as gym
import numpy as np
import torch
import torchvision.transforms as T

from mani_skill.envs.sapien_env import BaseEnv

class ResizeRGBSegObservationWrapper(gym.ObservationWrapper):
    """
    Resizes RGB and segmentation images and keeps the observation structure the same

    Args:
        new_size: tuple of new height and width
    """

    def __init__(self, env, new_size: Tuple[int, int] = (128, 128), normalize=False) -> None:
        self.base_env: BaseEnv = env.unwrapped
        super().__init__(env)
        self.resizer = T.Resize(new_size)
        self.normalize = normalize
        
        new_obs = self.observation(self.base_env._init_raw_obs)
        self.base_env.update_obs_space(new_obs)

    def observation(self, observation: Dict) -> Dict:
        for cam_key in observation["sensor_data"].keys():
            rgb = observation["sensor_data"][cam_key]["rgb"]            
            rgb = self.resizer(rgb.permute(0,3,1,2)).permute(0,2,3,1)
            
            if self.normalize:
                rgb = rgb / 255.0
                rgb = rgb * 2 -1
            observation["sensor_data"][cam_key]["rgb"] = rgb

            if "segmentation" in observation["sensor_data"][cam_key].keys():
                seg = observation["sensor_data"][cam_key]["segmentation"]
                seg = self.resizer(seg.permute(0,3,1,2)).permute(0,2,3,1)
                observation["sensor_data"][cam_key]["segmentation"] = seg
                
        return observation


class MaskResizeRGBSegObsWrapper(gym.ObservationWrapper):
    """
    Resizes RGB and segmentation images, then masks the background of the RGB using the segmentation image.
    Keeps the observation structure the same

    Args:
        new_size: tuple of new height and width
    """

    def __init__(self, env, new_size: Tuple[int, int] = (128, 128), normalize: bool = False) -> None:
        self.base_env: BaseEnv = env.unwrapped
        super().__init__(env)
        self.resizer = T.Resize(new_size)
        self.not_masked_ids = torch.cat([self.base_env.robot_link_ids, self.base_env.target_object_actor_ids, 
                        torch.tensor([self.base_env.scene.actors["sep_flat_table"]._objs[0].per_scene_id], device=self.base_env.device)])
        self.normalize = normalize
        
        new_obs = self.observation(self.base_env._init_raw_obs)
        self.base_env.update_obs_space(new_obs)


    def observation(self, observation: Dict) -> Dict:
        for cam_key in observation["sensor_data"].keys():
            rgb = observation["sensor_data"][cam_key]["rgb"]
            seg = observation["sensor_data"][cam_key]["segmentation"]
            
            rgb = self.resizer(rgb.permute(0,3,1,2)).permute(0,2,3,1)
            seg = self.resizer(seg.permute(0,3,1,2)).permute(0,2,3,1)

            mask = torch.ones_like(seg)
            mask[
                torch.isin(
                    seg,
                    self.not_masked_ids
                )
            ] = 0
            if self.normalize:
                rgb = rgb / 255.0
                rgb = rgb * 2 -1
            zero_img = torch.zeros_like(rgb, dtype=rgb.dtype, device=rgb.device)
            rgb = rgb * (1 - mask) + zero_img * mask

            # import matplotlib.pyplot as plt

            # plt.imshow(rgb[0].cpu().numpy())
            # plt.show()

            # exit(0)
            observation["sensor_data"][cam_key]["rgb"] = rgb.float()
            observation["sensor_data"][cam_key]["segmentation"] = seg.float()
        return observation