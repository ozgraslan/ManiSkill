import os
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

import gymnasium as gym
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from mani_skill.utils import gym_utils
from mani_skill.utils.io_utils import load_json
from mani_skill.utils.wrappers.obs import MaskResizeRGBSegObsWrapper, ResizeRGBSegObservationWrapper
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import BatchSampler, RandomSampler
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

from behavior_cloning.evaluate import evaluate
from behavior_cloning.make_env import make_eval_envs

@dataclass
class Args:
    exp_name: Optional[str] = None
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = False
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "bc_pick_carrot_on_plate"
    """the wandb's project name"""
    wandb_entity: Optional[str] = "jakd9"
    """the entity (team) of wandb's project"""
    capture_video: bool = True
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    env_id: str = "PegInsertionSide-v0"
    """the id of the environment"""
    demo_path: str = "data/ms2_official_demos/rigid_body/PegInsertionSide-v0/trajectory.state.pd_ee_delta_pose.h5"
    """the path of demo dataset (pkl or h5)"""

    # Behavior cloning specific arguments
    normalize_states: bool = False
    """if toggled, states are normalized to mean 0 and standard deviation 1"""
    lr: float = 3e-4
    """the learning rate for the actor"""

    # Environment/experiment specific arguments
    max_episode_steps: Optional[int] = None
    """Change the environments' max_episode_steps to this value. Sometimes necessary if the demonstrations being imitated are too short. Typically the default
    max episode steps of environments in ManiSkill are tuned lower so reinforcement learning agents can learn faster."""
    num_eval_episodes: int = 100
    """the number of episodes to evaluate the agent on"""
    num_eval_envs: int = 10
    """the number of parallel environments to evaluate the agent on"""
    sim_backend: str = "cpu"
    """the simulation backend to use for evaluation environments. can be "cpu" or "gpu"""
    control_mode: str = "pd_joint_delta_pos"
    """the control mode to use for the evaluation environments. Must match the control mode of the demonstration dataset."""

    mask_background: bool = False


def load_h5_data(data):
    out = dict()
    for k in data.keys():
        if isinstance(data[k], h5py.Dataset):
            out[k] = data[k][:]
        else:
            out[k] = load_h5_data(data[k])
    return out


def make_mlp(in_channels, mlp_channels, act_builder=nn.ReLU, last_act=True):
    c_in = in_channels
    module_list = []
    for idx, c_out in enumerate(mlp_channels):
        module_list.append(nn.Linear(c_in, c_out))
        if last_act or idx < len(mlp_channels) - 1:
            module_list.append(act_builder())
        c_in = c_out
    return nn.Sequential(*module_list)


def flatten_state_dict_with_space(state_dict: dict) -> np.ndarray:
    states = []
    for key in state_dict.keys():
        value = state_dict[key]
        if isinstance(value, (tuple, list)):
            state = None if len(value) == 0 else value
        elif isinstance(value, (bool, np.bool_, int, np.int32, np.int64)):
            # x = np.array(1) > 0 is np.bool_ instead of ndarray
            state = int(value)
        elif isinstance(value, (float, np.float32, np.float64)):
            state = np.float32(value)
        elif isinstance(value, np.ndarray) or isinstance(value, torch.Tensor):
            if value.ndim > 2:
                raise AssertionError(
                    "The dimension of {} should not be more than 2.".format(key)
                )
            state = value
        else:
            raise TypeError("Unsupported type: {}".format(type(value)))
        if state is not None:
            states.append(state)
    if len(states) == 0:
        return np.empty(0)
    else:
        if isinstance(states[0], torch.Tensor):
            try:
                return torch.hstack(states)
            except:
                return torch.column_stack(states)
        else:
            try:
                return np.hstack(states)
            except:  # dirty fix for concat trajectory of states
                return np.column_stack(states)

# taken from here
# https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/Segmentation/MaskRCNN/pytorch/maskrcnn_benchmark/data/samplers/iteration_based_batch_sampler.py
class IterationBasedBatchSampler(BatchSampler):
    """
    Wraps a BatchSampler, resampling from it until
    a specified number of iterations have been sampled
    """

    def __init__(self, batch_sampler, num_iterations, start_iter=0):
        self.batch_sampler = batch_sampler
        self.num_iterations = num_iterations
        self.start_iter = start_iter

    def __iter__(self):
        iteration = self.start_iter
        while iteration <= self.num_iterations:
            # if the underlying sampler has a set_epoch method, like
            # DistributedSampler, used for making each process see
            # a different split of the dataset, then set it
            if hasattr(self.batch_sampler.sampler, "set_epoch"):
                self.batch_sampler.sampler.set_epoch(iteration)
            for batch in self.batch_sampler:
                iteration += 1
                if iteration > self.num_iterations:
                    break
                yield batch

    def __len__(self):
        return self.num_iterations


class PlainConv(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_dim=256,
        max_pooling=True,
        inactivated_output=False,  # False for ConvBody, True for CNN
    ):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # [64, 64]
            nn.Conv2d(16, 16, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # [32, 32]
            nn.Conv2d(16, 32, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # [16, 16]
            nn.Conv2d(32, 64, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # [8, 8]
            nn.Conv2d(64, 128, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # [4, 4]
            nn.Conv2d(128, 128, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
        )

        if max_pooling:
            self.pool = nn.AdaptiveMaxPool2d((1, 1))
            self.fc = make_mlp(128, [out_dim], last_act=not inactivated_output)
        else:
            self.pool = None
            self.fc = make_mlp(128 * 4 * 4, [out_dim], last_act=not inactivated_output)

        self.reset_parameters()

    def reset_parameters(self):
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, image):
        x = self.cnn(image)
        if self.pool is not None:
            x = self.pool(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x


class Actor(nn.Module):
    def __init__(self, action_dim): # state_dim,
        super().__init__()
        self.encoder = PlainConv(
            in_channels=3, out_dim=256, max_pooling=False, inactivated_output=False
        )
        self.final_mlp = make_mlp(
            256, [512, 256, action_dim], last_act=False #  + state_dim
        )
        self.get_eval_action = self.get_action = self.forward

    def forward(self, rgb, state=None):
        img = rgb.permute(0, 3, 1, 2)  # (B, C, H, W)
        feature = self.encoder(img)
        # x = torch.cat([feature, state], dim=1)
        x = feature
        return self.final_mlp(x)


def load_ckpt(run_name, tag):
    print("Loading actor from", f"runs/{run_name}/checkpoints/{tag}.pt")
    actor.load_state_dict(torch.load(f"runs/{run_name}/checkpoints/{tag}.pt")["actor"])


if __name__ == "__main__":
    args = tyro.cli(Args)

    if args.exp_name is None:
        args.exp_name = os.path.basename(__file__)[: -len(".py")]
        run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    else:
        run_name = args.exp_name

    if args.demo_path.endswith(".h5"):
        import json

        json_file = args.demo_path[:-2] + "json"
        with open(json_file, "r") as f:
            demo_info = json.load(f)
            if "control_mode" in demo_info["env_info"]["env_kwargs"]:
                control_mode = demo_info["env_info"]["env_kwargs"]["control_mode"]
            elif "control_mode" in demo_info["episodes"][0]:
                control_mode = demo_info["episodes"][0]["control_mode"]
            else:
                raise Exception("Control mode not found in json")
            assert (
                control_mode == args.control_mode
            ), f"Control mode mismatched. Dataset has control mode {control_mode}, but args has control mode {args.control_mode}"

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(args.torch_deterministic)

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    env_kwargs = dict(
        control_mode=args.control_mode,
        reward_mode="dense",
        obs_mode="rgb+segmentation",
        render_mode="rgb_array",
    )
    if args.max_episode_steps is not None:
        env_kwargs["max_episode_steps"] = args.max_episode_steps


    def helper(wrpr, new_size, mean, std):
        def test(env):
            return wrpr(env, new_size=new_size, mean=mean, std=std)
        return test


    ds = torch.load(f"runs/{run_name}/dataset.pt")
    print(ds["actions_mean"].shape) # ds["states_mean"].shape,

    rgb_mean = torch.from_numpy(ds["rgb_mean"]).float()
    rgb_std = torch.from_numpy(ds["rgb_std"]).float()

    wrapper = MaskResizeRGBSegObsWrapper if args.mask_background else ResizeRGBSegObservationWrapper
    wrapper = helper(wrapper, new_size=(128, 128), mean=rgb_mean, std=rgb_std)
    envs = make_eval_envs(
        args.env_id,
        args.num_eval_envs,
        args.sim_backend,
        env_kwargs,
        video_dir=f"runs/{run_name}/videos" if args.capture_video else None,
        wrappers=[wrapper],
    )


    actor = Actor(ds["actions_mean"].shape[0]).to( # ds["states_mean"].shape[0],
        device=device
    )
    # actor = torch.compile(actor)
    load_ckpt(run_name=run_name, tag=199999)

    # state_mean = torch.from_numpy(ds["states_mean"]).float().to(device)
    # state_std = torch.from_numpy(ds["states_std"]).float().to(device)

    action_mean = torch.from_numpy(ds["actions_mean"]).float().to(device)
    action_std = torch.from_numpy(ds["actions_std"]).float().to(device)


    # norm_tensor = torch.Tensor([255.0, 255.0, 255.0]).float().to(device)

    def sample_fn(obs):
        rgb = obs["sensor_data"]["3rd_view_camera"]["rgb"]
        # agent = obs["agent"]
        # extra = obs["extra"]

        if isinstance(rgb, np.ndarray):
            rgb = torch.from_numpy(rgb).float().to(device)
            # agent = torch.from_numpy(agent).float().to(device)
            # extra = torch.from_numpy(extra).float().to(device)

        # rgb_t = torch.div(rgb, norm_tensor)

        # state = torch.hstack(
        #     [
        #         flatten_state_dict_with_space(agent),
        #         flatten_state_dict_with_space(extra),
        #     ]
        # )
        # rgb_norm = (rgb_t - rgb_mean) / rgb_std
        # state_norm = (state - state_mean) / state_std
        action_norm = actor(rgb) # , state_norm
        action_denorm = action_norm * action_std + action_mean
        if args.sim_backend == "cpu":
            action_denorm = action_denorm.cpu().numpy()
        return action_denorm


    writer = SummaryWriter(f"runs/{run_name}/eval")

    if args.track:
        import wandb

        config = vars(args)
        config["eval_env_cfg"] = dict(
            **env_kwargs,
            num_envs=args.num_eval_envs,
            env_id=args.env_id,
            env_horizon=gym_utils.find_max_episode_steps_value(envs),
        )
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=config,
            name=run_name,
            save_code=True,
            group="BehaviorCloning",
            tags=["behavior_cloning"],
        )

    actor.eval()
    evals = []
    for seed in list(range(args.num_eval_episodes)):

        obs, _ = envs.reset(seed=seed)
        with torch.no_grad():
            eval_metrics = evaluate(1, sample_fn, envs)


        print(f"Evaluated {len(eval_metrics['success_at_end'])} episodes")
        for k in eval_metrics.keys():
            eval_metrics[k] = np.mean(eval_metrics[k])
            writer.add_scalar(f"eval/{k}", eval_metrics[k], seed)
            print(f"{seed} {k}: {eval_metrics[k]:.4f}")

    envs.close()
    if args.track:
        wandb.finish()
