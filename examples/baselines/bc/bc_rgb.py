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

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.io_utils import load_json
from mani_skill.utils.gym_utils import inv_scale_action

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
    wandb_project_name: str = "put_carrot_on_plate"
    """the wandb's project name"""
    wandb_entity: Optional[str] = "jakd9"
    """the entity (team) of wandb's project"""
    wandb_group: str = "BC"
    """the group of the run for wandb"""

    capture_video: bool = True
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    env_id: str = "PegInsertionSide-v0"
    """the id of the environment"""
    demo_path: str = "data/ms2_official_demos/rigid_body/PegInsertionSide-v0/trajectory.state.pd_ee_delta_pose.h5"
    """the path of demo dataset (pkl or h5)"""
    num_demos: Optional[int] = None
    """number of trajectories to load from the demo dataset"""
    total_iters: int = 1_000_000
    """total timesteps of the experiment"""
    batch_size: int = 1024
    """the batch size of sample from the replay memory"""

    # Behavior cloning specific arguments
    normalize: bool = False
    """if toggled, states are normalized to mean 0 and standard deviation 1"""
    lr: float = 3e-4
    """the learning rate for the actor"""

    # Environment/experiment specific arguments
    max_episode_steps: Optional[int] = None
    """Change the environments' max_episode_steps to this value. Sometimes necessary if the demonstrations being imitated are too short. Typically the default
    max episode steps of environments in ManiSkill are tuned lower so reinforcement learning agents can learn faster."""
    log_freq: int = 1000
    """the frequency of logging the training metrics"""
    save_freq: Optional[int] = None
    """the frequency of saving the model checkpoints. By default this is None and will only save checkpoints based on the best evaluation metrics."""
    sim_backend: str = "cpu"
    """the simulation backend to use for evaluation environments. can be "cpu" or "gpu"""
    num_dataload_workers: int = 0
    """the number of workers to use for loading the training data in the torch dataloader"""
    control_mode: str = "pd_joint_delta_pos"
    """the control mode to use for the evaluation environments. Must match the control mode of the demonstration dataset."""

    # additional tags/configs for logging purposes to wandb and shared comparisons with other algorithms
    demo_type: Optional[str] = None
    shader: str = "default"

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


class ManiSkillDataset(Dataset):
    def __init__(self, dataset_file: str, device: torch.device, load_count, mask_background, not_masked_ids, normalize=False) -> None:
        self.dataset_file = dataset_file
        # for details on how the code below works, see the
        # quick start tutorial
        self.data = h5py.File(dataset_file, "r")
        json_path = dataset_file.replace(".h5", ".json")
        self.json_data = load_json(json_path)
        self.episodes = self.json_data["episodes"]

        self.env_info = self.json_data["env_info"]
        self.env_id = self.env_info["env_id"]
        self.env_kwargs = self.env_info["env_kwargs"]
        self.mask_background = mask_background
        self.normalize = normalize

        self.rgb = []
        self.mask = []
        self.actions = []
        self.dones = []
        self.states = []
        self.total_frames = 0
        self.device = device

        if load_count is None:
            load_count = len(self.episodes)

        for eps_id in tqdm(range(load_count)):
            eps = self.episodes[eps_id]
            trajectory = self.data[f"traj_{eps['episode_id']}"]
            trajectory = load_h5_data(trajectory)
            # agent = trajectory["obs"]["agent"]
            # extra = trajectory["obs"]["extra"]

            # state = np.hstack(
            #     [
            #         flatten_state_dict_with_space(agent),
            #         flatten_state_dict_with_space(extra),
            #     ]
            # )
            # # print(state.dtype)
            # self.states.append(state)

            # we use :-1 here to ignore the last observation as that
            # is the terminal observation which has no actions
            rgb_images = trajectory["obs"]["sensor_data"]["base_camera"]["rgb"][:-1]

            # import matplotlib.pyplot as plt

            # plt.imshow(rgb_images[0])
            # plt.show()

            # exit(0)

            if args.control_mode == "pd_joint_delta_pos":
                self.rgb.append(rgb_images[1:])

                # delta_arm_pos = trajectory["obs"]["agent"]["qpos"][1:, :7] - trajectory["obs"]["agent"]["qpos"][:-1, :7]
                # delta_arm_pos = inv_scale_action(delta_arm_pos, arm_controller.config.lower, arm_controller.config.upper)
                delta_arm_pos = inv_scale_action(trajectory["actions"][1:, :7]- trajectory["actions"][:-1, :7], arm_controller.config.lower, arm_controller.config.upper)
                actions = np.concatenate([delta_arm_pos, trajectory["actions"][1:,-2:-1]], axis=-1)
            else:
                self.rgb.append(rgb_images)

                actions = trajectory["actions"]

            self.actions.append(actions)
            # import pdb; pdb.set_trace()
            # print()

            # print(rgb_images.dtype)
            if self.mask_background:
                segment_images = np.int16(trajectory["obs"]["sensor_data"]["base_camera"]["segmentation"][:-1])
                mask = np.ones_like(segment_images)
                mask[
                    np.isin(
                        segment_images,
                        not_masked_ids,
                    )
                ] = 0
                self.mask.append(mask)


        self.rgb = np.vstack(self.rgb)
        if self.mask_background:
            self.mask = np.vstack(self.mask)
        # self.states = np.vstack(self.states)
        self.actions = np.vstack(self.actions)
        self.actions_mean = np.mean(self.actions, axis=0, dtype=np.float32)
        self.actions_std = (np.std(self.actions, axis=0, dtype=np.float32) + 0.001) 

        
        # self.states_mean = np.mean(self.states, axis=0, dtype=np.float32)
        # self.states_std = np.std(self.states, axis=0, dtype=np.float32)

        assert self.rgb.shape[0] == self.actions.shape[0]

    def __len__(self):
        return len(self.rgb)

    def __getitem__(self, idx):
        out = {}
        actions = torch.from_numpy(self.actions[idx])
        rgb = torch.from_numpy(self.rgb[idx])
        # state = torch.from_numpy(self.states[idx])

        if self.normalize:
            actions = (actions - self.actions_mean) / self.actions_std
            rgb = rgb / 255.0
            rgb = 2 * rgb - 1
            # state = (state - self.states_mean) / self.states_std

        if self.mask_background:
            mask = torch.from_numpy(self.mask[idx])
            zero_img = torch.zeros_like(rgb, dtype=rgb.dtype)
            rgb = rgb * (1 - mask) + zero_img * mask

        out["rgb"] = rgb.float().to(device=self.device)
        out["action"] = actions.float().to(device=self.device)
    
        # out["state"] = state.float().to(device=self.device)

        return out


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
            256, [512, 256, action_dim], last_act=False # state_dim
        )
        self.get_eval_action = self.get_action = self.forward

    def forward(self, rgb, state=None):
        img = rgb.permute(0, 3, 1, 2)  # (B, C, H, W)
        feature = self.encoder(img)
        # x = torch.cat([feature, state], dim=1)
        x = feature
        return self.final_mlp(x)


def save_ckpt(run_name, tag):
    os.makedirs(f"runs/{run_name}/checkpoints", exist_ok=True)
    torch.save(
        {
            "actor": actor.state_dict(),
        },
        f"runs/{run_name}/checkpoints/{tag}.pt",
    )


if __name__ == "__main__":
    args = tyro.cli(Args)

    if args.exp_name is None:
        args.exp_name = os.path.basename(__file__)[: -len(".py")]
        run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    else:
        run_name = args.exp_name

    # if args.demo_path.endswith(".h5"):
    #     import json

    #     json_file = args.demo_path[:-2] + "json"
    #     with open(json_file, "r") as f:
    #         demo_info = json.load(f)
    #         if "control_mode" in demo_info["env_info"]["env_kwargs"]:
    #             control_mode = demo_info["env_info"]["env_kwargs"]["control_mode"]
    #         elif "control_mode" in demo_info["episodes"][0]:
    #             control_mode = demo_info["episodes"][0]["control_mode"]
    #         else:
    #             raise Exception("Control mode not found in json")
    #         assert (
    #             control_mode == args.control_mode
    #         ), f"Control mode mismatched. Dataset has control mode {control_mode}, but args has control mode {args.control_mode}"

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(args.torch_deterministic)

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    env: gym.Env = gym.make(
    args.env_id,
    obs_mode="rgb+segmentation",
    reward_mode="dense",
    control_mode=args.control_mode,
    render_mode="rgb_array",
    sensor_configs=dict(shader_pack=args.shader),
    human_render_camera_configs=dict(shader_pack=args.shader),
    viewer_camera_configs=dict(shader_pack=args.shader),
    num_envs=1, # if num_envs > 1, GPU simulation backend is used.
    )
    arm_controller = env.agent.controller.controllers["arm"]        
    obs, _ = env.reset()
    env = env.unwrapped

    robot_link_ids = [x._objs[0].entity.per_scene_id for x in env.agent.robot.get_links()]

    actor_ids = [
        x._objs[0].per_scene_id
        for x in env.scene.actors.values()
        if x.name not in ["ground"] ## This is environment spesific
    ]

    object_ids = torch.tensor(
                robot_link_ids + actor_ids,
                dtype=torch.int16,
                device=env.device,
            )

    print("Backgroud masking:", args.mask_background)
    ds = ManiSkillDataset(
        args.demo_path,
        device=device,
        load_count=args.num_demos,
        mask_background=args.mask_background,
        not_masked_ids=object_ids,
        normalize=args.normalize,
    )
    # exit(0)
    # import pdb; pdb.set_trace()
    # tr_ds, te_ds = torch.utils.data.random_split(ds, [0.99, 0.01])


    sampler = RandomSampler(ds)
    batch_sampler = BatchSampler(sampler, args.batch_size, drop_last=True)
    iter_sampler = IterationBasedBatchSampler(batch_sampler, args.total_iters)

    data_loader = DataLoader(ds, batch_sampler=iter_sampler, num_workers=0)
    # test_loader = DataLoader(ds, batch_size=1)
    actor = Actor(action_dim=ds.actions.shape[1]).to( # ds.states.shape[1], 
        device=device
    )

    optimizer = optim.AdamW(actor.parameters(), lr=args.lr)

    if args.track:
        import wandb

        config = vars(args)

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=config,
            name=run_name,
            save_code=True,
            group=args.wandb_group,
            tags=["behavior_cloning"],
        )

    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    torch.save({
                # "rgb_mean": ds.rgb_mean, "rgb_std": ds.rgb_std, 
                "actions_mean": ds.actions_mean, "actions_std": ds.actions_std,
                # "states_mean": self.states_mean, "states_std": self.states_std 
                },
                f"runs/{run_name}/dataset.pt",                   
    )

    for iteration, batch in enumerate(data_loader):
        log_dict = {}

        optimizer.zero_grad()
        preds = actor(batch["rgb"]) # , batch["state"]
        loss = F.mse_loss(preds, batch["action"])
        loss.backward()
        optimizer.step()

        if (iteration+1) % args.log_freq == 0:
            # test_loss_list = []
            # with torch.no_grad():
            #     actor.eval()
            #     for test_batch in test_loader:
            #         test_pred = actor(test_batch["rgb"])
            #         test_loss = F.mse_loss(test_pred, test_batch["action"])
            #         test_loss_list.append(test_loss.item())
            #     mean_test_loss = torch.stack(test_loss_list).mean()

            print(f"Iteration {iteration}, loss: {loss.item()}") # , mean test loss: {mean_test_loss}
            writer.add_scalar(
                "charts/learning_rate", optimizer.param_groups[0]["lr"], iteration
            )
            writer.add_scalar("charts/train_loss", loss.item(), iteration)
            # writer.add_scalar("charts/mean_test_loss", loss.item(), iteration)

        if args.save_freq is not None and (iteration+1) % args.save_freq == 0:
            save_ckpt(run_name, str(iteration))

    if args.track:
        wandb.finish()
