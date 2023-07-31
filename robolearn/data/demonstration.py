import torch
from torch.utils.data import Dataset

import pickle as pkl
import numpy as np
from collections import deque
from pathlib import Path

from robolearn.data.utils import (
    flatten_normalize_dict,
    normalize_frames,
    normalize_states,
    state_to_frames,
    filter_state,
    denormalize_action,
)


class DemonstrationDataset(Dataset):
    def __init__(
        self,
        path,
        split,
        cam_list,
        frame_hist,
        delay_hist,
        state_keys=["gripper_pos", "gripper_theta"],
        task="diffusion",
    ):
        self.root = Path(path)
        if not self.root.exists():
            raise ValueError(f"Dataset path {self.root} doesn't exists")

        self.split = split

        self.frame_hist = frame_hist
        self.delay_hist = delay_hist
        self.num_frames = len(self.frame_hist) + 1
        assert self.num_frames > 0

        self.state_keys = state_keys

        self.task = task

        with open(str(self.root / "stats.pkl"), "rb") as f:
            self.stats = pkl.load(f)
        self.traj_stats = self.stats["traj_stats"]
        theta_stats = self.traj_stats["gripper_theta"]
        self.traj_stats["gripper_theta"] = dict(
            mean=np.array([np.sin(theta_stats["mean"]), np.cos(theta_stats["mean"])]),
            std=np.array([np.sin(theta_stats["std"]), np.cos(theta_stats["std"])]),
        )

        self.action_space = self.stats["action_space"]

        self.step_files = sorted(self.root.glob("*_*.pkl"))
        self.cam_list = cam_list or self.stats["cam_list"]
        self.num_streams = len(self.cam_list)

    def __len__(self):
        return self.stats["dataset_size"]

    def sample_hist(self):
        if self.delay_hist == 0:
            history_idx_list = self.frame_hist
        else:
            # sample indices in range (1, delay_hist+1)
            history_idx_list = sorted(
                np.random.randint(1, self.delay_hist + 1, size=len(self.frame_hist))
            )
        # add present frame history idx list
        history_idx_list = [0] + history_idx_list
        return history_idx_list

    # def sample_hist(self):
    #     history_idx_list = self.frame_hist
    #     # add present frame history idx list
    #     history_idx_list = [0] + history_idx_list

    #     # Randomly delay frames while maintaining time consistency
    #     last_idx = 0
    #     delayed_history_idx_list = []
    #     for idx in history_idx_list:
    #         if idx < last_idx:
    #             idx = last_idx
    #         last_idx = idx + np.random.randint(self.delay_hist + 1)
    #         delayed_history_idx_list.append(last_idx)

    #     return delayed_history_idx_list

    def __getitem__(self, idx):
        step_file = self.step_files[idx]
        step_info = str(step_file.stem).split("_")
        seed = int(step_info[0])
        step_idx = int(step_info[1])

        frames = deque(maxlen=self.num_frames)
        masks = deque(maxlen=self.num_frames)
        states = deque(maxlen=self.num_frames)

        history_idx_list = self.sample_hist()
        for history_idx in history_idx_list:
            frame_step_idx = step_idx - history_idx
            if frame_step_idx < 0:
                break
            frame_name = self.root / f"{seed:08d}_{frame_step_idx:05d}.pkl"
            with open(frame_name, "rb") as f:
                step = pkl.load(f)
            step_state, _ = step
            step_frames, step_masks = state_to_frames(step_state, self.cam_list)
            frames.appendleft(step_frames)
            masks.appendleft(step_masks)

            state = filter_state(step_state, select_keys=self.state_keys)
            states.appendleft(state)

            if history_idx == 0:
                present_step = step

        frames, masks = normalize_frames(frames, masks, self.num_frames)
        states = normalize_states(states, self.traj_stats, self.num_frames)

        _, action = present_step

        # only keep action space keys and normalize real-valued action
        action_space_keys = self.action_space.keys()
        action_keys = list(action.keys())

        # FIXME: Adapt normalization to work both for BC and diffusion
        action_norm = {}
        if "grip_open" in self.action_space.keys():
            # process tool action
            grip_open = float(action["grip_open"])
            if self.task == "diffusion":
                grip_binary = (
                    torch.tensor(1.0) if grip_open >= 0 else torch.tensor(-1.0)
                )
            else:
                grip_binary = torch.tensor(1.0) if grip_open >= 0 else torch.tensor(0.0)
            action_norm["grip"] = grip_binary.unsqueeze(-1)

        for k in action_keys:
            if k not in action_space_keys or "grip" in k:
                action.pop(k)
        vel = flatten_normalize_dict(action, self.traj_stats)

        action_norm["vel"] = vel

        return frames, masks, states, action_norm

    def denormalize_target(self, target):
        denorm_target = denormalize_action(target, self.stats, task=self.task)
        return denorm_target
