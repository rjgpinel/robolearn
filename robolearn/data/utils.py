import numpy as np
import re
import torch

from copy import deepcopy

EPS = 1e-3
RESNET_NORM_STATS = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
IMG_VIT_NORM_STATS = dict(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])


def get_im_norm_stats(normalization_name):
    if normalization_name == "resnet":
        norm_img = RESNET_NORM_STATS
    elif normalization_name == "vit":
        return IMG_VIT_NORM_STATS
    elif normalization_name == "":
        return dict(mean=np.zeros(3), std=np.ones(3))
    else:
        raise ValueError(f"Normalization name {normalization_name} is not a valid one.")

    return norm_img


def normalize(x, stats):
    device = x.device
    mean, std = (
        torch.tensor(stats["mean"]).to(device),
        torch.tensor(stats["std"]).to(device),
    )
    return ((x - mean) / (std + EPS)).float()


def denormalize(x, stats):
    device = x.device
    mean, std = (
        torch.tensor(stats["mean"]).to(device),
        torch.tensor(stats["std"]).to(device),
    )
    return ((x * (std)) + mean).float()


def state_to_frames(state, cam_list):
    # extract camera views from state
    step_frames = []
    step_masks = []
    for cam_name in cam_list:
        frame = torch.tensor(state[f"rgb_{cam_name}"].copy())
        if f"seg_{cam_name}" in state:
            mask = torch.tensor(state[f"seg_{cam_name}"])
        else:
            mask = torch.zeros(frame.shape[:-1])
        step_frames.append(frame)
        step_masks.append(mask)
    step_frames = torch.stack(step_frames)
    step_masks = torch.stack(step_masks)
    return step_frames, step_masks


def normalize_frames(frames, masks, num_frames):
    frames = deepcopy(frames)
    masks = deepcopy(masks)
    # complete frame stack and scale values to [0, 1]
    missing_frames = num_frames - len(frames)
    assert missing_frames >= 0
    if missing_frames > 0:
        oldest_frame = deepcopy(frames[0])
        oldest_mask = deepcopy(masks[0])
        for _ in range(missing_frames):
            frames.appendleft(oldest_frame)
            masks.appendleft(oldest_mask)
    frames = torch.stack(list(frames), dim=1)
    masks = torch.stack(list(masks), dim=1)
    frames = frames.float() / 255
    return frames, masks


def denormalize_frame(x, stats):
    mean = torch.tensor(stats["mean"])
    std = torch.tensor(stats["std"])
    for i in range(len(mean)):
        x[:, i, :, :] = x[:, i, :, :] * std[i] + mean[i]
    return x


def denormalize_action(action, stats, task="diffusion"):
    vel = action["vel"]
    denorm_action = {}

    action_space = stats["action_space"]
    traj_stats = stats["traj_stats"]

    action_offset = 0
    for k in action_space.keys():
        if "grip" not in k:
            action_size = action_space[k].shape[0]
            denorm_action[k] = denormalize(
                vel[:, action_offset : action_offset + action_size],
                traj_stats[k],
            )
            action_offset += action_size

    # FIXME: Adapt normalization to work both for BC and diffusion
    if "grip" in action.keys():
        grip = action["grip"]

        denorm_action["grip_open"] = torch.zeros(grip.shape).to(grip.device)

        if task == "diffusion":
            grip_threshold = 0.0
        else:
            grip_threshold = 0.5

        denorm_action["grip_open"][grip > grip_threshold] = torch.tensor(
            action_space["grip_open"].high
        )
        denorm_action["grip_open"][grip <= grip_threshold] = torch.tensor(
            action_space["grip_open"].low
        )

    return denorm_action


def flatten_normalize_dict(d, stats):
    # flatten with keys sorted in alphabetical order
    keys = sorted(d.keys())
    flat = []
    for k in keys:
        if k in stats.keys():
            x = normalize(torch.tensor(d[k]), stats[k])
            # if scalar add a dim for concatenation
            if not x.shape:
                x = x[None]
            flat.append(x)
    flat = torch.cat(flat)
    return flat


def process_state(state):
    if "gripper_theta" in state.keys():
        gripper_theta = state.pop("gripper_theta")
        state["gripper_theta"] = [np.sin(gripper_theta), np.cos(gripper_theta)]
    return state


def normalize_states(states, stats, hist_len):
    for i in range(len(states)):
        state = states[i]
        process_state(state)
        states[i] = flatten_normalize_dict(state, stats)

    missing_states = hist_len - len(states)
    assert missing_states >= 0
    if missing_states > 0:
        oldest_state = deepcopy(states[0])
        for _ in range(missing_states):
            states.appendleft(oldest_state)
    states = torch.stack(list(states), dim=0)
    return states


def filter_state(state, select_keys=None):
    filtered_state = dict()
    # process tool state information
    state_keys = list(state.keys())
    # filter parts out of state
    filter_keywords = [
        "rgb",
        "seg",
        "depth",
        "arms_joint_name",
        "info",
        "gripper_state",
    ]
    pattern = "|".join(f"{k}" for k in filter_keywords)
    pattern = re.compile(pattern)

    for k in state_keys:
        if not pattern.match(k):
            if select_keys is None or k in select_keys:
                filtered_state[k] = state[k]
    return filtered_state


def ema(current, last, weight=0.0):
    if last is None:
        last = current
    return (1 - weight) * current + weight * last
