import argparse
import muse.envs
import torch
import gym
import json

import torch.multiprocessing as mp
import robolearn.utils.torch as ptu
import skvideo.io
import numpy as np

from einops import rearrange
from pathlib import Path
from tqdm import tqdm
from collections import deque

from robolearn.model.factory import load_bc_model
from robolearn.collect.utils import compute_workers_seed
from robolearn.utils.attention import AttentionHook
from robolearn.data.utils import state_to_frames
from robolearn.data.utils import (
    filter_state,
    process_state,
    flatten_normalize_dict,
)
from muse.core.constants import REALSENSE_CROP


def get_args_parser():
    parser = argparse.ArgumentParser("BC evaluation script", add_help=False)
    # Model
    parser.add_argument("--checkpoint", default="")
    # Environment
    parser.add_argument("--env-name", default="Pick-v0", type=str)
    parser.add_argument("--seed", default=5000, type=int, help="Initial seed")
    parser.add_argument("--episodes", default=20, type=int)
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--att-maps", action="store_true")
    parser.add_argument("--cam-list", default="", type=str)
    parser.add_argument("--num-workers", default=5, type=int, help="Number of workers")
    parser.add_argument("--seed-setup-path", default="", type=str)
    parser.set_defaults(record=False, att_map=False)
    return parser


def evaluate_seeds(
    model,
    env,
    seeds,
    cam_list,
    frame_hist,
    transform,
    output_path,
    seed_setup_path,
    record=False,
    att_maps=False,
    pbar=None,
):

    stats = {"successful_seeds": [], "failure_seeds": []}
    num_frames = len(frame_hist) + 1
    history_idx_list = [0] + frame_hist
    max_frame_idx = max(history_idx_list)

    # Run policy
    for seed in range(*seeds):
        if record:
            video_dir = output_path / "evaluation"
            video_dir.mkdir(parents=True, exist_ok=True)
            video_writer = skvideo.io.FFmpegWriter(str(video_dir / f"{seed:08d}.mp4"))
            if att_maps:
                att_hook = AttentionHook(model)

        env.seed(seed)

        if seed_setup_path:
            # TODO: Override number of seeds with seed files
            with open(str(Path(seed_setup_path) / f"{seed:8d}.pkl"), "rb") as f:
                seed_setup = pkl.load(f)
            obs = env.reset(**seed_setup)
        else:
            obs = env.reset()

        W, H = REALSENSE_CROP
        num_streams = len(cam_list)

        frames = torch.zeros(
            (num_streams, max_frame_idx + 1, H, W, 3),
            dtype=torch.float32,
            device=ptu.device,
        )
        # FIXME: update for robot optimality
        states = torch.zeros(
            (max_frame_idx + 1, model.state_dim),
            dtype=torch.float32,
            device=ptu.device,
        )
        for i in range(env.spec.max_episode_steps):
            # Pre-process input
            step_frames, step_masks = state_to_frames(obs, cam_list)
            if record:
                record_frame = rearrange(step_frames, "v h w c -> h (v w) c").numpy()
                if att_maps and i > 0:
                    record_frame = att_hook.blend_map(record_frame)
                video_writer.writeFrame(record_frame)
            else:
                env.render()

            # shift frames to the left
            frames[:, :-1] = frames[:, 1:].clone()
            for j, cam_name in enumerate(cam_list):
                crop = obs[f"rgb_{cam_name}"].copy()
                frames[j, -1, :, :, :] = torch.tensor(crop)
            # if t=0, all the stack of frames is equal to the first frame
            if i == 0:
                frames[:, :-1] = frames[:, -1].unsqueeze(1)

            frames_select_idx = [max_frame_idx - i for i in reversed(history_idx_list)]

            frames_norm = frames[:, frames_select_idx].unsqueeze(0) / 255
            frames_norm = transform(frames_norm)

            # process states
            state = filter_state(obs, select_keys=["gripper_pos", "gripper_theta"])
            state = process_state(state)
            state = flatten_normalize_dict(state, model.data_stats["traj_stats"])
            # shift states to the left
            states[:-1] = states[1:].clone()
            states[-1] = state

            if i == 0:
                states[:] = states[-1].unsqueeze(0)

            current_state = states[frames_select_idx].unsqueeze(0)

            # Compute action
            with torch.no_grad():
                action = model(frames_norm, current_state)

            # Post-process action
            action = model.process_output(action)
            for k, v in action.items():
                if type(v) is not np.ndarray:
                    action[k] = v.cpu().detach().numpy().squeeze(0)

            obs, reward, done, info = env.step(action)
            if done:
                break

        if info["success"]:
            stats["successful_seeds"].append(seed)
        else:
            stats["failure_seeds"].append(seed)

        if record:
            video_writer.close()

        if pbar is not None:
            pbar.update()

    if pbar is not None:
        pbar.close()
    return stats


def worker_evaluate_seeds(
    worker_id,
    env_name,
    checkpoint,
    seeds,
    cam_list,
    data_queue,
    seed_setup_path,
    record=False,
    att_maps=False,
):
    print(f"Worker {worker_id} starting...")

    torch.set_num_threads(1)
    ptu.set_gpu_mode(True)
    env = gym.make(
        env_name,
    )

    pbar = None
    if worker_id == 0:
        pbar = tqdm(total=seeds[1] - seeds[0], ncols=80)

    model, model_args = load_bc_model(checkpoint, cam_list)
    model.eval()
    model.to(ptu.device)
    frame_hist = model_args.frame_hist
    cam_list = model_args.cam_list

    # Define normalization
    transform = model_args.im_transform
    output_path = Path(checkpoint).parent

    stats = evaluate_seeds(
        model,
        env,
        seeds,
        cam_list,
        frame_hist,
        transform,
        output_path,
        seed_setup_path,
        record=record,
        att_maps=att_maps,
        pbar=pbar,
    )

    data_queue.put((worker_id, stats))
    del env
    print(f"Worker {worker_id} finished")

    return worker_id, stats


def main(args):
    num_workers = args.num_workers
    seed_setup_path = args.seed_setup_path
    initial_seed = args.seed
    episodes = args.episodes

    if num_workers > episodes:
        num_workers = episodes

    seeds_worker = compute_workers_seed(episodes, num_workers, initial_seed)

    if args.cam_list != "":
        args.cam_list = args.cam_list.split(",")

    workers = []
    data_queue = mp.Queue()
    if num_workers == 0:
        worker_id, stats = worker_evaluate_seeds(
            0,
            args.env_name,
            args.checkpoint,
            seeds_worker[0],
            args.cam_list,
            data_queue,
            seed_setup_path,
            record=args.record,
            att_maps=args.att_maps,
        )

    else:
        for i in range(num_workers):
            w = mp.Process(
                target=worker_evaluate_seeds,
                args=(
                    i,
                    args.env_name,
                    args.checkpoint,
                    seeds_worker[i],
                    args.cam_list,
                    data_queue,
                    seed_setup_path,
                    args.record,
                    args.att_maps,
                ),
            )
            w.daemon = True
            w.start()
            workers.append(w)
        counter = 0
        data = []
        stats = dict()
        while counter < num_workers:
            i, worker_stats = data_queue.get()

            for k, v in worker_stats.items():
                if k not in stats:
                    stats[k] = list()
                stats[k] += v

            print(f"Worker {i} data received")
            counter += 1

    stats["success_rate"] = len(stats["successful_seeds"]) / episodes

    with open(str(Path(args.checkpoint).parent / "evaluation.json"), "w") as f:
        json.dump(stats, f)
    print(stats)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "BC evaluation script", parents=[get_args_parser()]
    )
    args = parser.parse_args()
    if not Path(args.checkpoint).exists():
        raise ValueError(f"Checkpoint directory {args.checkpoint} does not exist.")
    main(args)
