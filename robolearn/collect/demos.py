import argparse
import muse.envs
import gym
import sys
import numpy as np
import pickle as pkl
import torch.multiprocessing as mp
import torch

from copy import deepcopy
from tqdm import tqdm
from pathlib import Path

from robolearn.collect.utils import compute_workers_seed, process_mask
from robolearn.data.utils import filter_state


def get_args_parser():
    parser = argparse.ArgumentParser("BC data collector script", add_help=False)
    parser.add_argument(
        "--env-name",
        default="Pick-v0",
        type=str,
        help="Name of the environment",
    )
    parser.add_argument(
        "--episodes",
        default=1000,
        type=int,
        help="Number of expert demonstrations episodes to collect.",
    )
    parser.add_argument(
        "--num-textures", default=None, type=int, help="Number of textures to be used"
    )
    parser.add_argument("--num-workers", default=20, type=int, help="Number of workers")
    parser.add_argument("--seed", default=0, type=int, help="Initial seed")
    parser.add_argument(
        "--output-dir",
        default="",
        type=str,
        help="Output directory where the expert demonstrations will be saved",
    )
    parser.add_argument(
        "--num-distractors", default=0, type=int, help="Number of distractors"
    )
    parser.add_argument("--seed-setup-path", default="", type=str)
    parser.set_defaults(masks=False)
    return parser


def collect(worker_id, env_name, output_path, seeds, seed_setup_path, data_queue):
    torch.set_num_threads(1)
    env = gym.make(
        env_name,
        segmentation_masks=args.masks,
        num_textures=args.num_textures,
        num_distractors=args.num_distractors,
    )

    pbar = None
    if worker_id == 0:
        pbar = tqdm(total=seeds[1] - seeds[0], ncols=80)

    stats = {
        "num_steps": [],
        "successful_seeds": [],
        "failure_seeds": [],
        "action_space": env.unwrapped.action_space.spaces,
        "cam_list": env.unwrapped.cam_list,
        "actions": [],
        "obs": [],
    }

    for seed in range(*seeds):
        episode_traj = []
        env.seed(seed)
        if seed_setup_path:
            with open(str(Path(seed_setup_path) / f"{seed:8d}.pkl"), "rb") as f:
                seed_setup = pkl.load(f)
            obs = env.reset(**seed_setup)
        else:
            obs = env.reset()
        agent = env.unwrapped.oracle()

        for i in range(env.spec.max_episode_steps):
            action = agent.get_action(obs)
            if action is None:
                # If oracle is not able to solve the task
                info = {"success": False}
                stats["failure_seeds"].append(seed)
                break

            for cam_name in env.unwrapped.cam_list:
                if args.masks:
                    obs[f"seg_{cam_name}0"] = process_mask(env, obs[f"seg_{cam_name}0"])
            episode_traj.append((obs, action))
            obs, reward, done, info = env.step(action)
            if done:
                break

        if pbar is not None:
            pbar.update()

        # if trajectory is failed filter do not record in dataset
        if not info["success"]:
            continue

        for i, step in enumerate(episode_traj):
            with open(str(output_path / f"{seed:08d}_{i:05d}.pkl"), "wb") as f:
                pkl.dump(step, f)

        for obs, action in episode_traj:
            state = filter_state(obs)
            stats["obs"] += [state]
            stats["actions"] += [action]
        stats["successful_seeds"].append(seed)
        stats["num_steps"].append(len(episode_traj))

    if pbar is not None:
        pbar.close()

    data_queue.put((worker_id, stats))
    print(f"Worker {worker_id} finished")
    del env


def main(args):
    output_path = Path(args.output_dir)
    initial_seed = args.seed
    episodes = args.episodes
    num_workers = args.num_workers
    seed_setup_path = args.seed_setup_path

    if num_workers > episodes:
        num_workers = episodes

    seeds_worker = compute_workers_seed(episodes, num_workers, initial_seed)

    workers = []
    data_queue = mp.Queue()
    if num_workers == 0:
        collect(
            0,
            args.env_name,
            output_path,
            seeds_worker[0],
            seed_setup_path,
            data_queue,
        )

        i, collect_stats = data_queue.get()
    else:
        for i in range(num_workers):
            w = mp.Process(
                target=collect,
                args=(
                    i,
                    args.env_name,
                    output_path,
                    seeds_worker[i],
                    seed_setup_path,
                    data_queue,
                ),
            )
            w.daemon = True
            w.start()
            workers.append(w)
        counter = 0
        data = []
        collect_stats = None
        while counter < num_workers:
            i, worker_stats = data_queue.get()

            if collect_stats:
                for k, v in worker_stats.items():
                    if k not in ["cam_list", "action_space"]:
                        collect_stats[k] += v
            else:
                collect_stats = deepcopy(worker_stats)
            print(f"Worker {i} data received")
            counter += 1

    # Process Statistics
    total_num_steps = sum(collect_stats["num_steps"])
    num_success_traj = len(collect_stats["successful_seeds"])
    failure_seeds = collect_stats["failure_seeds"]

    print(f"[Data Collection] - Number of successful trajectories: {num_success_traj}")
    print(f"[Data Collection] - Number of steps: {total_num_steps}")
    print(f"[Data Collection] - Failure seeds: {failure_seeds}")

    data = {}
    print("[Data Statistics] - Computing dataset statistics")
    all_actions = collect_stats.pop("actions")
    for action in all_actions:
        for k, v in action.items():
            if k not in data:
                data[k] = []
            data[k].append(v)
    all_obs = collect_stats.pop("obs")
    for obs in all_obs:
        for k, v in obs.items():
            if k not in data:
                data[k] = []
            data[k].append(v)
    velocity_dim = 0

    for k, v in collect_stats["action_space"].items():
        if "grip" not in k:
            # FIXME: Update to use action space
            if "Push-v0" in args.env_name:
                velocity_dim += 2
            else:
                velocity_dim += v.shape[0]

    print(f"[Data Statistics] - Final dataset size {len(all_actions)}")
    stats = {
        "cam_list": collect_stats.pop("cam_list"),
        "action_space": collect_stats.pop("action_space"),
        "vel_dim": velocity_dim,
        "collect_stats": collect_stats,
        "dataset_size": len(all_actions),
        "traj_stats": {},
    }

    for k, v in data.items():
        stats["traj_stats"][k] = {
            "mean": np.mean(v, axis=0),
            "std": np.std(v, axis=0),
        }

    with open(str(output_path / "stats.pkl"), "wb") as f:
        pkl.dump(stats, f)

    sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "BC data collector script", parents=[get_args_parser()]
    )
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
