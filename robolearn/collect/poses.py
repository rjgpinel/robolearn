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
from robolearn.utils.tf import pos_quat_to_hom, hom_to_pos


def get_args_parser():
    parser = argparse.ArgumentParser("Pose data collector script", add_help=False)
    parser.add_argument(
        "--poses",
        default=20000,
        type=int,
        help="Number of poses to collect",
    )
    parser.add_argument(
        "--dr", dest="dr", action="store_true", help="Apply domain randomization"
    )
    parser.add_argument(
        "--masks", dest="masks", action="store_true", help="Collect segmentation masks"
    )
    parser.add_argument("--num-workers", default=10, type=int, help="Number of workers")
    parser.add_argument("--seed", default=0, type=int, help="Initial seed")
    parser.add_argument(
        "--output-dir",
        default="",
        type=str,
        help="Output directory where the expert demonstrations will be saved",
    )
    parser.set_defaults(masks=False, dr=False)
    return parser


def collect(worker_id, env_name, output_path, seeds, masks, data_queue):
    torch.set_num_threads(1)
    env = gym.make(
        env_name,
        num_distractors=2,
        segmentation_masks=masks,
    )
    pbar = None
    if worker_id == 0:
        pbar = tqdm(total=seeds[1] - seeds[0], ncols=80)

    cam_list = env.unwrapped.cam_list
    num_cubes = 1 + env.unwrapped.num_distractors
    stats = {
        "num_cubes": num_cubes,
        "cam_list": cam_list,
        "gripper_pos": [],
        "gripper_quat": [],
        "target_pos": [],
    }

    for seed in range(*seeds):
        env.seed(seed)
        obs = env.reset()

        gripper_pos, gripper_quat = obs["gripper_pos"], obs["gripper_quat"]
        target_pos = []
        for i in range(num_cubes):
            cube_pos, cube_quat = obs[f"cube{i}_pos"], [0, 0, 0, 1]

            world_T_target = pos_quat_to_hom(cube_pos, cube_quat)
            world_T_gripper = pos_quat_to_hom(gripper_pos, gripper_quat)
            gripper_T_world = np.linalg.inv(world_T_gripper)
            gripper_T_target = np.matmul(gripper_T_world, world_T_target)
            target_pos.append(hom_to_pos(gripper_T_target))

        target_pos = np.concatenate(target_pos)
        stats["gripper_pos"].append(gripper_pos)
        stats["gripper_quat"].append(gripper_quat)
        stats["target_pos"].append(target_pos)

        processed_obs = dict(
            gripper_pos=gripper_pos,
            gripper_quat=gripper_quat,
        )

        for cam_name in cam_list:
            processed_obs[f"rgb_{cam_name}"] = obs[f"rgb_{cam_name}"]
            if masks:
                processed_obs[f"seg_{cam_name}"] = obs[f"seg_{cam_name}"]

        if pbar is not None:
            pbar.update()

        with open(str(output_path / f"{seed:07d}.pkl"), "wb") as f:
            pkl.dump((processed_obs, target_pos), f)

    if pbar is not None:
        pbar.close()

    data_queue.put((worker_id, stats))
    print(f"Worker {worker_id} finished")
    del env


def main(args):
    output_path = Path(args.output_dir)
    initial_seed = args.seed
    poses = args.poses
    num_workers = args.num_workers
    env_name = "Pick-v0" if not args.dr else "DR-Pick-v0"
    masks = args.masks

    if num_workers > poses:
        num_workers = poses

    seeds_worker = compute_workers_seed(poses, num_workers, initial_seed)

    workers = []
    data_queue = mp.Queue()
    if num_workers == 0:
        collect(
            0,
            env_name,
            output_path,
            seeds_worker[0],
            masks,
            data_queue,
        )

        i, collect_stats = data_queue.get()
    else:
        for i in range(num_workers):
            w = mp.Process(
                target=collect,
                args=(
                    i,
                    env_name,
                    output_path,
                    seeds_worker[i],
                    masks,
                    data_queue,
                ),
            )
            w.daemon = True
            w.start()
            workers.append(w)
        counter = 0

        ###############################################
        collect_stats = None
        while counter < num_workers:
            i, worker_stats = data_queue.get()
            if collect_stats:
                for k, v in worker_stats.items():
                    if k not in ["cam_list", "num_cubes"]:
                        collect_stats[k] += v
            else:
                collect_stats = deepcopy(worker_stats)
            print(f"Worker {i} data received")
            counter += 1

    print("[Data Statistics] - Computing dataset statistics")
    stats = {
        "cam_list": collect_stats.pop("cam_list"),
        "num_cubes": collect_stats.pop("num_cubes"),
    }

    for k, v in collect_stats.items():
        stats[k] = {
            "mean": np.mean(v, axis=0),
            "std": np.std(v, axis=0),
        }

    with open(str(output_path / "stats.pkl"), "wb") as f:
        pkl.dump(stats, f)

    sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Pose data collector script", parents=[get_args_parser()]
    )
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
