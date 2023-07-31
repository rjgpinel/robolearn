import argparse
import torch
import numpy as np

import robolearn.utils.torch as ptu
from robolearn.model.factory import create_vision_model, create_pose_model
from robolearn.utils.logger import MetricLogger
from robolearn.data.factory import create_dataset, create_transform
from robolearn.data.utils import denormalize
from robolearn.evaluate.metric import PoseMetric

from pathlib import Path


NUM_CUBES = 3


def get_args_parser():
    parser = argparse.ArgumentParser("Pose evaluation script", add_help=False)
    # Model
    parser.add_argument("--checkpoint", default="")
    # Dataset
    parser.add_argument("--data-path", type=str)
    return parser


def evaluate_poses(checkpoint_dir, data_path):
    checkpoint = torch.load(checkpoint_dir, map_location="cpu")
    ptu.set_gpu_mode(True)
    normalization = "vit" if "vit" in checkpoint["args"].model else "resnet"
    val_loader = create_dataset(
        checkpoint["args"],
        path=data_path,
        split="val",
        loader=False,
    )

    val_transform = create_transform(
        checkpoint["args"],
        val_loader,
        normalization=normalization,
        data_aug="",
        merge_hist=checkpoint["args"].merge_hist,
    )

    # Model
    vision_model = create_vision_model(
        checkpoint["args"].model,
        num_frames=val_loader.num_frames,
        num_streams=val_loader.num_streams,
        stream_integration=checkpoint["args"].stream_integration,
        merge_hist=checkpoint["args"].merge_hist,
    )

    model = create_pose_model(vision_model=vision_model, data_stats=checkpoint["stats"])

    print("Loading model")
    model.load_state_dict(checkpoint["model"])
    model.eval()
    model.to(ptu.device)

    metric = PoseMetric()

    logger = MetricLogger(delimiter="  ")
    header = "Eval:"
    print_freq = 20

    #### Evaluating Data ####
    error_list = []
    error_mean_list = []
    error_cubes = {f"cube{i}_error": [] for i in range(val_loader.num_cubes)}
    preds = []
    targets = []

    for frames, masks, state, target in logger.log_every(
        val_loader, print_freq, header
    ):

        frames = frames.to(ptu.device).unsqueeze(0)
        state = state.to(ptu.device).unsqueeze(0)
        for k, v in target.items():
            if isinstance(v, torch.Tensor):
                target[k] = v.to(ptu.device).unsqueeze(0)
        with torch.no_grad():
            frames = val_transform(frames)
            pred = model(frames, state)
            denorm_pred = model.process_output(pred)
            preds.append(denorm_pred.cpu().numpy())
            denorm_target = denormalize(target["pose"], target["stats"])
            targets.append(denorm_target.cpu().numpy())

            metric_log = metric(denorm_pred, denorm_target)
            logger.update(prefix=f"{val_loader.split}_", **metric_log)
            error_list.append(metric_log["pose_error"])
            acc_error = 0
            for i in range(val_loader.num_cubes):
                acc_error += metric_log[f"error_cube{i}"]
                error_cubes[f"cube{i}_error"].append(metric_log[f"error_cube{i}"])
            error_mean_list.append(acc_error / val_loader.num_cubes)

    preds = np.asarray(preds)
    targets = np.asarray(targets)

    error_list = torch.tensor(error_list)
    error_mean_list = torch.tensor(error_mean_list)
    print(f"Model with mean error: {error_mean_list.mean()}")
    return error_mean_list.mean()


def main(args):
    evaluate_poses(args.checkpoint, args.data_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Pose evaluation script", parents=[get_args_parser()]
    )
    args = parser.parse_args()
    if not Path(args.checkpoint).exists():
        raise ValueError(f"Checkpoint directory {args.checkpoint} does not exist.")
    main(args)
