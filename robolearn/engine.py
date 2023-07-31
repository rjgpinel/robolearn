import math
import sys
import json
import muse.envs
import gym
import torch
import wandb

import robolearn.utils.torch as ptu
from tqdm import tqdm

from robolearn.utils.logger import MetricLogger
from robolearn.utils import distributed
from robolearn.evaluate.policy import evaluate_seeds



def train_one_epoch(
    model,
    data_loader,
    transform,
    criterion,
    optimizer,
    epoch,
    log_dir,
):
    model.train()
    criterion.train()

    logger = MetricLogger(delimiter="  ")
    header = f"Epoch: [{epoch}]"
    print_freq = 100
    num_frames = data_loader.dataset.num_frames
    num_streams = data_loader.dataset.num_streams

    data_loader.set_epoch(epoch)
    num_updates = epoch * len(data_loader)
    for frames, masks, state, target in logger.log_every(
        data_loader, print_freq, header
    ):
        frames = frames.to(ptu.device)
        masks = masks.to(ptu.device)
        state = state.to(ptu.device)
        for k, v in target.items():
            if isinstance(v, torch.Tensor):
                target[k] = v.to(ptu.device)
        frames = transform(frames, masks)

        pred = model(frames, state, target=target)
        loss, losses_log = criterion(pred, target)
        loss_value = loss.item()

        if num_updates % 100 == 0:
            losses_log = distributed.reduce_dict(losses_log)
            losses_log["num_updates"] = num_updates
            if ptu.dist_rank == 0:
                with open(log_dir / "log_train.txt", "a") as f:
                    f.write(json.dumps(losses_log) + "\n")

        logger.update(loss=loss_value)
        logger.update(learning_rate=optimizer.param_groups[0]["lr"])

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value), force=True)
            sys.exit(1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        num_updates += 1

        torch.cuda.synchronize()

    return logger


def evaluate(
    model,
    data_loaders,
    transform,
    criterion,
    metric,
    env,
    seeds,
    cam_list,
    frame_hist,
    seed_setup_path,
    log_dir,
    print_freq,
    wand_logs=False
):
    model.eval()
    criterion.eval()
    model_no_ddp = model
    if hasattr(model, "module"):
        model_no_ddp = model.module

    logger = MetricLogger(delimiter="  ")
    header = "Eval:"

    # offline evaluation
    for data_loader in data_loaders:
        for frames, masks, state, target in logger.log_every(
            data_loader, print_freq, header
        ):
            frames = frames.to(ptu.device)
            masks = masks.to(ptu.device)
            state = state.to(ptu.device)
            for k, v in target.items():
                if isinstance(v, torch.Tensor):
                    target[k] = v.to(ptu.device)
            frames = transform(frames, masks)

            with torch.no_grad():
                pred = model_no_ddp(frames, state)
                loss, losses_log = criterion(pred, target)
                if metric:
                    denorm_pred = model_no_ddp.process_output(pred)
                    denorm_target = data_loader.dataset.denormalize_target(target)
                    metric_log = metric(denorm_pred, denorm_target)
                    metric_log = distributed.reduce_dict(metric_log)
                    logger.update(prefix=f"{data_loader.dataset.split}_", **metric_log)

            losses_log = distributed.reduce_dict(losses_log)
            logger.update(prefix=f"{data_loader.dataset.split}_", **losses_log)

    if env:
        # online evaluation
        pbar = None
        if ptu.dist_rank == 0:
            pbar = tqdm(total=seeds[1] - seeds[0], ncols=80)
        cam_list = [f"{cam_name}" for cam_name in cam_list]
        stats = evaluate_seeds(
            model_no_ddp,
            env,
            seeds,
            cam_list,
            frame_hist,
            transform,
            log_dir,
            seed_setup_path=seed_setup_path,
            record=True,
            pbar=pbar,
        )

        if wand_logs:
            wandb.log(
                {
                    f"{seed:08d}": wandb.Video(
                        str(log_dir / f"evaluation/{seed:08d}.mp4"), format="gif"
                    )
                    for seed in range(*seeds)
                }
        )

        num_episodes = len(stats["successful_seeds"]) + len(stats["failure_seeds"])
        success_rate = len(stats["successful_seeds"]) / num_episodes

        logger.update(val_num_episodes=num_episodes, val_success_rate=success_rate)

    # gather the stats from all processes
    logger.synchronize_between_processes()

    return logger
