import argparse
import numpy as np
import math
import sys
import json
import torch
import gym
import wandb

from robolearn.model.factory import (
    create_vision_model,
    create_bc_model,
    create_pose_model,
)
import robolearn.utils.distributed as dist
import robolearn.utils.torch as ptu

from pathlib import Path

from timm.optim import create_optimizer
from timm.scheduler import create_scheduler
from torch.nn.parallel import DistributedDataParallel as DDP

from robolearn.collect.utils import compute_workers_seed
from robolearn.engine import train_one_epoch, evaluate
from robolearn.data.factory import create_dataset, create_transform
from robolearn.criterion import BCCriterion, PoseCriterion
from robolearn.evaluate.metric import PoseMetric, VelMetric


def get_args_parser():
    parser = argparse.ArgumentParser("BC training script", add_help=False)
    # Model
    parser.add_argument("--task", default="bc", type=str)
    parser.add_argument("--model", default="resnet18", type=str)
    parser.add_argument("--stream-integration", default="late", type=str)
    parser.add_argument("--merge-hist", default=True, type=bool)
    # Optimizer
    parser.add_argument("--opt", default="adamw", type=str)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--sched", default="step", type=str)
    parser.add_argument("--warmup-ratio", type=float, default=0.0)
    parser.add_argument("--decay-rate", type=float, default=0.1)
    parser.add_argument("--batch-size", default=32, type=int)
    # Dataset
    parser.add_argument("--train-steps", type=int, default=400000)
    parser.add_argument("--eval-steps", type=int, default=25000)
    # parser.add_argument("--data-aug", type=str, default="iros23_s2r")
    parser.add_argument("--data-aug", type=str, default="")
    parser.add_argument("--train-path", type=str)
    parser.add_argument("--eval-path-sim", type=str)
    parser.add_argument("--eval-path-real", default="", type=str)
    parser.add_argument("--frame-hist", type=str, default="1,2")
    parser.add_argument("--delay-hist", type=int, default=0)
    parser.add_argument("--num-workers", default=10, type=int)
    parser.add_argument("--cam-list", default="", type=str)
    # parser.add_argument("--state", default="gripper_pose_current", type=str)
    parser.add_argument("--state", default="", type=str)
    # Criterion
    parser.add_argument("--lam", default=0.8, type=float)
    # Evaluation
    parser.add_argument("--eval-env", default="Pick-v0", type=str)
    parser.add_argument("--eval-seed", default=9000, type=int, help="Initial seed")
    parser.add_argument("--eval-episodes", default=250, type=int)
    parser.add_argument("--seed-setup-path", default="", type=str)
    # Log
    parser.add_argument("--log-dir", default="")
    # Run name
    parser.add_argument("--run-name", default="")
    parser.add_argument("--wandb-logs", dest="wandb-logs", action="store_true", help="Save logs in wandb.ai")
    parser.add_argument("--wandb-project", default="robolearn")
    parser.add_argument("--wandb-entity", default="robolearn")
    parser.set_defaults(wandb_logs=False)
    return parser


def main(args):
    ptu.set_gpu_mode(True)
    dist.init_process()

    # Process user args
    if ptu.world_size > args.eval_episodes:
        args.episodes_eval = ptu.world_size

    eval_seeds = compute_workers_seed(
        args.eval_episodes, ptu.world_size, args.eval_seed
    )[ptu.dist_rank]

    args.log_dir = Path(args.log_dir)

    # Dataset
    if args.cam_list:
        args.cam_list = args.cam_list.split(",")

    if args.task == "pose":
        args.merge_hist = True

    if args.frame_hist:
        frame_hist = args.frame_hist.split(",")
        args.frame_hist = sorted(list(map(int, frame_hist)))
    else:
        args.frame_hist = []

    normalization_train = "vit" if "vit" in args.model else "resnet"
    normalization_val = normalization_train
    train_loader = create_dataset(
        args,
        path=args.train_path,
        split="train",
        loader=True,
    )
    args.cam_list = args.cam_list or train_loader.dataset.cam_list

    # TODO: update separation character
    args.eval_path_real = args.eval_path_real.split(";")
    val_data_loaders = []
    for split, path in zip(
        ["val_sim", *[f"val_real{i}" for i in range(len(args.eval_path_real))]],
        [args.eval_path_sim, *args.eval_path_real],
    ):
        if path:
            val_loader = create_dataset(
                args,
                path=path,
                split=split,
                loader=True,
            )
            val_data_loaders.append(val_loader)

    train_transform = create_transform(
        args,
        train_loader.dataset,
        normalization=normalization_train,
        data_aug=args.data_aug,
        merge_hist=args.merge_hist,
    )
    val_transform = create_transform(
        args,
        val_loader.dataset,
        normalization=normalization_val,
        data_aug="",
        merge_hist=args.merge_hist,
    )

    # Epochs
    num_epochs = args.train_steps / len(train_loader)
    num_epochs = math.ceil(num_epochs)
    args.epochs = num_epochs

    eval_steps = np.arange(0, args.train_steps, step=args.eval_steps)
    eval_epochs = set([step // len(train_loader) for step in eval_steps])
    eval_epochs.add(num_epochs - 1)
    eval_epochs = sorted(list(eval_epochs))
    eval_epochs = eval_epochs[1:]
    print(f"Eval epochs: {eval_epochs}")

    # Model
    vision_model = create_vision_model(
        args.model,
        num_frames=train_loader.dataset.num_frames,
        num_streams=train_loader.dataset.num_streams,
        stream_integration=args.stream_integration,
        merge_hist=args.merge_hist,
    )

    env = None
    if args.task == "bc":
        model = create_bc_model(
            vision_model=vision_model,
            data_stats=train_loader.dataset.stats,
            use_state=bool(args.state),
            state_hist=args.state != "gripper_pose_current",
            hist_len=len(args.frame_hist) + 1,
        )
        criterion = BCCriterion(args.lam)
        criterion = criterion.to(ptu.device)
        args.im_transform = val_transform
        env = gym.make(args.eval_env)
        if args.eval_env in ["Pick-v0", "Stack-v0"]:
            metric = VelMetric("linear_velocity")
        else:
            metric = None
    elif args.task == "pose":
        model = create_pose_model(
            vision_model=vision_model, data_stats=train_loader.dataset.stats
        )
        criterion = PoseCriterion()
        criterion = criterion.to(ptu.device)
        metric = PoseMetric(num_cubes=train_loader.dataset.num_cubes)
        args.im_transform = val_transform
    else:
        raise ValueError(f"Unknown task: {args.task}")
    model.to(ptu.device)

    # Optimization
    args.warmup_lr = 1e-6
    args.warmup_epochs = math.ceil(args.warmup_ratio * num_epochs)
    if args.sched == "step":
        # decay learning rate by a factor 0.1 after 80% of total epochs
        args.decay_rate = 0.1
        args.decay_epochs = np.round(0.8 * num_epochs)
    elif args.sched == "cosine":
        args.min_lr = 1e-6
        args.cooldown_epochs = 0

    optimizer = create_optimizer(args, model)
    lr_scheduler, _ = create_scheduler(args, optimizer)

    print(args)

    # Resume
    checkpoint_path = args.log_dir / "checkpoint.pth"
    start_epoch = 0
    if checkpoint_path.exists():
        print(f"Resuming training from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        start_epoch = checkpoint["epoch"] + 1
    else:
        dist.sync_model(args.log_dir, model)

    model_without_ddp = model
    if ptu.distributed:
        model = DDP(model, device_ids=[ptu.device])

    if args.wandb_logs:
        # Init wandb
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            dir=str(args.log_dir),
            name=f"{args.task} - {args.run_name}",
        )

        # Store parameters in wandb
        wandb.config = vars(args)

    # Train
    for epoch in range(start_epoch, num_epochs):
        train_logger = train_one_epoch(
            model=model,
            data_loader=train_loader,
            transform=train_transform,
            criterion=criterion,
            optimizer=optimizer,
            epoch=epoch,
            log_dir=args.log_dir,
        )
        lr_scheduler.step(epoch + 1)

        if ptu.dist_rank == 0:
            snapshot = dict(
                model=model_without_ddp.state_dict(),
                optimizer=optimizer.state_dict(),
                lr_scheduler=lr_scheduler.state_dict(),
                epoch=epoch,
                args=args,
                stats=train_loader.dataset.stats,
            )
            torch.save(snapshot, checkpoint_path)

        eval_logger = None
        if epoch in eval_epochs:
            eval_logger = evaluate(
                model=model,
                data_loaders=val_data_loaders,
                transform=val_transform,
                criterion=criterion,
                metric=metric,
                env=env,
                seeds=eval_seeds,
                cam_list=args.cam_list,
                frame_hist=args.frame_hist,
                seed_setup_path=args.seed_setup_path,
                log_dir=Path(args.log_dir),
                print_freq=40,
            )

        if ptu.dist_rank == 0:
            train_stats = {
                k: meter.global_avg for k, meter in train_logger.meters.items()
            }
            val_stats = {}
            if epoch in eval_epochs:
                print(f"Eval Stats [{epoch}]:", eval_logger)
                print("")

                val_stats = {
                    k: meter.global_avg for k, meter in eval_logger.meters.items()
                }

            log_stats = {
                **{f"train_{k}": v for k, v in train_stats.items()},
                **val_stats,
                "epoch": epoch,
                "num_updates": (epoch + 1) * len(train_loader),
            }

            for k, v in log_stats.items():
                if torch.is_tensor(v):
                    log_stats[k] = v.item()
            with open(args.log_dir / "log.txt", "a") as f:
                f.write(json.dumps(log_stats) + "\n")

            if args.wandb_logs:
                # Store wandb logs
                wandb.log(log_stats)

        dist.barrier()

    del env
    dist.destroy_process()
    sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("BC training script", parents=[get_args_parser()])
    args = parser.parse_args()
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    main(args)
