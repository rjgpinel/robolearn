import torch
from torchvision import transforms as T

from robolearn.data.loader import Loader
from robolearn.data.utils import get_im_norm_stats

from robolearn.data.demonstration import DemonstrationDataset

from robolearn.data.pose import PoseDataset
from robolearn.data.transform import ImageTransform
import robolearn.utils.torch as ptu


def create_dataset(args, path, split, loader=True):
    if args.task in ["bc"]:
        dataset = DemonstrationDataset(
            path=path,
            split=split,
            cam_list=args.cam_list,
            frame_hist=args.frame_hist,
            delay_hist=args.delay_hist,
            task=args.task,
        )
    elif args.task == "pose":
        dataset = PoseDataset(
            path=path,
            split=split,
            cam_list=args.cam_list,
        )
    else:
        raise ValueError(f"Unknown task type: {args.task}")

    if loader:
        dataset = Loader(
            dataset=dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            distributed=ptu.distributed,
            shuffle="train" in split,
        )

    return dataset


def create_transform(args, dataset, normalization=None, data_aug="", merge_hist=False):
    if normalization:
        stats_img = get_im_norm_stats(normalization)
        normalization = T.Normalize(stats_img["mean"], stats_img["std"])

    if data_aug == "":
        base_transform = torch.nn.Identity()
        return ImageTransform(base_transform, normalization, merge_hist)
    elif data_aug == "iros23_s2r":
        base_transform = []
        # DeepMind rgb_stack data augmentation
        base_transform.append(
            T.ColorJitter(
                brightness=32 / 255,
                contrast=[0.5, 1.5],
                saturation=[0.5, 1.5],
                hue=1 / 24,
            )
        )
        # NOTE: Hardcoded image resolution - Use dataset to get image resolution
        base_transform.append(T.RandomAffine(degrees=0, translate=(4 / 240, 4 / 180)))
        base_transform = T.Compose(base_transform)
        return ImageTransform(base_transform, normalization, merge_hist)
    else:
        raise ValueError(f"Unknown data augmentation: {data_aug}")
