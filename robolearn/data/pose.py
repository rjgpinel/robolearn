import torch
import pickle as pkl

from pathlib import Path
from torch.utils.data import Dataset
from robolearn.data.utils import (
    flatten_normalize_dict,
    normalize,
    normalize_frames,
    state_to_frames,
    filter_state,
    denormalize,
)


class PoseDataset(Dataset):
    def __init__(self, path, split, cam_list):
        self.root = Path(path)
        if not self.root.exists():
            raise ValueError(f"Dataset path {self.root} doesn't exists")

        self.split = split
        with open(str(self.root / "stats.pkl"), "rb") as f:
            self.stats = pkl.load(f)

        self.scene_files = sorted(self.root.glob("[0-9]*.pkl"))
        self.cam_list = cam_list or self.stats["cam_list"]
        self.num_streams = len(self.cam_list)
        self.num_frames = 1
        self.num_cubes = self.stats["num_cubes"]

    def __len__(self):
        return len(self.scene_files)

    def __getitem__(self, idx):
        scene_file = self.scene_files[idx]
        with open(scene_file, "rb") as f:
            state, target_pos = pkl.load(f)

        frames, masks = state_to_frames(state, self.cam_list)
        frames, masks = normalize_frames([frames], [masks], 1)

        state = filter_state(state)
        state = flatten_normalize_dict(state, self.stats)

        target_pos = torch.tensor(target_pos).float()

        target_pos = normalize(target_pos, self.stats["target_pos"])

        target = {
            "pose": target_pos,
            "stats": self.stats["target_pos"],
        }

        return frames, masks, state, target

    def denormalize_target(self, target):
        denorm_target = denormalize(target["pose"], target["stats"])
        return denorm_target
