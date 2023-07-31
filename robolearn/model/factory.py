import torch
import timm
import torch.nn as nn

from robolearn.model.vision import TimmModel, ViTModel, size_to_pad
from robolearn.model.bc import BCModel
from robolearn.model.pose import PoseModel
from muse.core.constants import REALSENSE_CROP


def create_vision_model(
    backbone_name, num_frames, num_streams, stream_integration, merge_hist
):
    w, h = REALSENSE_CROP
    in_chans = 3 * num_frames if merge_hist else 3

    backbone_kwargs = dict(
        model_name=backbone_name,
        pretrained=True,
        num_classes=0,
        in_chans=in_chans,
    )
    # pad the image during forward to get images divisible by patch_size
    if "vit" in backbone_name:
        if "patch16" in backbone_name:
            patch_size = 16
        elif "patch32" in backbone_name:
            patch_size = 32
        pad_h, pad_w = size_to_pad(h, w, patch_size=patch_size)
        w += pad_w
        h += pad_h
        backbone_kwargs["img_size"] = (w, h)

    backbone = timm.create_model(**backbone_kwargs)

    if "vit" in backbone_name:
        vision_model = ViTModel(backbone, num_streams, stream_integration)
    else:
        vision_model = TimmModel(backbone, num_streams, merge_hist)
    return vision_model


def create_bc_model(
    vision_model, data_stats, use_state, hist_len, state_hist=True
):
    model = BCModel(
        vision_model,
        data_stats=data_stats,
        use_state=use_state,
        state_hist=state_hist,
        hist_len=hist_len,
    )
    return model


def create_pose_model(vision_model, data_stats):
    model = PoseModel(vision_model, data_stats=data_stats, use_state=False)
    return model


def load_bc_model(ckp_dir, cam_list=""):
    ckp = torch.load(ckp_dir, map_location="cpu")
    args = ckp["args"]

    if cam_list == "":
        cam_list = args.cam_list
    cam_list = [f"{cam_name}" for cam_name in cam_list]
    args.cam_list = cam_list

    stats = ckp["stats"]
    num_frames = len(args.frame_hist) + 1
    num_streams = len(cam_list)
    vision_model = create_vision_model(
        args.model, num_frames, num_streams, args.stream_integration, args.merge_hist
    )
    model = create_bc_model(
        vision_model,
        stats,
        bool(args.state),
        state_hist=args.state != "gripper_pose_current",
        hist_len=len(args.frame_hist) + 1,
    )
    model.load_state_dict(ckp["model"])

    return model, args
