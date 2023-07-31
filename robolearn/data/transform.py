import torch.nn as nn
from einops import rearrange


class ImageTransform(nn.Module):
    def __init__(self, base_transform, normalization, merge_hist=False):
        super().__init__()
        self.base_transform = base_transform
        self.normalization = normalization
        self.merge_hist = merge_hist

    def __call__(self, frames, masks=None):
        # batch (b) view (v) frames (f) height (h) width (w) channels (c)
        b, v, f, h, w, c = frames.shape
        frames = rearrange(frames, "b v f h w c -> (b v f) c h w")
        frames = self.base_transform(frames)
        if self.normalization:
            frames = self.normalization(frames)
        frames = rearrange(frames, "(b v f) c h w -> b v f c h w", b=b, v=v, f=f)
        if self.merge_hist:
            frames = rearrange(frames, "b v f c h w -> b v (f c) h w", b=b, v=v, f=f)
        return frames
