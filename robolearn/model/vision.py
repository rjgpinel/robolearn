import torch
import torch.nn as nn
from einops import rearrange

from timm.models.layers import trunc_normal_


def size_to_pad(h, w, patch_size):
    pad_h = -h % patch_size
    pad_w = -w % patch_size
    return pad_h, pad_w


class TimmModel(nn.Module):
    def __init__(
        self,
        backbone,
        num_streams=1,
        merge_hist=False,
    ):
        super().__init__()
        self.backbone = backbone
        self.num_streams = num_streams
        self.num_features = backbone.num_features
        self.merge_hist = merge_hist

    def forward(self, im):
        if self.merge_hist:
            x = rearrange(im, "b n c h w -> (b n) c h w")
        else:
            x = rearrange(im, "b n f c h w -> (b n f) c h w")
            f = im.shape[2]
        h = self.backbone(x)
        if self.merge_hist:
            h = rearrange(h, "(b n) y -> b n y", n=self.num_streams)
        else:
            h = rearrange(h, "(b n f) y -> b n f y", n=self.num_streams, f=f)
        return h


class ViTModel(nn.Module):
    def __init__(
        self,
        backbone,
        num_streams=1,
        stream_integration="late",
    ):
        super().__init__()
        self.backbone = backbone
        self.patch_size = backbone.patch_embed.patch_size[0]
        self.num_streams = num_streams
        self.stream_integration = stream_integration
        self.num_features = backbone.num_features
        if self.stream_integration == "early":
            self.cam_tokens = nn.Parameter(
                torch.zeros(1, self.num_streams, 1, self.num_features)
            )
            trunc_normal_(self.cam_tokens, std=0.02)

    def pad(self, im):
        b, n, c, h, w = im.shape
        pad_h, pad_w = size_to_pad(h, w, self.patch_size)
        if pad_h > 0:
            im_pad_h = torch.zeros_like(im[:, :, :, :pad_h, :])
            im = torch.cat((im, im_pad_h), dim=3)
        if pad_w > 0:
            im_pad_w = torch.zeros_like(im[:, :, :, :, :pad_w])
            im = torch.cat((im, im_pad_w), dim=4)
        return im

    def forward(self, im):
        backbone = self.backbone
        im = self.pad(im)

        if self.stream_integration == "late":
            x = rearrange(im, "b n c h w -> (b n) c h w")
            x = backbone.patch_embed(x)
        elif self.stream_integration == "early":
            x = rearrange(im, "b n c h w -> (b n) c h w")
            x = backbone.patch_embed(x)
            x = rearrange(x, "(b n) l c -> b n l c", n=self.num_streams)
            cam_tokens = self.cam_tokens.expand(x.shape[0], -1, -1, -1)
            x = x + cam_tokens
            # cat view sequences
            x = rearrange(x, "b n l c -> b (n l) c")

        cls_token = backbone.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)

        pos_embed = backbone.pos_embed

        if self.stream_integration == "early":
            pos_embed_cls = pos_embed[:, :1]
            pos_embed_xy = pos_embed[:, 1:]
            pos_embed = torch.cat((pos_embed_cls, pos_embed_xy, pos_embed_xy), 1)

        x = backbone.pos_drop(x + pos_embed)
        h = backbone.blocks(x)
        h = backbone.norm(h)

        h = h[:, 0]

        if self.stream_integration == "late":
            h = rearrange(h, "(b n) y -> b n y", n=self.num_streams)

        return h
