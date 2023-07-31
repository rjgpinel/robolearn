import torch.nn as nn
from einops import rearrange

from timm.models.layers import Mlp

from robolearn.data.utils import denormalize


class PoseModel(nn.Module):
    def __init__(
        self,
        vision_model,
        data_stats,
        use_state,
    ):
        super().__init__()
        self.vision_model = vision_model
        self.data_stats = data_stats
        self.num_cubes = self.data_stats["num_cubes"]

        embed_dim = vision_model.num_streams * vision_model.num_features
        act_layer = nn.ReLU
        self.pose_mlp = Mlp(
            embed_dim, out_features=3 * self.num_cubes, act_layer=act_layer
        )

    def forward(self, im, state=None, **kwargs):
        h = self.vision_model.forward(im)
        h = rearrange(h, "b n y -> b (n y)")

        pose = self.pose_mlp(h)

        pred = {"pose": pose}
        return pred

    def process_output(self, output):
        return denormalize(output["pose"], self.data_stats["target_pos"])
