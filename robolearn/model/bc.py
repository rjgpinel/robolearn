import torch
import torch.nn as nn
from einops import rearrange

from timm.models.layers import Mlp
from timm.models.vision_transformer import Block, _init_vit_weights
from timm.models.layers import trunc_normal_

from robolearn.data.utils import denormalize_action


class BCModel(nn.Module):
    def __init__(
        self,
        vision_model,
        data_stats,
        use_state,
        state_hist,
        hist_len,
    ):
        super().__init__()
        self.vision_model = vision_model
        self.data_stats = data_stats
        self.vel_dim = data_stats["vel_dim"]
        self.use_state = use_state
        self.state_hist = state_hist
        self.hist_len = hist_len
        self.merge_hist = vision_model.merge_hist
        self.state_dim = 5

        # standard mlp
        act_layer = nn.ReLU
        if self.merge_hist:
            embed_dim = vision_model.num_streams * vision_model.num_features
        else:
            embed_dim = (
                self.hist_len * vision_model.num_streams * vision_model.num_features
            )
        if self.use_state:
            if self.state_hist:
                embed_dim += self.hist_len * self.state_dim
            else:
                embed_dim += self.state_dim
        self.vel_mlp = Mlp(embed_dim, out_features=self.vel_dim, act_layer=act_layer)
        self.vel_mlp.apply(_init_vit_weights)
        if "grip_open" in self.data_stats["action_space"].keys():
            self.grip_mlp = Mlp(embed_dim, out_features=1, act_layer=act_layer)
            self.grip_mlp.apply(_init_vit_weights)

    def forward(self, im, state=None, **kwargs):
        # h shape: (batch, view, features)
        h = self.vision_model.forward(im)

        # standard mlp
        if self.merge_hist:
            h = rearrange(h, "b n y -> b (n y)")
        else:
            h = rearrange(h, "b n f y -> b (n f y)")

        if self.use_state:
            # state shape: (batch, hist, features)

            if self.state_hist:
                state = rearrange(state, "b h f -> b (h f)")
            else:
                # Use only current propioceptive info
                state = state[:, -1]

            h = torch.cat((h, state), dim=1)
        vel = self.vel_mlp(h)
        pred = {"vel": vel}

        if "grip_open" in self.data_stats["action_space"].keys():
            grip = self.grip_mlp(h)
            grip = grip.sigmoid()
            pred["grip"] = grip

        return pred

    def process_output(self, action):
        denorm_action = denormalize_action(action, self.data_stats, task="bc")
        return denorm_action
