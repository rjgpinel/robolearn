import pylab
import numpy as np

from collections import OrderedDict
from PIL import Image
from functools import partial
from einops import rearrange

COLOR_MAP = pylab.get_cmap("viridis")


class AttentionHook:
    def __init__(self, model):
        self.model = model
        self.hooks = []
        self.attentions = OrderedDict()
        self.num_streams = self.model.vision_model.num_streams

        # base resnet layers
        for i in range(1, 5):
            name = f"layer{i}"
            module = getattr(model.vision_model.backbone, name, None)
            if module is not None:
                self._register_hook(module, name)

    def _register_hook(self, module, name):
        print(f"registring hook for {name}")
        hook = module.register_forward_hook(partial(self._hook_fn, name=name))
        self.hooks.append(hook)

    def _hook_fn(self, module, input, output, name):
        self.attentions[name] = output.detach().pow(2).mean(1).squeeze().cpu().numpy()

    def get_map(self, w, h):
        maps = list(self.attentions.values())

        resized_att_maps = []
        for att_map in maps:
            att_map = att_map / att_map.max()
            att_map = rearrange(att_map, "v h w -> h (v w)")
            att_map = Image.fromarray(att_map)
            resized_att_map = att_map.resize((w, h), Image.BILINEAR)
            resized_att_maps.append(np.asarray(resized_att_map))

        attention = np.prod(resized_att_maps, axis=0)
        attention = attention / np.max(attention)
        return attention

    def blend_map(self, image, alpha=0.6):
        h, w, _ = image.shape
        attention = self.get_map(w, h)

        image = Image.fromarray(image)
        heatmap = (COLOR_MAP(attention)[:, :, :3] * 255).astype(np.uint8)

        heatmap = Image.fromarray(heatmap)
        im_blend = Image.blend(image, heatmap, alpha).convert("RGB")
        im_blend = np.asanyarray(im_blend).copy()
        return im_blend
