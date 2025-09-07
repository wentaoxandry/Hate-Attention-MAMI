import torch
from torch.nn.functional import normalize
import torch.nn.functional as F
import torch.nn as nn
from transformers import CLIPTextModel, CLIPVisionModel


class CLIPmodel(torch.nn.Module):
    def __init__(self, modelname, cachedir):
        torch.nn.Module.__init__(self)
        self.cliptext = CLIPTextModel.from_pretrained(modelname, cache_dir=cachedir)
        self.clipimage = CLIPVisionModel.from_pretrained(modelname, cache_dir=cachedir)

    def forward(self, nodes, mask, image):
        x_text = self.cliptext(nodes, mask).pooler_output
        x_image = self.clipimage(image).pooler_output

        return x_text, x_image

  