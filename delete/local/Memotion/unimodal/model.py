import torch
from torch.nn.functional import normalize
import torch.nn.functional as F
import torch.nn as nn
from transformers import CLIPTextModel, CLIPVisionModel


class CLIPtext(torch.nn.Module):
    def __init__(self, odim, modelname, cachedir, hiddendim):
        torch.nn.Module.__init__(self)
        self.cliptext = CLIPTextModel.from_pretrained(modelname, cache_dir=cachedir)
        self.text_map_input_dim = self.cliptext.config.hidden_size
        self.text_map = nn.Sequential(
            nn.Linear(self.text_map_input_dim, hiddendim),
            nn.Dropout(0.1),
        )
        self.pre_output_block = nn.Sequential(
                nn.Dropout(0.4),
                nn.Linear(hiddendim, hiddendim),
                nn.ReLU(), 
                nn.Dropout(0.2),
                nn.Linear(hiddendim, hiddendim),
                nn.ReLU(), 
                nn.Dropout(0.2),
                nn.Linear(hiddendim, hiddendim),
                nn.ReLU(), 
                nn.Dropout(0.2),
                nn.Linear(hiddendim, hiddendim),
                nn.ReLU(), 
                nn.Dropout(0.2)
            )
        self.output_layer = nn.Linear(hiddendim, odim) 


    def forward(self, nodes, mask):
        x = self.cliptext(nodes, mask).pooler_output
        x = self.text_map(x)
        emb = F.normalize(x, p=2, dim=1)
        emb = self.pre_output_block(emb)
        out = self.output_layer(emb)
        return out

    
class CLIPimage(torch.nn.Module):
    def __init__(self, odim, modelname, cachedir, hiddendim):
        torch.nn.Module.__init__(self)
        self.clipimage = CLIPVisionModel.from_pretrained(modelname, cache_dir=cachedir)
        self.image_map_input_dim = self.clipimage.config.hidden_size
        self.image_map = nn.Sequential(
            nn.Linear(self.image_map_input_dim, hiddendim),
            nn.Dropout(0.1),
        )
        self.pre_output_block = nn.Sequential(
                nn.Dropout(0.4),
                nn.Linear(hiddendim, hiddendim),
                nn.ReLU(), 
                nn.Dropout(0.2),
                nn.Linear(hiddendim, hiddendim),
                nn.ReLU(), 
                nn.Dropout(0.2),
                nn.Linear(hiddendim, hiddendim),
                nn.ReLU(), 
                nn.Dropout(0.2),
                nn.Linear(hiddendim, hiddendim),
                nn.ReLU(), 
                nn.Dropout(0.2)
            )
        self.output_layer = nn.Linear(hiddendim, odim) 

    def forward(self, image):
        x = self.clipimage(image).pooler_output
        x = self.image_map(x)
        emb = F.normalize(x, p=2, dim=1)
        emb = self.pre_output_block(emb)
        out = self.output_layer(emb)
        return out
