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
        return out, emb

    
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
        return out, emb


class DSW(torch.nn.Module):
    def __init__(self, odim, modelname, cachedir, hiddendim, RMs):
        torch.nn.Module.__init__(self)
        self.textclassifier = CLIPtext(odim=odim, modelname=modelname, cachedir=cachedir, hiddendim=768)
        self.imageclassifier = CLIPimage(odim=odim, modelname=modelname, cachedir=cachedir, hiddendim=768)
        self.RMs = RMs
        if self.RMs == 'all_RMs':
            ifeatdim = 3
        elif self.RMs == 'text_RMs':
            ifeatdim = 1
        elif self.RMs == 'image_RMs':
            ifeatdim = 2
        elif self.RMs == 'no_RMs':
            ifeatdim = 2 * odim

        self.weightpredictor = torch.nn.Sequential(torch.nn.Linear(ifeatdim, hiddendim, bias=True),
                                                   torch.nn.GELU(),
                                                   torch.nn.Linear(hiddendim, int(hiddendim / 2), bias=True),
                                                   torch.nn.Tanh(),
                                                   torch.nn.Linear(int(hiddendim / 2), 2, bias=True),
                                                   torch.nn.Sigmoid())
        for param in self.textclassifier.parameters():
            param.requires_grad = False
        for param in self.imageclassifier.parameters():
            param.requires_grad = False
    def forward(self, nodes, mask, pixel, features):
        text_logits, _ = self.textclassifier(nodes, mask)
        image_logits, _ = self.imageclassifier(pixel)
        if self.RMs == 'all_RMs':
            features = features
        elif self.RMs == 'text_RMs':
            features = features[:, -1].unsqueeze(-1)
        elif self.RMs == 'image_RMs':
            features = features[:, :-1]
        elif self.RMs == 'no_RMs':
            features = torch.cat([text_logits, image_logits], dim=-1)
        weight = self.weightpredictor(features)
        x = weight[:, 0].unsqueeze(1) * image_logits + weight[:, 1].unsqueeze(1) * text_logits
        return x


class Representation(torch.nn.Module):
    def __init__(self, odim, modelname, cachedir, hiddendim, RMs):
        torch.nn.Module.__init__(self)
        self.textclassifier = CLIPtext(odim=odim, modelname=modelname, cachedir=cachedir, hiddendim=hiddendim)
        self.imageclassifier = CLIPimage(odim=odim, modelname=modelname, cachedir=cachedir, hiddendim=hiddendim)

        self.RMs = RMs
        if self.RMs == 'all_RMs':
            ifeatdim = 3
        elif self.RMs == 'text_RMs':
            ifeatdim = 1
        elif self.RMs == 'image_RMs':
            ifeatdim = 2
        elif self.RMs == 'no_RMs':
            ifeatdim = 0

        self.pre_output_block = nn.Sequential(
                nn.Dropout(0.4),
                nn.Linear(2 * hiddendim + ifeatdim, hiddendim),
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
        for param in self.textclassifier.parameters():
            param.requires_grad = False
        for param in self.imageclassifier.parameters():
            param.requires_grad = False

    def forward(self, nodes, mask, pixel, features):
        text_logits, text_feature = self.textclassifier(nodes, mask)
        image_logits, image_feature = self.imageclassifier(pixel)

        if self.RMs == 'no_RMs':
            fusion_feat = torch.cat([text_feature, image_feature], dim=-1)
        else:
            if self.RMs == 'all_RMs':
                scores = features
            elif self.RMs == 'text_RMs':
                scores = features[:, -1].unsqueeze(-1)
            elif self.RMs == 'image_RMs':
                scores = features[:, :-1]
            fusion_feat = torch.cat([scores, image_feature, text_feature], dim=-1)   


        emb = self.pre_output_block(fusion_feat)
        out = self.output_layer(emb)
        return out