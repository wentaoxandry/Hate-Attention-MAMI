import torch
from torch.nn.functional import normalize
import torch.nn.functional as F
import torch.nn as nn
from transformers import CLIPTextModel, CLIPVisionModel

class CLIPtext(torch.nn.Module):
    def __init__(self, itdim, odim, modelname, cachedir, hiddendim, task):
        torch.nn.Module.__init__(self)
        self.cliptext = CLIPTextModel.from_pretrained(modelname, cache_dir=cachedir)
        self.text_map_input_dim = self.cliptext.config.hidden_size
        self.text_map = nn.Sequential(
            nn.Linear(self.text_map_input_dim, hiddendim),
            nn.Dropout(0.1),
        )
        self.task = task
        if self.task == 'taskA':
            self.pre_output_block = nn.Sequential(
                nn.Dropout(0.4),
                nn.Linear(hiddendim, hiddendim),
                nn.ReLU(), 
                nn.Dropout(0.2)
            )
            self.output_layer = nn.Linear(hiddendim, odim) 
        else:
            self.pre_output_block1 = nn.Sequential(
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
            self.output_layer1 = nn.Linear(hiddendim, 1)  
            self.pre_output_block2 = nn.Sequential(
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
            self.output_layer2 = nn.Linear(hiddendim, 1)     
            self.pre_output_block3 = nn.Sequential(
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
            self.output_layer3 = nn.Linear(hiddendim, 1)     
            self.pre_output_block4 = nn.Sequential(
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
            self.output_layer4 = nn.Linear(hiddendim, 1) 


    def forward(self, nodes, mask):
        x = self.cliptext(nodes, mask).pooler_output
        x = self.text_map(x)
        emb = F.normalize(x, p=2, dim=1)
        if self.task == "taskA":
            emb = self.pre_output_block(emb)
            out = self.output_layer(emb)
            return out, emb

        else:
            emb1 = self.pre_output_block1(emb)
            out1 = self.output_layer1(emb1)
            emb2 = self.pre_output_block2(emb)
            out2 = self.output_layer2(emb2)
            emb3 = self.pre_output_block3(emb)
            out3 = self.output_layer3(emb3)
            emb4 = self.pre_output_block4(emb)
            out4 = self.output_layer4(emb4)
            out = torch.cat([out1, out2, out3, out4], dim=-1)
            return out, emb
    
class CLIPimage(torch.nn.Module):
    def __init__(self, iidim, odim, modelname, cachedir, hiddendim, task):
        torch.nn.Module.__init__(self)
        self.clipimage = CLIPVisionModel.from_pretrained(modelname, cache_dir=cachedir)
        self.image_map_input_dim = self.clipimage.config.hidden_size
        self.image_map = nn.Sequential(
            nn.Linear(self.image_map_input_dim, hiddendim),
            nn.Dropout(0.1),
        )
        self.task = task
        if self.task == 'taskA':
            self.pre_output_block = nn.Sequential(
                nn.Dropout(0.4),
                nn.Linear(hiddendim, hiddendim),
                nn.ReLU(), 
                nn.Dropout(0.2)
            )
            self.output_layer = nn.Linear(hiddendim, odim) 
        else:
            self.pre_output_block1 = nn.Sequential(
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
            self.output_layer1 = nn.Linear(hiddendim, 1)  
            self.pre_output_block2 = nn.Sequential(
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
            self.output_layer2 = nn.Linear(hiddendim, 1)     
            self.pre_output_block3 = nn.Sequential(
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
            self.output_layer3 = nn.Linear(hiddendim, 1)     
            self.pre_output_block4 = nn.Sequential(
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
            self.output_layer4 = nn.Linear(hiddendim, 1) 

    def forward(self, image):
        x = self.clipimage(image).pooler_output
        x = self.image_map(x)
        emb = F.normalize(x, p=2, dim=1)
        if self.task == "taskA":
            emb = self.pre_output_block(emb)
            out = self.output_layer(emb)
            return out, emb

        else:
            emb1 = self.pre_output_block1(emb)
            out1 = self.output_layer1(emb1)
            emb2 = self.pre_output_block2(emb)
            out2 = self.output_layer2(emb2)
            emb3 = self.pre_output_block3(emb)
            out3 = self.output_layer3(emb3)
            emb4 = self.pre_output_block4(emb)
            out4 = self.output_layer4(emb4)
            out = torch.cat([out1, out2, out3, out4], dim=-1)
            return out, emb
    
class DSW(torch.nn.Module):
    def __init__(self, itdim, iidim, odim, modelname, cachedir, hiddendim, RMs, task):
        torch.nn.Module.__init__(self)
        self.textclassifier = CLIPtext(itdim=itdim, odim=odim, modelname=modelname, cachedir=cachedir, hiddendim=itdim, task=task) 
        self.imageclassifier = CLIPimage(iidim=iidim, odim=odim, modelname=modelname, cachedir=cachedir, hiddendim=itdim, task=task) 
        
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
                                              torch.nn.Linear(hiddendim, int(hiddendim/2), bias=True),
                                              torch.nn.Tanh(),
                                              torch.nn.Linear(int(hiddendim/2), 2, bias=True),
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
    def __init__(self, itdim, iidim, odim, modelname, cachedir, hiddendim, RMs, task):
        torch.nn.Module.__init__(self)
        self.textclassifier = CLIPtext(itdim=itdim, odim=odim, modelname=modelname, cachedir=cachedir, hiddendim=itdim, task=task) 
        self.imageclassifier = CLIPimage(iidim=iidim, odim=odim, modelname=modelname, cachedir=cachedir, hiddendim=itdim, task=task) 

        self.RMs = RMs
        if self.RMs == 'all_RMs':
            ifeatdim = 3
        elif self.RMs == 'text_RMs':
            ifeatdim = 1
        elif self.RMs == 'image_RMs':
            ifeatdim = 2
        elif self.RMs == 'no_RMs':
            ifeatdim = 0
        
        self.task = task
        if self.task == 'taskA':
            self.pre_output_block = nn.Sequential(
                nn.Dropout(0.4),
                nn.Linear(2 * hiddendim + ifeatdim, hiddendim),
                nn.ReLU(), 
                nn.Dropout(0.2)
            )
            self.output_layer = nn.Linear(hiddendim, odim) 
        else:
            self.pre_output_block1 = nn.Sequential(
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
            self.output_layer1 = nn.Linear(hiddendim, 1)  
            self.pre_output_block2 = nn.Sequential(
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
            self.output_layer2 = nn.Linear(hiddendim, 1)     
            self.pre_output_block3 = nn.Sequential(
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
            self.output_layer3 = nn.Linear(hiddendim, 1)     
            self.pre_output_block4 = nn.Sequential(
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
            self.output_layer4 = nn.Linear(hiddendim, 1) 

        for param in self.textclassifier.parameters():
            param.requires_grad = False
        for param in self.imageclassifier.parameters():
            param.requires_grad = False
    def forward(self, nodes, mask, pixel, scores):
        text_logits, text_feature = self.textclassifier(nodes, mask)
        image_logits, image_feature = self.imageclassifier(pixel)

        if self.RMs == 'no_RMs':
            features = torch.cat([text_feature, image_feature], dim=-1)
            if self.task == "taskA":
                emb = self.pre_output_block(features)
                out = self.output_layer(emb)
                return out

            else:
                emb1 = self.pre_output_block1(features)
                out1 = self.output_layer1(emb1)
                emb2 = self.pre_output_block2(features)
                out2 = self.output_layer2(emb2)
                emb3 = self.pre_output_block3(features)
                out3 = self.output_layer3(emb3)
                emb4 = self.pre_output_block4(features)
                out4 = self.output_layer4(emb4)
                out = torch.cat([out1, out2, out3, out4], dim=-1)
                return out
        
        elif self.RMs == 'all_RMs':
            scores = scores
        elif self.RMs == 'text_RMs':
            scores = scores[:, -1].unsqueeze(-1)
        elif self.RMs == 'image_RMs':
            scores = scores[:, :-1]
        features = torch.cat([scores, image_feature, text_feature], dim=-1)    
        if self.task == "taskA":
            emb = self.pre_output_block(features)
            out = self.output_layer(emb)
            return out

        else:
            emb1 = self.pre_output_block1(features)
            out1 = self.output_layer1(emb1)
            emb2 = self.pre_output_block2(features)
            out2 = self.output_layer2(emb2)
            emb3 = self.pre_output_block3(features)
            out3 = self.output_layer3(emb3)
            emb4 = self.pre_output_block4(features)
            out4 = self.output_layer4(emb4)
            out = torch.cat([out1, out2, out3, out4], dim=-1)
            return out
