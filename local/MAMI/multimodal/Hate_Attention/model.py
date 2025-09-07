import torch
import math, random
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import copy
from transformers import CLIPTextModel, CLIPVisionModel, CLIPModel
class CLIP_multi(torch.nn.Module):
    def __init__(self, odim, modelname, cachedir, mconfig, type=None):
        torch.nn.Module.__init__(self)
        self.text_encoder = CLIPTextModel.from_pretrained(modelname, cache_dir=cachedir)
        self.image_encoder = CLIPVisionModel.from_pretrained(modelname, cache_dir=cachedir)
        self.image_map_input_dim = self.image_encoder.config.hidden_size
        self.text_map_input_dim = self.text_encoder.config.hidden_size
        self.image_map = nn.Sequential(
            nn.Linear(self.image_map_input_dim, mconfig['d_model']),
            nn.Dropout(0.1),
        )
        self.text_map = nn.Sequential(
            nn.Linear(self.text_map_input_dim, mconfig['d_model']),
            nn.Dropout(0.1),
        )

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=mconfig["d_model"], nhead=mconfig["n_head"])
        self.Fusion = nn.TransformerEncoder(self.encoder_layer, num_layers=mconfig['n_block'])
        self.task = mconfig["task"]
        if mconfig["task"] == 'taskA':
            self.pre_output_block = nn.Sequential(
                nn.Dropout(0.4),
                nn.Linear(mconfig['d_model'], mconfig['d_model']),
                nn.ReLU(), 
                nn.Dropout(0.2)
            )
            self.output_layer = nn.Linear(mconfig['d_model'], odim) 
        else:
            self.pre_output_block1 = nn.Sequential(
                nn.Dropout(0.4),
                nn.Linear(mconfig['d_model'], mconfig['d_model']),
                nn.ReLU(), 
                nn.Dropout(0.2),
                nn.Linear(mconfig['d_model'], mconfig['d_model']),
                nn.ReLU(), 
                nn.Dropout(0.2),
                nn.Linear(mconfig['d_model'], mconfig['d_model']),
                nn.ReLU(), 
                nn.Dropout(0.2),
                nn.Linear(mconfig['d_model'], mconfig['d_model']),
                nn.ReLU(), 
                nn.Dropout(0.2)
            )
            self.output_layer1 = nn.Linear(mconfig['d_model'], 1)  
            self.pre_output_block2 = nn.Sequential(
                nn.Dropout(0.4),
                nn.Linear(mconfig['d_model'], mconfig['d_model']),
                nn.ReLU(), 
                nn.Dropout(0.2),
                nn.Linear(mconfig['d_model'], mconfig['d_model']),
                nn.ReLU(), 
                nn.Dropout(0.2),
                nn.Linear(mconfig['d_model'], mconfig['d_model']),
                nn.ReLU(), 
                nn.Dropout(0.2),
                nn.Linear(mconfig['d_model'], mconfig['d_model']),
                nn.ReLU(), 
                nn.Dropout(0.2)
            )
            self.output_layer2 = nn.Linear(mconfig['d_model'], 1)     
            self.pre_output_block3 = nn.Sequential(
                nn.Dropout(0.4),
                nn.Linear(mconfig['d_model'], mconfig['d_model']),
                nn.ReLU(), 
                nn.Dropout(0.2),
                nn.Linear(mconfig['d_model'], mconfig['d_model']),
                nn.ReLU(), 
                nn.Dropout(0.2),
                nn.Linear(mconfig['d_model'], mconfig['d_model']),
                nn.ReLU(), 
                nn.Dropout(0.2),
                nn.Linear(mconfig['d_model'], mconfig['d_model']),
                nn.ReLU(), 
                nn.Dropout(0.2)
            )
            self.output_layer3 = nn.Linear(mconfig['d_model'], 1)     
            self.pre_output_block4 = nn.Sequential(
                nn.Dropout(0.4),
                nn.Linear(mconfig['d_model'], mconfig['d_model']),
                nn.ReLU(), 
                nn.Dropout(0.2),
                nn.Linear(mconfig['d_model'], mconfig['d_model']),
                nn.ReLU(), 
                nn.Dropout(0.2),
                nn.Linear(mconfig['d_model'], mconfig['d_model']),
                nn.ReLU(), 
                nn.Dropout(0.2),
                nn.Linear(mconfig['d_model'], mconfig['d_model']),
                nn.ReLU(), 
                nn.Dropout(0.2)
            )
            self.output_layer4 = nn.Linear(mconfig['d_model'], 1)     

        for param in self.text_encoder.parameters():
            param.requires_grad = False

        for param in self.image_encoder.parameters():
            param.requires_grad = False

    

    def forward(self, nodes, text_mask, pixel):
        text_features = self.text_encoder(input_ids=nodes,
                                          attention_mask=text_mask).pooler_output#.last_hidden_state  # [batch_size, 77, 768]
        image_features = self.image_encoder(pixel_values=pixel).pooler_output# last_hidden_state

        text_features = self.text_map(text_features)
        image_features = self.image_map(image_features)
        image_features = F.normalize(image_features, p=2, dim=1)
        text_features = F.normalize(text_features, p=2, dim=1)
        
        fused_features = torch.cat((image_features.unsqueeze(1), text_features.unsqueeze(1)), dim=1)
        fused_mask = torch.ones(fused_features.size()[0], fused_features.size()[1]).to(nodes.device)
        #BS, _ = nodes.size()
        fused_mask = (fused_mask == 0.0)
        fused_mask = fused_mask.float()

        
        _, seq_length, _ = fused_features.shape
        mask = nn.Transformer.generate_square_subsequent_mask(seq_length).to(nodes.device)
        outs = self.Fusion(fused_features.permute(1, 0, 2), mask=mask, src_key_padding_mask=fused_mask, is_causal=False)
        
        image_features = outs[0]
        text_features = outs[1]

        #itc_loss = self.ITC_loss(image_features, text_features)
        emb = text_features * image_features #emb = torch.mul(text_emb, image_emb)


        if self.task == "taskA":
            emb = self.pre_output_block(emb)
            out = self.output_layer(emb)
            return out

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
            return out

