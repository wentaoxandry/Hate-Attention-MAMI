import torch
import math, random
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import copy
from transformers import LlavaForConditionalGeneration

class LLAVA(torch.nn.Module):
    def __init__(self, odim, modelname, cachedir, mconfig):
        torch.nn.Module.__init__(self)
        self.llava = LlavaForConditionalGeneration.from_pretrained(modelname)
        self.reform = nn.Sequential(
                nn.Linear(32064, mconfig['d_model']),
                nn.ReLU(), 
                nn.Dropout(0.2))

           
        self.pre_output_block = nn.Sequential(
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
        self.output_layer = nn.Linear(mconfig['d_model'], odim)  
           
        self.attention = nn.Sequential(
            nn.Linear(mconfig['d_model'], mconfig['d_model']),
            nn.Tanh(),
            nn.Linear(mconfig['d_model'], 1)
        )

        for param in self.llava.parameters():
            param.requires_grad = False

    def forward(self, input):
        outs = self.llava(**input).logits  # [batch_size, 77, 768]     
        outs = self.reform(outs)

        attention_scores = self.attention(outs)
        
        # Apply softmax to get attention weights
        attention_weights = torch.softmax(attention_scores, dim=1)
        
        # Compute weighted sum of hidden states
        weighted_sum = torch.sum(attention_weights * outs, dim=1) 
        emb = self.pre_output_block(weighted_sum)
        out = self.output_layer(emb)
        return out

        


