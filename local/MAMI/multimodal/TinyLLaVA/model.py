import torch
import torch.nn as nn
from transformers import LlavaForConditionalGeneration

class LLAVA(torch.nn.Module):
    """
    Tiny-LLaVA model with attention pooling and task-specific heads.

    Parameters
    ----------
    odim : int
        Output dimensionality. For taskA this is typically 1; for taskB, each
        head outputs 1 logit and they are concatenated to size 4.
    modelname : str
        Hugging Face model identifier or local path for CLIP weights.
    mconfig : dict
        Model configuration containing:
            - 'd_model' (int): Hidden size for projection/fusion.
            - 'n_head' (int): Number of attention heads in MHA block.
            - 'n_block' (int): Number of MHA blocks.
            - 'task' (str): Either 'taskA' or 'taskB'.    
    """
    def __init__(self, odim, modelname, mconfig):
        torch.nn.Module.__init__(self)
        # load the pre-trained Tiny-LLaVA model
        self.llava = LlavaForConditionalGeneration.from_pretrained(modelname)

        # single layer feed-forward projector 
        self.reform = nn.Sequential(
                nn.Linear(32064, mconfig['d_model']),
                nn.ReLU(), 
                nn.Dropout(0.2))

        # for task A and B we have different feed-forward classifiers
        self.task = mconfig["task"]
        if mconfig["task"] == 'taskA':
            self.pre_output_block = nn.Sequential(
                nn.Dropout(0.4),
                nn.Linear(mconfig['d_model'], mconfig['d_model']),
                nn.ReLU(), 
                nn.Dropout(0.2)
            )
            self.attention = nn.Sequential(
                nn.Linear(mconfig['d_model'], mconfig['d_model']),
                nn.Tanh(),
                nn.Linear(mconfig['d_model'], 1)
            )
            self.output_layer = nn.Linear(mconfig['d_model'], odim) 
        else:
            # four classifiers for each categories
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
            self.attention1 = nn.Sequential(
                nn.Linear(mconfig['d_model'], mconfig['d_model']),
                nn.Tanh(),
                nn.Linear(mconfig['d_model'], 1)
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
            self.attention2 = nn.Sequential(
                nn.Linear(mconfig['d_model'], mconfig['d_model']),
                nn.Tanh(),
                nn.Linear(mconfig['d_model'], 1)
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
            self.attention3 = nn.Sequential(
                nn.Linear(mconfig['d_model'], mconfig['d_model']),
                nn.Tanh(),
                nn.Linear(mconfig['d_model'], 1)
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
            self.attention4 = nn.Sequential(
                nn.Linear(mconfig['d_model'], mconfig['d_model']),
                nn.Tanh(),
                nn.Linear(mconfig['d_model'], 1)
            )
            self.output_layer4 = nn.Linear(mconfig['d_model'], 1)  

        # the Tiny-LLaVA model parameters are frozen to avoid catastrophic forgetting.
        for param in self.llava.parameters():
            param.requires_grad = False

    def forward(self, input):
        """
        Forward pass through Tiny-LLaVA, attention pooling, and task head(s).

        Parameters
        ----------
        input : dict
            Input dictionary for `LlavaForConditionalGeneration`. Must be compatible
            with `self.llava(**input)`.

        Returns
        -------
        torch.Tensor
            Output logits:
                - (batch_size, odim) for taskA
                - (batch_size, 4) for taskB (concatenated per-category logits)
        """

        # extract features from Tiny-LLaVA model
        outs = self.llava(**input).logits  # [batch_size, 77, 768]   

        # map extracted features with the pre-defined projector  
        outs = self.reform(outs)

        # get logits from different classifiers
        if self.task == "taskA":
            attention_scores = self.attention(outs)
        
            # Apply softmax to get attention weights
            attention_weights = torch.softmax(attention_scores, dim=1)
        
            # Compute weighted sum of hidden states
            weighted_sum = torch.sum(attention_weights * outs, dim=1) 
            emb = self.pre_output_block(weighted_sum)
            out = self.output_layer(emb)
            return out

        else:
            attention_scores1 = self.attention1(outs)
            # Apply softmax to get attention weights
            attention_weights1 = torch.softmax(attention_scores1, dim=1)
            # Compute weighted sum of hidden states
            weighted_sum1 = torch.sum(attention_weights1 * outs, dim=1)        
            emb1 = self.pre_output_block1(weighted_sum1)
            out1 = self.output_layer1(emb1)

            attention_scores2 = self.attention2(outs)
            # Apply softmax to get attention weights
            attention_weights2 = torch.softmax(attention_scores2, dim=1)
            # Compute weighted sum of hidden states
            weighted_sum2 = torch.sum(attention_weights2 * outs, dim=1) 
            emb2 = self.pre_output_block2(weighted_sum2)
            out2 = self.output_layer2(emb2)
       
            attention_scores3 = self.attention3(outs)
            # Apply softmax to get attention weights
            attention_weights3 = torch.softmax(attention_scores3, dim=1)
            # Compute weighted sum of hidden states
            weighted_sum3 = torch.sum(attention_weights3 * outs, dim=1) 
            emb3 = self.pre_output_block3(weighted_sum3)
            out3 = self.output_layer3(emb3)
            
            attention_scores4 = self.attention4(outs)
            # Apply softmax to get attention weights
            attention_weights4 = torch.softmax(attention_scores4, dim=1)
            # Compute weighted sum of hidden states
            weighted_sum4 = torch.sum(attention_weights4 * outs, dim=1) 
            emb4 = self.pre_output_block4(weighted_sum4)
            out4 = self.output_layer4(emb4)
            out = torch.cat([out1, out2, out3, out4], dim=-1)
            return out


