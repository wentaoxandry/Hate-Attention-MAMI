import torch
import torch.nn.functional as F
import torch.nn as nn
from transformers import CLIPTextModel, CLIPVisionModel

class CLIP_multi(torch.nn.Module):
    """
    Hate-CLIPper model

    Parameters
    ----------
    odim : int
        Output dimensionality. For taskA this is typically 1; for taskB, each
        head outputs 1 logit and they are concatenated to size 4.
    modelname : str
        Hugging Face model identifier or local path for CLIP weights.
    cachedir : str
        Directory used to cache pretrained model files.
    mconfig : dict
        Model configuration containing:
            - 'd_model' (int): Hidden size for projection/fusion.
            - 'n_head' (int): Number of attention heads in MHA block.
            - 'n_block' (int): Number of MHA blocks.
            - 'task' (str): Either 'taskA' or 'taskB'.    
    """
    def __init__(self, odim, modelname, cachedir, mconfig, type=None):
        torch.nn.Module.__init__(self)
        # load the pre-trained CLIP text and vision encoder
        self.text_encoder = CLIPTextModel.from_pretrained(modelname, cache_dir=cachedir)
        self.image_encoder = CLIPVisionModel.from_pretrained(modelname, cache_dir=cachedir)
        
        # create the single feed-forward projectors
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

        # for task A and B we have different feed-forward classifiers
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

        # the CLIP text and vision encoders are frozen to avoid catastrophic forgetting.
        for param in self.text_encoder.parameters():
            param.requires_grad = False

        for param in self.image_encoder.parameters():
            param.requires_grad = False

    def forward(self, nodes, text_mask, pixel):
        """
        Forward the Hate-CLIPper model

        Parameters
        ----------
        nodes : torch.Tensor
            Token IDs for the text encoder. Shape: (batch_size, seq_len).
        text_mask : torch.Tensor
            Attention mask for the text encoder. Shape: (batch_size, seq_len).
        pixel : torch.Tensor
            Preprocessed image tensor for the vision encoder. Typically
            (batch_size, 3, H, W) after CLIP preprocessing.

        Returns
        -------
        torch.Tensor
            Output logits:
                - (batch_size, odim) for taskA
                - (batch_size, 4) for taskB (concatenated per-category logits)
        """

        # extract text and image classification features
        text_features = self.text_encoder(input_ids=nodes,
                                          attention_mask=text_mask).pooler_output#.last_hidden_state  # [batch_size, 77, 768]
        image_features = self.image_encoder(pixel_values=pixel).pooler_output# last_hidden_state

        # map extracted features into identical dimension by projectors
        text_features = self.text_map(text_features)
        image_features = self.image_map(image_features)
        image_features = F.normalize(image_features, p=2, dim=1)
        text_features = F.normalize(text_features, p=2, dim=1)
    
        # multimodal classification features obtained by element-wise multiplication
        emb = text_features * image_features

        # get logits from different classifiers
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

