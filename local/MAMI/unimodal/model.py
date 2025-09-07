import torch
import torch.nn.functional as F
import torch.nn as nn
from transformers import CLIPTextModel, CLIPVisionModel


class CLIPtext(torch.nn.Module):
    """
    CLIP-based text encoder with task-specific classification heads.

    For task A:
        - A single feed-forward classifier is used.
    For task B:
        - Four independent classifiers are applied, and their outputs are concatenated.
    """
    def __init__(self, odim, modelname, cachedir, hiddendim, task):
        torch.nn.Module.__init__(self)
        # load the pre-trained clip text encoder
        self.cliptext = CLIPTextModel.from_pretrained(modelname, cache_dir=cachedir)
        # get extracted feature embedding dimension
        self.text_map_input_dim = self.cliptext.config.hidden_size
        # the single layer feed-forward projector
        self.text_map = nn.Sequential(
            nn.Linear(self.text_map_input_dim, hiddendim),
            nn.Dropout(0.1),
        )

        # for task A and B we have different feed-forward classifiers
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
            # four classifiers for each categories
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
        """
        Forward pass of the CLIPtext model.

        Parameters
        ----------
        nodes : torch.Tensor
            Tokenized text input (IDs).
        mask : torch.Tensor
            Attention mask for the text input.

        Returns
        -------
        torch.Tensor
            Model output logits:
                - Shape (batch_size, 1) for task A
                - Shape (batch_size, 4) for task B
        """
        x = self.cliptext(nodes, mask).pooler_output
        x = self.text_map(x)
        emb = F.normalize(x, p=2, dim=1)
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
    
class CLIPimage(torch.nn.Module):
    """
    CLIP-based image encoder with task-specific classification heads.

    For task A:
        - A single feed-forward classifier is used.
    For task B:
        - Four independent classifiers are applied, and their outputs are concatenated.
    """
    def __init__(self, odim, modelname, cachedir, hiddendim, task):
        torch.nn.Module.__init__(self)
        # load the pre-trained clip image encoder
        self.clipimage = CLIPVisionModel.from_pretrained(modelname, cache_dir=cachedir)
        # get extracted feature embedding dimension
        self.image_map_input_dim = self.clipimage.config.hidden_size
        # the single layer feed-forward projector
        self.image_map = nn.Sequential(
            nn.Linear(self.image_map_input_dim, hiddendim),
            nn.Dropout(0.1),
        )

        # for task A and B we have different feed-forward classifiers
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
            # four classifiers for each categories
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
        """
        Forward pass of the CLIPimage model.

        Parameters
        ----------
        image : torch.Tensor
            Preprocessed image tensor input.

        Returns
        -------
        torch.Tensor
            Model output logits:
                - Shape (batch_size, 1) for task A
                - Shape (batch_size, 4) for task B
        """
        x = self.clipimage(image).pooler_output
        x = self.image_map(x)
        emb = F.normalize(x, p=2, dim=1)
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