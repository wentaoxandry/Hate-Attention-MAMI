import torch
from torch.nn.functional import normalize
import torch.nn.functional as F
import math
from transformers import CLIPTextModel, CLIPVisionModel, CLIPModel
class LayerNorm(torch.nn.LayerNorm):
    """Layer normalization module.
    Args:
        nout (int): Output dim size.
        dim (int): Dimension to be normalized.
    """

    def __init__(self, nout, dim=-1):
        """Construct an LayerNorm object."""
        super(LayerNorm, self).__init__(nout, eps=1e-12)
        self.dim = dim

    def forward(self, x):
        """Apply layer normalization.
        Args:
            x (torch.Tensor): Input tensor.
        Returns:
            torch.Tensor: Normalized tensor.
        """
        if self.dim == -1:
            return super(LayerNorm, self).forward(x)
        return (
            super(LayerNorm, self)
            .forward(x.transpose(self.dim, -1))
            .transpose(self.dim, -1)
        )


class MultiHeadedAttention(torch.nn.Module):
    """Multi-Head Attention layer.
    Args:
        n_head (int): The number of heads.
        n_feat (int): The number of features.
        dropout_rate (float): Dropout rate.
    """

    def __init__(self, n_head, n_feat, dropout_rate):
        """Construct an MultiHeadedAttention object."""
        super(MultiHeadedAttention, self).__init__()
        assert n_feat % n_head == 0
        # We assume d_v always equals d_k
        self.d_k = n_feat // n_head
        self.h = n_head
        self.linear_q = torch.nn.Linear(n_feat, n_feat)
        self.linear_k = torch.nn.Linear(n_feat, n_feat)
        self.linear_v = torch.nn.Linear(n_feat, n_feat)
        self.linear_out = torch.nn.Linear(n_feat, n_feat)
        self.attn = None
        self.dropout = torch.nn.Dropout(p=dropout_rate)

    def forward_qkv(self, query, key, value):
        """Transform query, key and value.
        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).
        Returns:
            torch.Tensor: Transformed query tensor (#batch, n_head, time1, d_k).
            torch.Tensor: Transformed key tensor (#batch, n_head, time2, d_k).
            torch.Tensor: Transformed value tensor (#batch, n_head, time2, d_k).
        """
        n_batch = query.size(0)
        q = self.linear_q(query).view(n_batch, -1, self.h, self.d_k)
        k = self.linear_k(key).view(n_batch, -1, self.h, self.d_k)
        v = self.linear_v(value).view(n_batch, -1, self.h, self.d_k)
        q = q.transpose(1, 2)  # (batch, head, time1, d_k)
        k = k.transpose(1, 2)  # (batch, head, time2, d_k)
        v = v.transpose(1, 2)  # (batch, head, time2, d_k)

        return q, k, v

    def forward_attention(self, value, scores, mask):
        """Compute attention context vector.
        Args:
            value (torch.Tensor): Transformed value (#batch, n_head, time2, d_k).
            scores (torch.Tensor): Attention score (#batch, n_head, time1, time2).
            mask (torch.Tensor): Mask (#batch, 1, time2) or (#batch, time1, time2).
        Returns:
            torch.Tensor: Transformed value (#batch, time1, d_model)
                weighted by the attention score (#batch, time1, time2).
        """
        n_batch = value.size(0)
        if mask is not None:
            mask = mask.unsqueeze(1).eq(0)  # (batch, 1, *, time2)
            min_value = torch.finfo(scores.dtype).min
            # scores = scores.masked_fill(mask, -2 ** 32 + 1)
            scores = scores.masked_fill(mask, min_value)
            self.attn = torch.softmax(scores, dim=-1).masked_fill(
                mask, 0.0
            )  # (batch, head, time1, time2)

        else:
            self.attn = torch.softmax(scores, dim=-1)  # (batch, head, time1, time2)

        p_attn = self.dropout(self.attn)
        x = torch.matmul(p_attn, value)  # (batch, head, time1, d_k)
        x = (
            x.transpose(1, 2).contiguous().view(n_batch, -1, self.h * self.d_k)
        )  # (batch, time1, d_model)

        return self.linear_out(x)  # (batch, time1, d_model)

    def forward(self, query, key, value, mask):
        """Compute scaled dot product attention.
        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).
            mask (torch.Tensor): Mask tensor (#batch, 1, time2) or
                (#batch, time1, time2).
        Returns:
            torch.Tensor: Output tensor (#batch, time1, d_model).
        """
        q, k, v = self.forward_qkv(query, key, value)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        return self.forward_attention(v, scores, mask)

class CLIP_multi(torch.nn.Module):
    def __init__(self, odim, modelname, cachedir):
        torch.nn.Module.__init__(self)
        n_block = 6
        self.clip = CLIPModel.from_pretrained(modelname, cache_dir=cachedir)
        self.textproj = torch.nn.Linear(768, 1024)
        self.imgproj = torch.nn.Linear(1024, 1024)
        if n_block > 1:
            self.multiattblock = True
            self.Fusion = torch.nn.ModuleList()
            for n_layer in range(n_block - 1):
                self.Fusion.append(MultiHeadedAttention(8, 1024, 0.1))
        else:
            self.multiattblock = False
            self.Fusion = MultiHeadedAttention(8, 1024, 0.1)
        self.preout = torch.nn.Sequential(
            torch.nn.Linear(1024 * 2, 1024, bias=True),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(1024, 1024, bias=True),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(1024, 1024, bias=True),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1))
        self.classifier = torch.nn.Linear(1024, odim)

    def forward(self, nodes, text_mask, pixel):
        clip_out = self.clip(nodes, pixel, text_mask, return_loss=False)
        text_emb = self.textproj(clip_out.text_model_output.last_hidden_state)
        image_emb = self.imgproj(clip_out.vision_model_output.last_hidden_state)

        fused_features = torch.cat((image_emb, text_emb), dim=1)
        BS, _ = nodes.size()
        combinedmask = torch.concat((torch.ones(BS, 257).to(fused_features.device), text_mask), dim=-1)
        if self.multiattblock is True:
            for module in self.Fusion:
                fused_features = module(fused_features, fused_features, fused_features,
                                         combinedmask.unsqueeze(1))
        else:
            fused_features = self.Fusion(fused_features, fused_features, fused_features, combinedmask.unsqueeze(1))

        imagefeat = fused_features[:, 0, :]
        text_ems = fused_features[:, 257:, :]
        pooled_output = text_ems[
            torch.arange(text_ems.shape[0], device=text_ems.device),
            nodes.to(dtype=torch.int, device=text_ems.device).argmax(dim=-1),
        ]
        fused_cls_features = torch.cat((pooled_output, imagefeat), dim=1)


        emb = self.preout(fused_cls_features)

        return self.classifier(emb)