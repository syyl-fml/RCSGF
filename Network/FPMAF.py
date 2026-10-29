import torch
import torch.nn as nn
import torch.nn.functional as F
from .Gated import GatedConv
from .XBSAM import *
from .SCAM import *

class FPMAF(nn.Module):
    def __init__(self, cnum):
        super().__init__()
        self.conv_branch = nn.Sequential(
            GatedConv(4*cnum, 4*cnum, 3),
            GatedConv(4*cnum, 4*cnum, 3, dilation=2),
            GatedConv(4*cnum, 4*cnum, 3, dilation=4),
            GatedConv(4*cnum, 4*cnum, 3, dilation=8),
            GatedConv(4*cnum, 4*cnum, 3, dilation=16)
        )

        self.attn_branch = nn.Sequential(
            GatedConv(4*cnum, 4*cnum, 3, activation=nn.ReLU())
        )

        self.XBSAM = XBSAM(ksize=3,stride=1,rate=2,fuse_k=3,softmax_scale=10.0)
        self.SCAM  = SCAM(4*cnum, top_k=256, kernel_size=3, temperature=0.25)

        self.attn_post = nn.Sequential(
            GatedConv(4*cnum, 4*cnum, 3),
            GatedConv(4*cnum, 4*cnum, 3)
        )

    def forward(self, pre_feat, mask):

        conv_feat = self.conv_branch(pre_feat)
        attn_feat = self.attn_branch(pre_feat)

        attn_feat1 = self.SCAM(attn_feat, mask)
        attn_feat2 = self.XBSAM(attn_feat, mask)

        attn_feat1 = self.attn_post(attn_feat1)
        attn_feat2 = self.attn_post(attn_feat2)

        fused = torch.cat([conv_feat, attn_feat1, attn_feat2], dim=1)
        return fused


