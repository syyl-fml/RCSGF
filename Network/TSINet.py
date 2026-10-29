import torch
import torch.nn as nn
import torch.nn.functional as F
from .Gated import GatedConv
from .FCR import FCR
from .FPMAF import FPMAF

class TSINet(nn.Module):
    def __init__(self, img_ch=1, output_ch=1, cnum=48):
        super().__init__()
        self.stage1 = nn.ModuleDict({
            'conv1': GatedConv(img_ch+2, cnum, 5),
            'conv2_down': GatedConv(cnum, 2*cnum, 3, stride=2),
            'conv3': GatedConv(2*cnum, 2*cnum, 3),
            'conv4_down': GatedConv(2*cnum, 4*cnum, 3, stride=2),
            'conv5': GatedConv(4*cnum, 4*cnum, 3),
            'conv5_down': GatedConv(4*cnum, 6*cnum, 3, stride=2),  
            'conv6': GatedConv(6*cnum, 6*cnum, 3),
            'conv6_1_down': GatedConv(6*cnum, 8*cnum, 3, stride=2),  
            'conv6_1': GatedConv(8*cnum, 8*cnum, 3),
            'conv7_dilate': GatedConv(8*cnum, 8*cnum, 3, dilation=2),
            'conv8_dilate': GatedConv(8*cnum, 8*cnum, 3, dilation=4),
            'conv9_dilate': GatedConv(8*cnum, 8*cnum, 3, dilation=8),
            'conv10_dilate': GatedConv(8*cnum, 8*cnum, 3, dilation=16),
            'conv11': GatedConv(8*cnum, 8*cnum, 3),            
            'conv12': GatedConv(8*cnum, 8*cnum, 3),
            'FCR1': FCR(8*cnum,3),  # 用于1/8尺度
            'deconv13_up': self._make_deconv(8*cnum, 6*cnum),         
            'conv14': GatedConv(6*cnum, 6*cnum, 3),
            'FCR2': FCR(6*cnum,3),  # 用于1/2尺度
            'deconv15_up': self._make_deconv(6*cnum, 4*cnum),
            'conv15': GatedConv(4*cnum, 4*cnum, 3),
            'FCR3': FCR(4*cnum,5),  # 原始尺度，通道数 cnum
            'deconv16_up': self._make_deconv(4*cnum, 2*cnum),
            'conv16': GatedConv(2*cnum, 2*cnum, 3),
            'FCR4': FCR(2*cnum,7),  # 用于1/2尺度
            'deconv16_1_up': self._make_deconv(2*cnum, cnum),
            'conv16_1': GatedConv(cnum, cnum//2, 3),         
            'conv17': nn.Conv2d(cnum//2, output_ch, 3, padding=1),
        })
        
        self.stage2 = nn.ModuleDict({
            'pre_conv1': GatedConv(output_ch, cnum, 5),
            'pre_conv2_down': GatedConv(cnum, 2*cnum, 3, stride=2),
            'pre_conv3': GatedConv(2*cnum, 2*cnum, 3),
            'pre_conv4_down': GatedConv(2*cnum, 4*cnum, 3, stride=2),
            'pre_conv5': GatedConv(4*cnum, 4*cnum, 3),
            
            'FPMAF': FPMAF(cnum),

            'fusion_conv1': GatedConv(12*cnum, 8*cnum, 3),
            'fusion_conv2': GatedConv(8*cnum, 4*cnum, 3),
            'deconv3_up': self._make_deconv(4*cnum, 2*cnum),
            'fusion_conv4': GatedConv(2*cnum, 2*cnum, 3),
            'deconv5_up': self._make_deconv(2*cnum, cnum),
            'fusion_conv6': GatedConv(cnum, cnum//2, 3),
            'fusion_conv7': GatedConv(cnum//2, cnum//2, 3),
            'fusion_conv8': nn.Conv2d(cnum//2, output_ch, 3, padding=1)
        })
        self.tanh = nn.Tanh()

    def _make_deconv(self, img_ch, output_ch):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            GatedConv(img_ch, output_ch, 3)
        )

    def resize_mask_like(self, mask, ref_tensor):
        return F.interpolate(mask.float(), size=ref_tensor.shape[2:], mode='nearest')

    def forward(self, x, mask):
        ones = torch.ones_like(x)
        xin = x
        x = torch.cat([x, ones, ones*mask.float()], dim=1)  

        s1 = self.stage1
        x = s1['conv1'](x)
        x = s1['conv2_down'](x) 
        x = s1['conv3'](x)
        enc1 = x
        x = s1['conv4_down'](x)
        x = s1['conv5'](x)
        enc2 = x  
        x = s1['conv5_down'](x)
        x = s1['conv6'](x)
        enc3 = x  
        x = s1['conv6_1_down'](x)
        x = s1['conv6_1'](x)
        enc4 = x 
        x = s1['conv7_dilate'](x)  
        x = s1['conv8_dilate'](x)  
        x = s1['conv9_dilate'](x)  
        x = s1['conv10_dilate'](x) 
        x = s1['conv11'](x)
        x = s1['conv12'](x)
        x = s1['FCR1'](enc4, x) 
        x = s1['deconv13_up'](x)   
        x = s1['conv14'](x)
        x = s1['FCR2'](enc3, x)
        x = s1['deconv15_up'](x)  
        x = s1['conv15'](x)
        x = s1['FCR3'](enc2, x) 
        x = s1['deconv16_up'](x)  
        x = s1['conv16'](x)
        x = s1['FCR4'](enc1, x) 
        x = s1['deconv16_1_up'](x)  
        x = s1['conv16_1'](x)
        stage1_out = self.tanh(s1['conv17'](x))


        x_stage2 = stage1_out * mask.float() + xin * (1 - mask.float())
        x_pre = self.stage2['pre_conv1'](x_stage2)
        x_pre = self.stage2['pre_conv2_down'](x_pre)
        x_pre = self.stage2['pre_conv3'](x_pre)     
        x_pre = self.stage2['pre_conv4_down'](x_pre)
        pre_feat = self.stage2['pre_conv5'](x_pre)

        FPMAF_feat = self.stage2['FPMAF'](pre_feat, mask)

        x = self.stage2['fusion_conv1'](FPMAF_feat)
        x = self.stage2['fusion_conv2'](x)
        x = self.stage2['deconv3_up'](x)
        x = self.stage2['fusion_conv4'](x)
        x = self.stage2['deconv5_up'](x)
        x = self.stage2['fusion_conv6'](x)
        x = self.stage2['fusion_conv7'](x)
        stage2_out = self.tanh(self.stage2['fusion_conv8'](x))

        return (stage1_out + 1) / 2, (stage2_out + 1) / 2



