import torch
import torch.nn as nn

class GatedConv(nn.Module):
    def __init__(self, img_channels, output_channels, kernel_size, stride=1, dilation=1, activation=nn.ELU()):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.conv = nn.Conv2d(
            img_channels, 
            output_channels * 2,  
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation
        )
        self.sigmoid = nn.Sigmoid()
        self.activation = nn.ELU()

    def forward(self, x):
        x = self.conv(x)
        x, gate = torch.chunk(x, 2, dim=1)
        return self.activation(x) * self.sigmoid(gate)

