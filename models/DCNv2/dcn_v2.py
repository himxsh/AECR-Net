import torch
from torch import nn
from torchvision.ops import deform_conv2d
import math

class DCN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation=1, deformable_groups=1):
        super(DCN, self).__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding)
        if isinstance(dilation, int):
            dilation = (dilation, dilation)
            
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.deformable_groups = deformable_groups

        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, *kernel_size))
        self.bias = nn.Parameter(torch.empty(out_channels))
        
        channels_ = deformable_groups * 3 * kernel_size[0] * kernel_size[1]
        self.conv_offset_mask = nn.Conv2d(in_channels, channels_, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            nn.init.zeros_(self.bias)
        self.conv_offset_mask.weight.data.zero_()
        self.conv_offset_mask.bias.data.zero_()

    def forward(self, x):
        out = self.conv_offset_mask(x)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        return deform_conv2d(x, offset, self.weight, self.bias, self.stride, self.padding, self.dilation, mask)
