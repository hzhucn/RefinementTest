#!/usr/bin/env python
#-*- coding:utf-8 -*-
# Author: Donny You(yas@meitu.com)


import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.functional as F
import numpy as np
import torch.optim as optim
import math


class Downsampler(nn.Module):
    def __init__(self):
        super(Downsampler, self).__init__()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

    def forward(self, x):
        out, index = self.pool1(x)
        return out, index


class NonBottleNeck(nn.Module):
    def __init__(self, inplanes, outplanes, dilated_rate):
        super(NonBottleNeck, self).__init__()
        self.conv1_v = nn.Conv2d(inplanes, outplanes, kernel_size=(3, 1), 
                                 padding=(dilated_rate, 0), dilation=dilated_rate)
        self.conv1_h = nn.Conv2d(outplanes, outplanes, kernel_size=(1, 3), 
                                 padding=(0, dilated_rate), dilation=dilated_rate)
        self.bn1 = nn.BatchNorm2d(outplanes)
        self.relu1_v = nn.ReLU(inplace=True)
        self.relu1_h = nn.ReLU(inplace=True)

        self.conv2_v = nn.Conv2d(outplanes, outplanes, kernel_size=(3, 1),
                                 padding=(dilated_rate, 0), dilation=dilated_rate)
        self.conv2_h = nn.Conv2d(outplanes, outplanes, kernel_size=(1, 3),
                                 padding=(0, dilated_rate), dilation=dilated_rate)

        self.bn2 = nn.BatchNorm2d(outplanes)
        self.relu2_v = nn.ReLU(inplace=True)
        self.relu3 = nn.ReLU(inplace=True)

    def forward(self, x):
        x_in = self.conv1_v(x)
        x_in = self.relu1_v(x_in)
        x_in = self.conv1_h(x_in)
        x_in = self.relu1_h(x_in)
        x_in = self.bn1(x_in)

        x_in = self.conv2_v(x_in)
        x_in = self.relu2_v(x_in)
        x_in = self.conv2_h(x_in)
        x_in = self.bn2(x_in)
        out = x_in + x
        out = self.relu3(out)
        return out


class nxBottleNeck(nn.Module):
    def __init__(self, planes, times):
        super(nxBottleNeck, self).__init__()
        self.times = times
        self.bottleneck1 = NonBottleNeck(planes, planes, 1)
        self.bottleneck2 = NonBottleNeck(planes, planes, 1)
        self.bottleneck3 = NonBottleNeck(planes, planes, 1)
        self.bottleneck4 = NonBottleNeck(planes, planes, 1)
        self.bottleneck5 = NonBottleNeck(planes, planes, 1)

    def forward(self, x):
        for i in range(self.times):
            x = self.bottleneck1(x)

        return x


class CasBottleNeck(nn.Module):
    def __init__(self, planes):
        super(CasBottleNeck, self).__init__()
        self.bottleneck1 = NonBottleNeck(planes, planes, 2)
        self.bottleneck2 = NonBottleNeck(planes, planes, 4)
        self.bottleneck3 = NonBottleNeck(planes, planes, 8)
        self.bottleneck4 = NonBottleNeck(planes, planes, 16)

    def forward(self, x):
        x = self.bottleneck1(x)
        x = self.bottleneck2(x)
        x = self.bottleneck3(x)
        x = self.bottleneck4(x)
        return x


class DUC(nn.Module):
    def __init__(self, inplanes, planes, upscale_factor=2):
        super(DUC, self).__init__()
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(inplanes, planes*4, kernel_size=3,
                              padding=1)
        self.bn = nn.BatchNorm2d(planes*4)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pixel_shuffle(x)
        return x


class Deconv(nn.Module):
    def __init__(self, inplanes, planes):
        super(Deconv, self).__init__()
        self.deconv = nn.ConvTranspose2d(inplanes, planes, 3, stride=2, padding=1, output_padding=1)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.deconv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
    

class Fusion(nn.Module):
    def __init__(self, inplanes, planes):
        super(Fusion, self).__init__()
        self.deconv = nn.ConvTranspose2d(inplanes, planes/2, 3, stride=2, padding=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(planes/2)
        self.relu1 = nn.ReLU()
        self.duc = DUC(inplanes, planes/2)
        self.conv = nn.Conv2d(planes, planes, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        out1 = self.deconv(x)
        out1 = self.bn1(out1)
        out1 = self.relu1(out1)
        out2 = self.duc(x)
        out = torch.cat((out1, out2), 1)
        out = self.conv(out)
        out = self.bn2(out)
        out = self.relu2(out)
        return out


class UP(nn.Module):
    def __init__(self, inplanes, planes):
        super(UP, self).__init__()
        self.unpool = nn.MaxUnpool2d(2, stride=2)
        self.conv = nn.Conv2d(inplanes, planes, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, indices):
        outputs = self.unpool(x, indices = indices)
        outputs = self.conv(outputs)
        outputs = self.bn(outputs)
        outputs = self.relu(outputs)
        return outputs


class GaitUnit(nn.Module):
    def __init__(self, h_planes, l_planes):
        super(GaitUnit, self).__init__()
        self.conv_h_1 = nn.Conv2d(h_planes, h_planes+l_planes, pkernel_size=3, adding=1)
        self.bn_h_1 = nn.BatchNorm2d(h_planes+l_planes)
        self.relu_h_1 = nn.ReLU()
        self.conv_l_1 = nn.Conv2d(l_planes, h_planes+l_planes, pkernel_size=3, adding=1)
        self.bn_l_1 = nn.BatchNorm2d(h_planes+l_planes)
        self.relu_l_1 = nn.ReLU()
        self.deconv_l_1 = Deconv(h_planes+l_planes, h_planes+l_planes)
        

    def forward(self, h_x, l_x):
        h_x = self.conv_h_1(h_x)
        h_x = self.bn_h_1(h_x)
        h_x = self.relu_h_1(x)
        l_x = self.conv_l_1(l_x)
        l_x = self.bn_l_1(l_x)
        l_x = self.relu_l_1(l_x)
        l_x = self.deconv_l_1(l_x)
        x = h_x * l_x
        return x


class RefineUnit(nn.Module):
    def __init__(self, m_planes, r_planes, outplanes):
        super(RefineUnit, self).__init__()
        self.conv1 = nn.Conv2d(m_planes, r_planes, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(r_planes)
        self.relu1 = nn.ReLU()
        self.deconv1 = nn.ConvTranspose2d(r_planes*2, outplanes, 3, stride=2, padding=1, output_padding=1)

    def forward(self, m_x, r_x):
        m_x = self.conv1(m_x)
        m_x = self.bn1(m_x)
        m_x = self.relu1(x)
        x = torch.cat((m_x, r_x), 1)
        out = self.deconv1(x)
        return out


class MaskUnit(nn.Module):
    def __init__(self):
        super(MaskUnit, self).__init__()
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x_1 = -x
        x_1 = self.pool1(x_1)
        x_2 = self.pool2(x)
        x = x_1 + x_2
        return x


class DotUnit(nn.Module):
    def __init__(self, num_classes):
        super(DotUnit, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(num_classes, num_classes, 3, stride=2, padding=1, output_padding=1)
        self.softmax1 = nn.Softmax2d()
        self.maskunit1 = MaskUnit()

    def forward(self, l_x, h_x):
        l_x = self.deconv1(l_x)
        l_x_1 = self.softmax1(l_x)
        l_x_1 = self.maskunit1(l_x_1)
        x = l_x_1 * h_x
        x = l_x + x
        return x


class FCN(nn.Module):
    def __init__(self, num_classes):
        super(FCN, self).__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.downsampler1 = Downsampler()
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.downsampler2 = Downsampler() 
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()
        self.bottleneck_5 = nxBottleNeck(128, 5)
        self.downsampler3 = Downsampler()

        self.casbottleneck1 = CasBottleNeck(128)
        self.casbottleneck2 = CasBottleNeck(128)

        self.dotunit1 = DotUnit(6)
        self.dotunit2 = DotUnit(6)

        self.deconv1 = nn.ConvTranspose2d(128, self.num_classes, 3, stride=2, padding=1, output_padding=1)

        self.deconv2 = nn.ConvTranspose2d(128, self.num_classes, 3, stride=2, padding=1, output_padding=1)

        self.deconv3 = nn.ConvTranspose2d(64, self.num_classes, 3, stride=2, padding=1, output_padding=1)
        
        self.bottleneck_21 = nxBottleNeck(64, 2)
        self.bottleneck_22 = nxBottleNeck(32, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x, indices1 = self.downsampler1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x_1 = self.relu2(x) 
        x, indices2 = self.downsampler2(x_1)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x_2 = self.bottleneck_5(x)
        x, indices3 = self.downsampler3(x_2)
        x_3 = self.casbottleneck1(x)
        x = self.casbottleneck2(x_3)
        out1 = self.deconv1(x)
        x_2 = self.deconv2(x_2)
        x_1 = self.deconv3(x_1)
        out2 = self.dotunit1(out1, x_2)
        out3 = self.dotunit2(out2, x_1)
        return out1, out2, out3


if __name__ == "__main__":
    pass       
