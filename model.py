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


class BottleNeck5(nn.Module):
    def __init__(self, planes):
        super(BottleNeck5, self).__init__()
        self.bottleneck1 = NonBottleNeck(planes, planes, 1)
        self.bottleneck2 = NonBottleNeck(planes, planes, 1)
        self.bottleneck3 = NonBottleNeck(planes, planes, 1)
        self.bottleneck4 = NonBottleNeck(planes, planes, 1)
        self.bottleneck5 = NonBottleNeck(planes, planes, 1)

    def forward(self, x):
        x = self.bottleneck1(x)
        x = self.bottleneck2(x)
        x = self.bottleneck3(x)
        x = self.bottleneck4(x)
        x = self.bottleneck5(x)

        return x


class BottleNeck2(nn.Module):
    def __init__(self, planes):
        super(BottleNeck2, self).__init__()
        self.bottleneck1 = NonBottleNeck(planes, planes, 1)
        self.bottleneck2 = NonBottleNeck(planes, planes, 1)

    def forward(self, x):
        x = self.bottleneck1(x)
        x = self.bottleneck2(x)
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


class Deconv(nn.Module):
    def __init__(self, inplanes, planes):
        super(Deconv, self).__init__()
        self.deconv = nn.ConvTranspose2d(inplanes, planes, 3, stride=2, padding=1, output_padding=1)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()

    def forward(self, x, is_last=False):
        x = self.deconv(x)
        if not is_last:
            x = self.bn(x)
            x = self.relu(x)
        return x


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


class RefineUnit(nn.Module):
    def __init__(self, inplanes, planes):
        super(RefineUnit, self).__init__()
        self.planes = planes
        self.conv1 = nn.Conv2d(inplanes, planes, 1, padding=0)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU()

    def forward(self, x, b_x, branch):
        b_x = self.conv1(b_x)
        b_x = self.bn1(b_x)
        b_x = self.relu1(b_x)
        branch = torch.sum(branch, 1)
        branch = branch.repeat(1, self.planes, 1, 1)
        b_x = b_x * branch
        x = b_x + x
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
        self.bottleneck_5 = BottleNeck5(128)
        self.downsampler3 = Downsampler()

        self.casbottleneck1 = CasBottleNeck(128)
        self.casbottleneck2 = CasBottleNeck(128)

        # self.dotunit1 = DotUnit(6)
        # self.dotunit2 = DotUnit(6)

        self.deconv1 = Deconv(128, 64)
        self.conv1_1 = nn.Conv2d(64, self.num_classes, 1, padding=0)
        self.softmax1 = nn.Softmax2d()
        self.maskunit1 = MaskUnit()
        self.refineunit1 = RefineUnit(128, 64)
        self.bottleneck_21 = BottleNeck2(64)

        self.deconv2 = Deconv(64, 32)
        self.conv2_1 = nn.Conv2d(32, self.num_classes, 1, padding=0)
        self.softmax2 = nn.Softmax2d()
        self.maskunit2 = MaskUnit()
        self.refineunit2 = RefineUnit(64, 32)
        self.bottleneck_22 = BottleNeck2(32)

        self.deconv3 = Deconv(32, self.num_classes)

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
        x = self.casbottleneck1(x)
        x = self.casbottleneck2(x)

        x = self.deconv1(x)
        out1 = self.conv1_1(x)
        branch_1 = self.softmax1(out1)
        branch_1 = self.maskunit1(branch_1)
        x = self.refineunit1(x, x_2, branch_1)
        x = self.bottleneck_21(x)

        x = self.deconv2(x)
        out2 = self.conv2_1(x)
        branch_2 = self.softmax2(out2)
        branch_2 = self.maskunit2(branch_2)
        x = self.refineunit2(x, x_1, branch_2)
        x = self.bottleneck_22(x)
        out3 = self.deconv3(x, True)
        return out1, out2, out3


if __name__ == "__main__":
    pass       
