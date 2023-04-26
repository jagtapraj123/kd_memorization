# Ref: https://github.com/cydonia999/VGGFace2-pytorch/blob/master/models/resnet.py

import torch
import torch.nn as nn
import math

__all__ = ["ConvNet", "convnet3", "convnet1"]


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        # bias=False
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        # self.downsample = downsample
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True)
        self.dropout = nn.Dropout2d(0.3)
        self.stride = stride

    def forward(self, x):
        # residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # if self.downsample is not None:
        #     residual = self.downsample(x)

        # out += residual
        out = self.maxpool(out)
        out = self.dropout(out)
        out = self.relu(out)

        return out


class ConvNet(nn.Module):
    def __init__(
        self,
        block: nn.Module,
        num_layers: int,
        num_classes: int = 1000,
        include_top: bool = True,
        connector: int = None,
        connector_sigmoid: bool = False,
        final_activation: bool = False,
    ):
        super(ConvNet, self).__init__()
        self.include_top = include_top
        self.connector = connector
        self.connector_sigmoid = connector_sigmoid
        self.final_activation = final_activation

        # self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # self.bn1 = nn.BatchNorm2d(64)
        
        # self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=3, bias=False)
        # self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)

        # self.layer1 = self._make_layer(block, 64, layers[0])
        # self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        # self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        # self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        inplanes = 3
        outplanes = 32
        layers = []
        layers.append(block(inplanes, outplanes))

        inplanes = outplanes
        outplanes *= 2
        for _ in range(1, num_layers):
            layers.append(block(inplanes, outplanes))
            inplanes = outplanes
            outplanes *= 2

        self.hidden = nn.Sequential(*layers)

        # self.avgpool = nn.AvgPool2d(7, stride=1)
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        if self.connector is not None:
            self.fc_ = nn.Linear(inplanes * (32//(2**num_layers))**2, self.connector)
            if self.connector_sigmoid:
                self.sig = nn.Sigmoid()
            self.fc = nn.Linear(self.connector, num_classes)
        else:
            self.fc = nn.Linear(inplanes * (32//(2**num_layers))**2, num_classes)

        if self.final_activation:
            # self.act = nn.Softmax(dim=1)
            self.act = nn.LogSoftmax(dim=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x: torch.Tensor):
        # # print(x.shape)
        # x = self.conv1(x)
        # # print(x.shape)
        # x = self.bn1(x)
        
        # x = self.conv1(x)
        # x = self.relu(x)
        # x = self.maxpool(x)

        # x = self.layer1(x)
        # x = self.layer2(x)
        # x = self.layer3(x)
        # x = self.layer4(x)
        # # print(x.shape)
        # x = self.avgpool(x)
        # print(x.shape)

        x = self.hidden(x)
        # print(x.shape)

        x = x.view(x.size(0), -1)
        # print(x.shape)

        if self.connector is not None:
            x = self.fc_(x)
            if self.connector_sigmoid:
                x = self.sig(x)

        if not self.include_top:
            return x

        x = self.fc(x)

        if self.final_activation:
            x = self.act(x)

        return x


def convnet5(**kwargs):
    """Constructs a ConvNet-5 model."""
    model = ConvNet(BasicBlock, 5, **kwargs)
    return model

def convnet4(**kwargs):
    """Constructs a ConvNet-4 model."""
    model = ConvNet(BasicBlock, 4, **kwargs)
    return model

def convnet3(**kwargs):
    """Constructs a ConvNet-3 model."""
    model = ConvNet(BasicBlock, 3, **kwargs)
    return model

def convnet2(**kwargs):
    """Constructs a ConvNet-2 model."""
    model = ConvNet(BasicBlock, 2, **kwargs)
    return model

def convnet1(**kwargs):
    """Constructs a ConvNet-1 model."""
    model = ConvNet(BasicBlock, 1, **kwargs)
    return model

