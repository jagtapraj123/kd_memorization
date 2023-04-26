import torch
import torch.nn as nn
import math

__all__ = ["ConvNet"]


class ConvNet(nn.Module):
    def __init__(
        self,
        num_classes: int = 105,
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

        self.layer1 = self.make_conv_layer(3, 32, 3, 1, 1)

        self.pool = nn.MaxPool2d(kernel_size=2)

        self.layer2 = self.make_conv_layer(32, 128, 3, 2, 1)
        self.layer2_2 = self.make_conv_layer(128, 64, 3, 2, 1)

        self.layer3 = self.make_conv_layer(64, 256, 3, 2, 1)
        # self.layer3_2 = self.make_conv_layer(128, 128, 3, 2, 3)
        # self.layer3_3 = self.make_conv_layer(128, 64, 3, 1, 3)

        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.layer4 = self.make_conv_layer(256, 128, 3, 2, 1)
        self.layer4_2 = self.make_conv_layer(128, 512, 3, 2, 1)

        self.layer5 = self.make_conv_layer(512, 1024, 2, 2, 1)

        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.layer6 = self.make_conv_layer(1024, 512, 3, 2, 1)
        self.layer6_2 = self.make_conv_layer(512, 2048, 3, 2, 1)

        if self.connector is not None:
            self.fc_ = nn.Linear(2048, self.connector)
            if self.connector_sigmoid:
                self.sig = nn.Sigmoid()
            self.fc = nn.Linear(self.connector, num_classes)
        else:
            self.fc = nn.Linear(2048, num_classes)

        if self.final_activation:
            self.act = nn.Softmax(1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        # self.dropout = nn.Dropout(0.25)

    def make_conv_layer(
        self, in_channels, out_channels, kernel_size=3, stride=1, padding=0
    ):
        """
        Convolution followed by Batch Normalization Layer and ReLU Activation
        """

        return nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),
        )

    # Feed forwad function
    def forward(self, input: torch.Tensor):

        output = self.layer1(input)

        output = self.pool(output)

        output = self.layer2(output)
        output = self.layer2_2(output)

        output = self.layer3(output)

        output = self.pool2(output)

        output = self.layer4(output)
        output = self.layer4_2(output)

        output = self.layer5(output)

        output = self.pool3(output)

        output = self.layer6(output)
        output = self.layer6_2(output)

        output = output.view(-1, 2048)

        if self.connector is not None:
            output = self.fc_(output)
            if self.connector_sigmoid:
                output = self.sig(output)

        if not self.include_top:
            return output

        output = self.fc(output)

        if self.final_activation:
            output = self.act(output)

        return output
