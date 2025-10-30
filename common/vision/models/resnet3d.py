import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import copy

__all__ = ['ResNet3D', 'ResNet18_3D', 'ResNet34_3D']


class BasicBlock3D(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet3D(nn.Module):
    def __init__(self, block, num_blocks, num_classes=2, pretrained=False):
        super(ResNet3D, self).__init__()

        self.in_planes = 64

        self.conv1 = nn.Conv3d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.db1 = torchvision.ops.DropBlock3d(p=0.5, block_size=15)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        # self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        # self.dropout = nn.Dropout(0.5)
        # self.fc = nn.Linear(512 * block.expansion, num_classes)

        self._out_features = 512 * block.expansion

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, use_dropblock=True):
    # def forward(self, x):
    #     if use_dropblock:
    #         out = self.db1(x)
    #         out = F.relu(self.bn1(self.conv1(out)))
    #     else:
    #         out = F.relu(self.bn1(self.conv1(x)))

        x = F.relu(self.bn1(self.conv1(x)))
        # print(x.shape)
        if use_dropblock:
            out = self.db1(x)
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        # out = self.avgpool(out)
        # out = out.view(out.size(0), -1)
        # out = self.dropout(out)  # 应用Dropout
        # out = self.fc(out)
        return out


    @property
    def out_features(self) -> int:
        """The dimension of output features"""
        return self._out_features

    def copy_head(self) -> nn.Module:
        """Copy the origin fully connected layer"""
        return copy.deepcopy(self.fc)


def ResNet18_3D(num_classes=2, pretrained=False):
    return ResNet3D(BasicBlock3D, [2, 2, 2, 2], num_classes=num_classes, pretrained=pretrained)

def ResNet34_3D(num_classes=2, pretrained=False):
    return ResNet3D(BasicBlock3D, [3, 4, 6, 3], num_classes=num_classes, pretrained=pretrained)

net = ResNet18_3D(num_classes=2)

