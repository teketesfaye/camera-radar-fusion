import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet50


class ResNetSSDBackbone(nn.Module):
    """ResNet-based SSD backbone producing 6 multi-scale feature maps."""

    def __init__(self, backbone='resnet18', pretrained=True):
        super().__init__()

        if backbone == 'resnet18':
            base = resnet18(pretrained=pretrained)
            c4_ch, c5_ch = 256, 512
        elif backbone == 'resnet50':
            base = resnet50(pretrained=pretrained)
            c4_ch, c5_ch = 1024, 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Early layers
        self.conv1 = nn.Sequential(base.conv1, base.bn1, base.relu)
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4

        # Channel adapters to match SSD expected dimensions
        self.conv4_3 = nn.Conv2d(c4_ch, 512, kernel_size=1, bias=False)
        self.conv7 = nn.Conv2d(c5_ch, 1024, kernel_size=1, bias=False)
        nn.init.xavier_uniform_(self.conv4_3.weight)
        nn.init.xavier_uniform_(self.conv7.weight)

        # Dilated convolution block (replaces FC6/FC7 from original SSD)
        self.fc6fc7 = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=3, padding=6, dilation=6),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=1),
            nn.ReLU(inplace=True),
        )

        # Extra feature map layers for smaller scales
        self.conv8_2 = self._extra_layer(1024, 256, 512)
        self.conv9_2 = self._extra_layer(512, 128, 256)
        self.conv10_2 = self._extra_layer(256, 128, 256, stride=1, padding=0)
        self.conv11_2 = self._extra_layer(256, 128, 256, stride=1, padding=0)

    def _extra_layer(self, in_ch, mid_ch, out_ch, stride=2, padding=1):
        return nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, kernel_size=3, stride=stride, padding=padding, bias=False),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        c1 = self.conv1(x)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        # Feature map 1: from layer3 output
        feat1 = self.conv4_3(c4)
        feat1 = feat1 * self.conv4_3.weight.mean()

        # Feature map 2: from layer4 + dilated conv
        feat2 = self.fc6fc7(self.conv7(c5))

        # Feature maps 3-6: progressively smaller
        feat3 = self.conv8_2(feat2)
        feat4 = self.conv9_2(feat3)
        feat5 = self.conv10_2(feat4)
        feat6 = self.conv11_2(feat5)

        return [feat1, feat2, feat3, feat4, feat5, feat6]
