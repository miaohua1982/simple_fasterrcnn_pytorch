import torch as t
from torch import nn

class FeaturePyramidNetwork(nn.Module):
    def __init__(self, backbone_output_channels, output_channels=256):
        super(FeaturePyramidNetwork, self).__init__()

        self.p2_conv = nn.Conv2d(in_channels=backbone_output_channels, out_channels=output_channels, kernel_size=3, stride=1, padding=1)
        self.p3_conv = nn.Conv2d(in_channels=backbone_output_channels, out_channels=output_channels, kernel_size=3, stride=1, padding=1)
        self.p4_conv = nn.Conv2d(in_channels=backbone_output_channels, out_channels=output_channels, kernel_size=3, stride=1, padding=1)
        self.p5_conv = nn.Conv2d(in_channels=backbone_output_channels, out_channels=output_channels, kernel_size=3, stride=1, padding=1)

        self.p6_downsample = nn.MaxPool2d(kernel_size=1, stride=2)

        normal_init(self.p2_conv, 0, 0.01)
        normal_init(self.p3_conv, 0, 0.01)
        normal_init(self.p4_conv, 0, 0.01)
        normal_init(self.p5_conv, 0, 0.01)

    def forward(self, X):
        p2, p3, p4, p5 = X
        p2 = self.p2_conv(p2)
        p3 = self.p3_conv(p3)
        p4 = self.p4_conv(p4)
        p5 = self.p5_conv(p5)
        p6 = self.p6_downsample(p5)

        return p2, p3, p4, p5, p6


def normal_init(m, mean, stddev, truncated=False):
    """
    weight initializer: truncated normal and random normal.
    """
    # x is a parameter
    if truncated:
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
    else:
        m.weight.data.normal_(mean, stddev)
        m.bias.data.zero_()

