from torch import nn
from torchvision.model import resnet18, resnet34, resnet50, resnet101, resnet152


class TwoMLPHead(nn.Module):
    """
    Standard heads for Roi Header
    Arguments:
        in_channels (int): number of input channels
        representation_size (int): size of the intermediate representation
    """

    def __init__(self, in_channels, representation_size):
        super(TwoMLPHead, self).__init__()

        self.fc6 = nn.Linear(in_channels, representation_size)
        self.fc7 = nn.Linear(representation_size, representation_size)

    def forward(self, x):
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))

        return x

def decom_resnet(opt):
    """Get torch pretrained resnet model
    Returns:
        extractor: the top layers of resnet conv1, layer1~layer3, note the scale is 1/16
        classifier: the two layers classifier
    """
    assert opt.resnet_layers in [18,34,50,101,152]
    resnet_layers = opt.resnet_layers
    if resnet_layers == 18:
        model = resnet18(pretrained=True)
        out_channels = 256
    if resnet_layers == 34:
        model = resnet34(pretrained=True)
        out_channels = 256
    if resnet_layers == 50:
        model = resnet50(pretrained=True)
        out_channels = 1024
    if resnet_layers == 101:
        model = resnet101(pretrained=True)
        out_channels = 1024
    if resnet_layers == 152:
        model = resnet152(pretrained=True)
        out_channels = 1024

    # structure of resnet101
    # (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    # (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    # (relu): ReLU(inplace=True)
    # (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
    # (layer1)~(layer5)
    # (avgpool): AdaptiveAvgPool2d(output_size=(1, 1))
    # (fc): Linear(in_features=2048, out_features=1000, bias=True)
    features = nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool, model.layer1, model.layer2, model.layer3)
    # we have roi pool, do not need avgpool
    # two layers classifier
    classifier = TwoMLPHead(in_features=out_channels*opt.roi_size*opt.roi_size, out_features=4096, bias=True)

    # freeze parameters of layers
    # here i just freeze parameters to layer1, make model to fine tune layer2, layer3
    # you can change the value of unfreeze_layers from 1 to 3
    unfreezed_layers = 2
    num = len(features)-unfreezed_layers
    for layer in features[:num]:
        for p in layer.parameters():
            p.requires_grad = False

    return features, classifier