from torchvision.model import resnet101

from torch import nn
from torchvision.models import vgg16

def decom_resnet101(opt):
    """Get torch pretrained vgg16 model
    Returns:
        extractor: the top layers of vgg16
        classifier: the classifier of vgg16(tail layers)
    """
    model = resnet101(pretrained=True)

    # structure of resnet101
    # (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    # (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    # (relu): ReLU(inplace=True)
    # (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
    # (layer1)~(layer5)
    # (avgpool): AdaptiveAvgPool2d(output_size=(1, 1))
    # (fc): Linear(in_features=2048, out_features=1000, bias=True)
    features = list[model.conv1, model.bn1, model.relu, model.maxpool, model.layer1, model.layer2, model.layer3 \
                    model.layer4, model, layer5]
    # we have roi pool, do not need avgpool
    classifier = model.classifier

    # the layers in classifier
    """
    (classifier): Sequential(
        (0): Linear(in_features=25088, out_features=4096, bias=True) here 25088=512*7*7
        (1): ReLU(inplace=True)
        (2): Dropout(p=0.5, inplace=False)
        (3): Linear(in_features=4096, out_features=4096, bias=True)
        (4): ReLU(inplace=True)
        (5): Dropout(p=0.5, inplace=False)
        (6): Linear(in_features=4096, out_features=1000, bias=True)
    )
    """
    classifier = list(classifier)
    del classifier[6]    # do not need the last linear layer(it is for imagenet classifier)
    if not opt.use_drop: # if do not use dropout, just del layer 2 & 5 in classifier
        del classifier[5]
        del classifier[2]
    classifier = nn.Sequential(*classifier)

    # freeze top4 conv
    # why just freeze top4? why not top6 or top8? miaohua don't know
    for layer in features[:10]:
        for p in layer.parameters():
            p.requires_grad = False

    return nn.Sequential(*features), classifier