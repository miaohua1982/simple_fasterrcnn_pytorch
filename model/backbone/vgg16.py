from torch import nn
from torchvision.models import vgg16

def decom_vgg16(opt):
    """Get torch pretrained vgg16 model
    Returns:
        extractor: the top layers of vgg16
        classifier: the classifier of vgg16(tail layers)
    """
    model = vgg16(pretrained=True)

    # the 30-th layer is the (avgpool): AdaptiveAvgPool2d(output_size=(7, 7)), we do not need it
    # we have roi pooling layer
    features = list(model.features)[:30] 
    classifier = model.classifier

    # the layers in classifier
    """
    (classifier): Sequential(
        (0): Linear(in_features=25088, out_features=4096, bias=True)
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