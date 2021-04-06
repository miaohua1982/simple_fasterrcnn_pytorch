import torch as t
from torch import nn
from torchvision.ops import RoIPool


class RoIHeader(nn.Module):
    def __init__(self, n_class, roi_size, spatial_scale, classifier):
        # n_class includes the background
        super(RoIHeader, self).__init__()

        self.classifier = classifier
        self.cls_loc = nn.Linear(4096, n_class * 4)
        self.score = nn.Linear(4096, n_class)

        normal_init(self.cls_loc, 0, 0.001)
        normal_init(self.score, 0, 0.01)

        self.n_class = n_class
        self.roi_size = roi_size
        self.spatial_scale = spatial_scale
        self.roi = RoIPool((self.roi_size, self.roi_size), self.spatial_scale)

    def forward(self, x, rois, roi_indices):
        # add batch index to roi, get shape 128*5, first column is batch index, next 4 columns are left-top x&y, right-down x&y
        indices_and_rois = t.cat([roi_indices[:, None], rois], dim=1) 
        # roi pooling
        pool = self.roi(x, indices_and_rois) # pool has shape(128,512,7,7)
        pool = pool.view(pool.size(0), -1)   # change shape to (128, 25088)
        fc7 = self.classifier(pool)          # change shape to (128, 4096)
        roi_cls_locs = self.cls_loc(fc7)     # change shape to (128, classes*4), classes' typical value is num_classes+1, here 1 for background
        roi_scores = self.score(fc7)         # change shape to (128, classes),   classes' typical value is num_classes+1
        return roi_cls_locs, roi_scores


def normal_init(m, mean, stddev, truncated=False):
    """
    weight initalizer: truncated normal and random normal.
    """
    # x is a parameter
    if truncated:
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
    else:
        m.weight.data.normal_(mean, stddev)
        m.bias.data.zero_()
