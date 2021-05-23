from __future__ import  absolute_import

import torch as t
import numpy as np
from torch import nn
from torch.nn import functional  as F
from config.config import running_args

class RegionProposalNetwork(nn.Module):
    def __init__(self, input_channels, mid_channels, n_per_anchor):
        super(RegionProposalNetwork, self).__init__()

        self.conv = nn.Conv2d(input_channels, mid_channels, 3, 1, 1)
        self.score = nn.Conv2d(mid_channels, n_per_anchor * 2, 1, 1, 0)
        self.loc = nn.Conv2d(mid_channels, n_per_anchor * 4, 1, 1, 0)

        # parameter init
        normal_init(self.conv, 0, 0.01)
        normal_init(self.score, 0, 0.01)
        normal_init(self.loc, 0, 0.01)

    def forward(self, x):
        n = x.shape[0]   # assert n==1
        feat = F.relu(self.conv(x))
        rpn_score_digits = self.score(feat)
        rpn_loc_delta = self.loc(feat)
        
        # get score for every anchor box, here the score is only about fg & bg
        rpn_score_digits = rpn_score_digits.permute(0, 2, 3, 1).contiguous().view(n, -1, 2)
        # [:,:,[1,0,3,2]]  here is somthing needed to note
        # if using [:,:,[1,0,3,2]], we assume that the network generates coordinates is like [y1, x1, y2 ,x2]
        # if not, we assume the network generates coordinates is like [x1, y1, x2 ,y2]
        # it seems no differences between the two operation
        rpn_loc_delta = rpn_loc_delta.permute(0, 2, 3, 1).contiguous().view(n, -1, 4)  #[:,:,[1,0,3,2]]
        rpn_score = F.softmax(rpn_score_digits, dim=2)

        # rpn_score & rpn_reg_loc with type of tensor, rois with type of np.array
        # with shape (n,-1, 2 or 4)
        return rpn_score_digits, rpn_score, rpn_loc_delta


def rpn_to_proposal(rpn, proposal_creator, X, anchor_boxes, img_size, scale, is_training):
    rpn_digits = []
    rpn_scores = []
    rpn_loc_deltas = []
    for p in X:
        digits, score, delta = rpn(p)
        rpn_digits.append(digits.squeeze(dim=0))
        rpn_scores.append(score.squeeze(dim=0))
        rpn_loc_deltas.append(delta.squeeze(dim=0))
    
    # rpn_scores has shape n*2, where n is the number of anchors
    rpn_scores = t.cat(rpn_scores)
    # rpn_loc_deltas has shape n*4, where n is the number of anchors
    rpn_loc_deltas = t.cat(rpn_loc_deltas)
    # rois has shape n*4, where n is the number of predefined number(training=2000, testing=300)
    rois = proposal_creator(anchor_boxes, rpn_loc_deltas.detach().cpu().numpy(), rpn_scores[:,1].detach().cpu().numpy(), img_size, scale, is_training)
    
    return rpn_scores, rpn_loc_deltas, rois
    
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
