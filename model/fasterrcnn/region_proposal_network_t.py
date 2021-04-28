from __future__ import  absolute_import

import torch as t
from torch import nn
from torch.nn import functional  as F
from model.proposal.proposal_creator_t import ProposalCreator
from config.config import running_args

class RegionProposalNetwork(nn.Module):
    def __init__(self, input_channels, mid_channels, n_per_anchor):
        super(RegionProposalNetwork, self).__init__()

        self.conv = nn.Conv2d(input_channels, mid_channels, 3, 1, 1)
        self.score = nn.Conv2d(mid_channels, n_per_anchor * 2, 1, 1, 0)
        self.loc = nn.Conv2d(mid_channels, n_per_anchor * 4, 1, 1, 0)
        self.proposal_creator = ProposalCreator(running_args.pre_train_num, running_args.post_train_num, running_args.pre_test_num, \
                                                running_args.post_test_num, running_args.min_roi_size, running_args.proposal_nms_thresh)

        # parameter init
        normal_init(self.conv, 0, 0.01)
        normal_init(self.score, 0, 0.01)
        normal_init(self.loc, 0, 0.01)

    def forward(self, x, anchor_boxes, img_size, scale):
        n = x.shape[0]   # assert n==1
        feat = F.relu(self.conv(x))
        rpn_score = self.score(feat)
        rpn_reg_loc = self.loc(feat)

        rpn_score = rpn_score.permute(0, 2, 3, 1).contiguous().view(n, -1, 2)
        # [:,:,[1,0,3,2]]  here is somthing needed to note
        # if using [:,:,[1,0,3,2]], we assume that the network generates coordinates is like [y1, x1, y2 ,x2]
        # if not, we assume the network generates coordinates is like [x1, y1, x2 ,y2]
        # it seems no differences between the two operation
        rpn_reg_loc = rpn_reg_loc.permute(0, 2, 3, 1).contiguous().view(n, -1, 4)  #[:,:,[1,0,3,2]]
        rpn_softmax_score = F.softmax(rpn_score, dim=2)

        assert n==1, "we only support batch size=1"
        # the score index is 1, we only need fg
        roi = self.proposal_creator(anchor_boxes, rpn_reg_loc[i], rpn_softmax_score[i][:,1], img_size, scale, self.training)
        rois = rois.unsqueeze(dim=0)
        # rpn_score & rpn_reg_loc & rois with type of t.tensor
        # with shape (n,-1, 2 or 4), though the value of n is 1
        return rpn_score, rpn_reg_loc, rois   

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
