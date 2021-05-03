from __future__ import  absolute_import

import torch as t
from torchvision.ops import nms
from model.util.bbox_opt_t import delta2box
from functools import wraps 

def nograd(f):
    #@wraps
    def new_f(*args,**kwargs):
        with t.no_grad():
           return f(*args,**kwargs)
    return new_f

class ProposalCreator:
    def __init__(self, pre_train_num, post_train_num, pre_test_num, post_test_num, min_roi_size, nms_thresh):
        self.pre_train_num = pre_train_num   # default value is 12000, you can change it in config.py
        self.post_train_num = post_train_num # default value is 6000, you can change it in config.py
        self.pre_test_num = pre_test_num     # default value is 2000, you can change it in config.py
        self.post_test_num = post_test_num   # default value is 300, you can change it in config.py
        self.min_roi_size = min_roi_size     # default value is 16, you can change it in config.py
        self.nms_thresh = nms_thresh         # default value is 0.7, you can change it in config.py

    @nograd
    def __call__(self, anchors, rpn_reg_loc, rpn_score, img_size, scale, is_training):
        '''
        Args:
            anchors(tensor): the pre defined anchor box in feature map, with shape [h*w*9,4],
                                where h & w is feat map size
            rpn_reg_loc(tensor): the rpn loc regression output, the scale & offset from gt boxes
            rpn_score(tensor): the rpn socre output, note its only about foreground, with shape [h*w*9, ]
            img_size(tuple): (height, width)
            scale(scaler):the ratio which real image scales to network input image
            is_training(boolean):wether the network in training mode
        Returns
            rois(tensor): return at most post_train_num(at training) or post_test_num(at eval)  roi
        '''
        # 0. check wether is training
        if is_training:
            pre_num = self.pre_train_num
            post_num = self.post_train_num
        else:
            pre_num = self.pre_test_num
            post_num = self.post_test_num

        # 1. from scale & offset to position 
        proposal_rois = delta2box(anchors, rpn_reg_loc)

        # 2. clip, img_size=(img_height, img_width)
        proposal_rois[:,0] = t.clamp(proposal_rois[:,0], 0, img_size[1])
        proposal_rois[:,2] = t.clamp(proposal_rois[:,2], 0, img_size[1])
        proposal_rois[:,1] = t.clamp(proposal_rois[:,1], 0, img_size[0])
        proposal_rois[:,3] = t.clamp(proposal_rois[:,3], 0, img_size[0])

        # 3. remove small ones, the default min_roi_size is 16
        # note this step is very important, not only it takes off the small roi, but also it gets rid of 
        # some rois whose right-down point is smaller than left top point, this is a very important property for function nms
        ws = proposal_rois[:,2]-proposal_rois[:,0]
        hs = proposal_rois[:,3]-proposal_rois[:,1]
        min_scale = self.min_roi_size*scale
        keep_idx = t.where((hs>=min_scale) & (ws>=min_scale))[0]
        rois = proposal_rois[keep_idx]
        scores = rpn_score[keep_idx]

        # 4. we only need fg score
        sort_keep_idx = scores.argsort(descending=True)
        rois = rois[sort_keep_idx[:pre_num]]
        scores = scores[sort_keep_idx[:pre_num]]

        # 5. nms
        # note  rois (Tensor[N, 4])): boxes to perform NMS on. They 
        # are expected to be in ``(x1, y1, x2, y2)`` format with ``0 <= x1 < x2`` and ``0 <= y1 < y2``.
        #keep_idx = nms(t.from_numpy(rois).cuda() if t.cuda.is_available() else t.from_numpy(rois), \
        #               t.from_numpy(scores).cuda() if t.cuda.is_available() else t.from_numpy(scores), \
        #               self.nms_thresh)
        #rois = rois[keep_idx[:post_num].numpy()]
        keep_idx = nms(rois, scores, self.nms_thresh)
        rois = rois[keep_idx[:post_num]]
        return rois
