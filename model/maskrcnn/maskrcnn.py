from __future__ import  absolute_import

import torch as t
import numpy as np
from torch import full, nn
from torch.nn import functional as F
import nms_mh as mh             # my implementaton of nms & calc_iou
from model.util.bbox_opt import gen_pyramid_anchors, delta2box
from data.util import unmold_mask, resize_bbox

from config.config import mask_running_args
from model.maskrcnn.feature_pyramid_network import FeaturePyramidNetwork
from model.maskrcnn.region_proposal_network import RegionProposalNetwork, rpn_to_proposal
from model.maskrcnn.roi_header import RoIHeader, MaskHeader
from model.proposal.proposal_creator import ProposalCreator
from model.proposal.anchor_target_creator import AnchorTargetCreator
from model.proposal.proposal_target_creator_m import ProposalTargetCreator   # maskrcnn version proposal target creator
from functools import wraps 

def nograd(f):
    #@wraps
    def new_f(*args,**kwargs):
        with t.no_grad():
           return f(*args,**kwargs)
    return new_f

def _smooth_l1_loss(x, t, in_weight, sigma):
    sigma2 = sigma ** 2
    diff = in_weight * (x - t)
    abs_diff = diff.abs()
    flag = (abs_diff.data < (1. / sigma2)).float()
    y = (flag * (sigma2 / 2.) * (diff ** 2) +
         (1 - flag) * (abs_diff - 0.5 / sigma2))
    return y.sum()

def smooth_l1_loss(pred_loc, gt_loc, gt_label, sigma=1):
    in_weight = t.zeros(gt_loc.shape).cuda() if t.cuda.is_available() else t.zeros(gt_loc.shape)
    # Localization loss is calculated only for positive rois.
    # NOTE:  unlike origin implementation, 
    # we don't need inside_weight and outside_weight, they can calculate by gt_label
    # in_weight[(gt_label > 0).view(-1, 1).expand_as(in_weight).cuda()] = 1
    if t.cuda.is_available():
        in_weight[(gt_label > 0).view(-1, 1).expand_as(in_weight).cuda()] = 1
    else:
        in_weight[(gt_label > 0).view(-1, 1).expand_as(in_weight)] = 1  #remove cuda for cpu running
    loc_loss = _smooth_l1_loss(pred_loc, gt_loc, in_weight.detach(), sigma)
    # Normalize by total number of negative and positive rois.
    loc_loss /= ((gt_label >= 0).sum().float()) # ignore gt_label==-1 for rpn_loss
    return loc_loss

class MaskRCNN(nn.Module):
    def __init__(self, n_fg_class, backbone, backbone_channels):
        super(MaskRCNN, self).__init__()
        # img information extractor, which could be resnet family
        self.backbone = backbone
        
        # the fpn network
        self.fpn = FeaturePyramidNetwork(backbone_output_channels=backbone_channels, output_channels=256)

        # the rpn network
        self.rpn = RegionProposalNetwork(input_channels=256, mid_channels=512, n_per_anchor=len(mask_running_args.anchor_ratios))
        
        # the roi header network
        # the class including the bg
        self.roi_header = RoIHeader(depth=256, pool_size=mask_running_args.roi_size, num_classes=n_fg_class+1)
        self.mask_header = MaskHeader(depth=256, pool_size=mask_running_args.mask_roi_size, num_classes=n_fg_class+1)

        # proposal creator
        self.proposal_creator = ProposalCreator(mask_running_args.pre_train_num, mask_running_args.post_train_num, mask_running_args.pre_test_num, \
                                                mask_running_args.post_test_num, mask_running_args.min_roi_size, mask_running_args.proposal_nms_thresh,\
                                                mask_running_args.skip_small_obj)  # wether to skip small obj

        # anchor target creator for training rpn network
        # its task is to choose *n_sample*(256) sample from anchor boxes
        self.anchor_target_creator = AnchorTargetCreator(n_sample=mask_running_args.n_sample, pos_ratio=mask_running_args.pos_ratio, \
                                                         neg_iou_thresh=mask_running_args.neg_iou_thresh, pos_iou_thresh=mask_running_args.pos_iou_thresh,\
                                                         loc_normalize_mean=mask_running_args.loc_normalize_mean, loc_normalize_std=mask_running_args.loc_normalize_std)
        
        # proposal target creator for training roi head network
        # its task is to choose *n_sample*(128) sample from proposal rois which is generated from rpn's proposal creator
        self.proposal_target_creator = ProposalTargetCreator(n_sample=mask_running_args.n_roi_sample, pos_ratio=mask_running_args.pos_roi_ratio, \
                                                          pos_iou_thresh=mask_running_args.pos_roi_iou_thresh, neg_iou_thresh_hi=mask_running_args.neg_iou_thresh_hi, \
                                                          neg_iou_thresh_lo=mask_running_args.neg_iou_thresh_lo, loc_normalize_mean=mask_running_args.loc_normalize_mean, \
                                                          loc_normalize_std=mask_running_args.loc_normalize_std, mask_size=mask_running_args.gt_mask_size)
        

        # class number(add one to label background)
        self.n_class = n_fg_class+1
 
        # set thresh value for predict
        self.nms_thresh = 0.3
        self.score_thresh = 0.7

        # feat stride
        self.feat_stride = mask_running_args.feat_stride

    def forward(self, x, gt_boxes, gt_labels, gt_masks, scale):
        # only support batch size == 1
        assert x.shape[0] == 1, 'we only support batch size=1'
        # image size
        img_size = (x.shape[2], x.shape[3]) # height & width

        # feature extract, here we get p2, p3, p4, p5
        feats = self.backbone(x)

        # feature pyramid network, here we get p2, p3, p4, p5, p6, all of them have 256 channels
        fpn_feats = self.fpn(feats)
        mask_header_feats = fpn_feats[:-1]
        # rpn, 
        # rpn_digits & rpn_reg_loc with shape (n,2) & (n,4), n=feat.h*feat.w*9
        # rpn_rois with shape (n,4), n is at most 2000(at train mode) or 1000(at test mode), rois is np.array
        # base anchor boxes, with shape [R, 4]
        # R=len(ratios)*len(feat_stride)*int(feat_height/anchor_stride)*int(feat_width/anchor_stride)
        self.base_anchor_boxes = gen_pyramid_anchors(mask_running_args.anchor_scales, mask_running_args.anchor_ratios, \
                                                     img_size, mask_running_args.backbone_stride, mask_running_args.anchor_stride)
        rpn_digits, rpn_loc_deltas, rpn_rois = rpn_to_proposal(self.rpn, self.proposal_creator, fpn_feats, self.base_anchor_boxes, img_size, scale, self.training)

        # anchor target creator
        if self.training:
            # loc shape is still n*4, labels shape is still n*1
            # mask rcnn: n's typical value is (feat_height/anchor_stride)*(feat_width/anchor_stride)*15*number of feat map
            # number of feat map is 5(p2, p3, p4, p5, p6), anchor stride is 1(means generat anchor box for every feature map cell, can be other values)
            # different feat map has different feat_height & feat_width       
            # ***but note, there are only 256 sample is useful***, except that, in other position the value is -1 for label, 0 for loc
            # gt_boxes' shape [0] == 1, we only support one batch, loc & labels with type of np.array
            loc_deltas, labels = self.anchor_target_creator(self.base_anchor_boxes, gt_boxes[0].cpu().numpy(), img_size)
            # the type of loc & labels is np.array, so we need to change it into tensor
            loc_deltas = t.from_numpy(loc_deltas).cuda() if t.cuda.is_available() else t.from_numpy(loc_deltas)
            labels = t.from_numpy(labels).long().cuda() if t.cuda.is_available() else t.from_numpy(labels).long()
            # rpn loss
            # score cross entropy loss, ignore index = -1, labels only contain 0(bg),1(fg),-1(ignore)
            rpn_cls_loss = F.cross_entropy(rpn_digits, labels, ignore_index=-1)
            # smooth l1 loss
            rpn_reg_loss = smooth_l1_loss(rpn_loc_deltas, loc_deltas, labels, mask_running_args.rpn_sigma)
        else:
            rpn_cls_loss = 0
            rpn_reg_loss = 0
        
        # proposal target creator, all its outputs 'type are np.array
        # choose samples for roi head network
        # sample_rois 128*4(xy,xy), rois
        # gt_sample_locs: 128*4(xywh scale & offset), delta between sample rois and gt boxes 
        # gt_sample_labels: 128*1 labels including background as 0
        # gt_pooled_masks: 128*28*28
        if self.training:
            sample_rois, gt_sample_locs, gt_sample_labels, gt_pooled_masks, gt_pos_num = \
                self.proposal_target_creator(rpn_rois, gt_boxes[0].cpu().numpy(), gt_labels[0].cpu().numpy(), gt_masks[0].cpu().numpy())
        else:
            sample_rois = rpn_rois

        # head network output, classes
        # roi_reg_locs shape:[128,classes*4]
        # roi_scores shape:[128,classes]
        sample_rois = t.from_numpy(sample_rois).cuda() if t.cuda.is_available() else t.from_numpy(sample_rois)
        roi_scores, _, roi_reg_locs  = self.roi_header(mask_header_feats, sample_rois, img_size)
        # with shape [num of rois, num of cls, 28, 28]
        mask_scores = self.mask_header(mask_header_feats, sample_rois, img_size)

        if self.training:
            # reshape roi locs to n*(class_num+1)*4, plus 1 is for background
            n = roi_reg_locs.shape[0]
            # [:,:,[1,0,3,2]]  here is somthing needed to note
            # if using [:,:,[1,0,3,2]], we assume that the network generates coordinates is like [y1, x1, y2 ,x2]
            # if not, we assume the network generates coordinates is like [x1, y1, x2 ,y2]
            # it seems no differences between the two operation
            roi_locs = roi_reg_locs.view(n, -1, 4) #[:,:,[1,0,3,2]]
            # according to gt labels, pick the gt box from (class_num+1) , the shape will be changed to n*4
            gt_sample_labels = t.from_numpy(gt_sample_labels).long().cuda() if t.cuda.is_available() else t.from_numpy(gt_sample_labels).long()
            gt_sample_locs = t.from_numpy(gt_sample_locs).cuda() if t.cuda.is_available() else t.from_numpy(gt_sample_locs)
            gt_pooled_masks = t.from_numpy(gt_pooled_masks).float().cuda() if t.cuda.is_available() else t.from_numpy(gt_pooled_masks).float()

            roi_locs = roi_locs[t.arange(n).cuda() if t.cuda.is_available() else t.arange(n), gt_sample_labels].contiguous()
            mask_scores = mask_scores[t.arange(n).cuda() if t.cuda.is_available() else t.arange(n), gt_sample_labels].contiguous()

            roi_reg_loss = smooth_l1_loss(roi_locs, gt_sample_locs, gt_sample_labels, mask_running_args.roi_sigma)
            roi_cls_loss = F.cross_entropy(roi_scores, gt_sample_labels)
            mask_score_loss = F.binary_cross_entropy(mask_scores[:gt_pos_num], gt_pooled_masks[:gt_pos_num])
        else:
            roi_cls_loss = 0
            roi_reg_loss = 0
            mask_score_loss = 0

        return rpn_cls_loss, rpn_reg_loss, roi_cls_loss, roi_reg_loss, mask_score_loss, roi_reg_locs, roi_scores, sample_rois, mask_scores
        

    def _suppress(self, raw_cls_bbox, raw_prob, raw_masks):
        bbox = []
        label = []
        score = []
        masks = []
        # skip cls_id = 0 because it is the background class
        for one_cls in range(1, self.n_class):
            # extract the one_cls's boxes & scores
            cls_bbox_l = raw_cls_bbox[:, one_cls, :]
            prob_l = raw_prob[:, one_cls]
            mask_l = raw_masks[:, one_cls]
            # get the target whose iou bigger than thresh
            mask = prob_l > self.score_thresh
            cls_bbox_l = cls_bbox_l[mask]
            prob_l = prob_l[mask]
            # do nms
            keep = mh.nms(cls_bbox_l.cpu().numpy(), prob_l.cpu().numpy(), self.nms_thresh)

            bbox.append(cls_bbox_l[keep])
            label.append(t.tensor(one_cls - 1).repeat(len(keep)))  # set class label index back to 0 based, 
            score.append(prob_l[keep])
            masks.append(mask_l[keep])

        bbox = t.cat(bbox, dim=0).float()
        label = t.cat(label, dim=0).int()
        score = t.cat(score, dim=0).float()
        masks = t.cat(masks, dim=0).float()
        return bbox, label, score, masks
    
    def _format_rois(self, sample_rois, roi_reg_locs, img_size):
        # normalized back
        loc_std = t.tensor(mask_running_args.loc_normalize_std)
        loc_mean = t.tensor(mask_running_args.loc_normalize_mean)
        if t.cuda.is_available():
            loc_std, loc_mean = loc_std.cuda(), loc_mean.cuda()
        roi_reg_locs = roi_reg_locs*loc_std+loc_mean

        # [:,:,[1,0,3,2]]  here is somthing needed to note
        # if using [:,:,[1,0,3,2]], we assume that the network generates coordinates is like [y1, x1, y2 ,x2]
        # if not, we assume the network generates coordinates is like [x1, y1, x2 ,y2]
        # it seems no differences between the two operation
        roi_reg_locs = roi_reg_locs.view(-1, 4) #[:,[1,0,3,2]]
        # expand the sample_rois' shape from n*4 to n*n_class*4
        rois = sample_rois.view(sample_rois.shape[0], 1, 4).expand((sample_rois.shape[0], self.n_class, 4))
        rois = rois.contiguous().view(-1, 4)
        # shape [sample_roi.shape[0]*n_class, 4], note n_class=fg_class+1
        pred_boxes = delta2box(rois.cpu().numpy(), roi_reg_locs.cpu().numpy())
        pred_boxes[:, 0::2] = (pred_boxes[:, 0::2]).clip(min=0, max=img_size[1])
        pred_boxes[:, 1::2] = (pred_boxes[:, 1::2]).clip(min=0, max=img_size[0])
        pred_boxes = pred_boxes.reshape(-1, self.n_class, 4)
        pred_boxes = t.from_numpy(pred_boxes).cuda() if t.cuda.is_available() else t.from_numpy(pred_boxes)

        return pred_boxes

    def _format_result(self, pred_boxes, pred_scores, mask_scores, scale, img_size, present):
        # do nms for every class
        bbox, label, score, masks = self._suppress(pred_boxes, pred_scores, mask_scores)

        # resize box & mask to original image size if in evaluate mode
        h, w = img_size
        if present == 'evaluate':
            img_size = (int(h/scale), int(w/scale))
        full_masks = []
        scaled_boxes = []
        for one_box, one_mask in zip(bbox, masks):
            if present == 'evaluate':
                one_box = resize_bbox(one_box.numpy(), (h,w), img_size)
                scaled_boxes.append(t.from_numpy(one_box))
            mask = unmold_mask(one_mask, one_box, img_size)
            full_masks.append(t.from_numpy(mask))
        if len(full_masks) > 0:
            full_masks = t.stack(full_masks, dim=0)
        else:
            full_masks = t.tensor(full_masks)
        if len(scaled_boxes) > 0:
            bbox = t.stack(scaled_boxes, dim=0)
        return bbox, label, score, full_masks

    def _set_thresh(self, present):
        if present == 'visualize':
            self.nms_thresh = 0.3
            self.score_thresh = 0.7
        elif present == 'evaluate':
            self.nms_thresh = 0.3
            self.score_thresh = 0.05

    @nograd
    def predict(self, img, gt_boxes, gt_labels, gt_masks, scale, present='evaluate'):
        # assert for one batch
        n, c, h, w = img.shape
        assert n==1, 'predict function only support one image at once'

        # set thresh
        self._set_thresh(present)

        # evaluate, note: 128 is a parameter and can be changed, 28*28 the mask size
        # roi_reg_locs shape [128,(classes+1)*4]
        # roi_scores shape [128, classes+1]
        # sample_rois shape [128, 4]
        # mask_scores shape [128,(classes+1),28,28]
        _, _, _, _, _, roi_reg_locs, roi_scores, sample_rois, mask_scores = self(img, gt_boxes, gt_labels, gt_masks, scale)
       
        # get box(x1,y1,x2,y2) from delta(roi_reg_locs) and sample rois 
        pred_boxes = self._format_rois(sample_rois, roi_reg_locs, (h,w))
        
        # get scores(0~1) for every class
        pred_scores = F.softmax(roi_scores, dim=1)
        
        # do nms for every class, and expand mask, resize boxes to original image size(if in evaluate mode) 
        bbox, label, score, full_masks = self._format_result(pred_boxes, pred_scores, mask_scores, scale, (h,w), present)

        return bbox, label, score, full_masks
