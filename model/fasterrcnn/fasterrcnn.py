from __future__ import  absolute_import

import torch as t
import numpy as np
from torch import nn
from torch.nn import functional as F

from torchvision.ops import nms

from model.util.bbox_opt import gen_anchor_boxes, shift_anchor_boxes, delta2box
from config.config import running_args
from model.fasterrcnn.region_proposal_network import RegionProposalNetwork
from model.fasterrcnn.roi_header import RoIHeader
from model.proposal.anchor_target_creator import AnchorTargetCreator
from model.proposal.proposal_target_creator import ProposalTargetCreator

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
    in_weight[(gt_label > 0).view(-1, 1).expand_as(in_weight)] = 1  #remove cuda for cpu running
    loc_loss = _smooth_l1_loss(pred_loc, gt_loc, in_weight.detach(), sigma)
    # Normalize by total number of negtive and positive rois.
    loc_loss /= ((gt_label >= 0).sum().float()) # ignore gt_label==-1 for rpn_loss
    return loc_loss



class FasterRCNN(nn.Module):
    def __init__(self, n_fg_class, backbone, classifier):
        super(FasterRCNN, self).__init__()
        # img information extractor, which is always vgg16, vgg19, resnet51
        self.backbone = backbone
        
        # classifier, accept roi_width*roi_height resolution feature map, to generate score & reg_loc for every *n_fg_class+1* classes(1 for background)
        self.classifier = classifier
        
        # the rpn network
        self.rpn = RegionProposalNetwork(input_channels=512, mid_channels=512, n_per_anchor=running_args.n_base_anchors_num)
        
        # the roi header network
        # the class including the bg
        self.head = RoIHeader(n_class=n_fg_class+1, roi_size=running_args.roi_size, spatial_scale=running_args.spatial_scale, classifier=classifier)
        
        # anchor target creator for training rpn network
        # its task is to choose *n_sample*(256) sample from anchor boxes
        self.anchor_target_creator = AnchorTargetCreator(n_sample=running_args.n_sample, pos_ratio=running_args.pos_ratio, \
                                                         neg_iou_thresh=running_args.neg_iou_thresh, pos_iou_thresh=running_args.pos_iou_thresh)
        
        # proposal target creator for training roi head network
        # its task is to choose *n_sample*(128) sample from proposal rois which is generated from rpn's proposal creator
        self.proposal_target_creator = ProposalTargetCreator(n_sample=running_args.n_roi_sample, pos_ratio=running_args.pos_roi_ratio, \
                                                          pos_iou_thresh=running_args.pos_roi_iou_thresh, neg_iou_thresh_hi=running_args.neg_iou_thresh_hi, \
                                                          neg_iou_thresh_lo=running_args.neg_iou_thresh_lo, loc_normalize_mean=running_args.loc_normalize_mean, \
                                                          loc_normalize_std=running_args.loc_normalize_std)
        
        # base anchor boxes, with shape(9,4)
        self.base_anchor_boxes = gen_anchor_boxes()

        # class number(add one to label background)
        self.n_class = n_fg_class+1
 
        # set thresh value for predict
        self.nms_thresh = 0.3
        self.score_thresh = 0.7

        # feat stride
        self.feat_stride = running_args.feat_stride

    def forward(self, x, gt_boxes, gt_labels, scale):
        # only support batch size == 1
        assert x.shape[0] == 1, 'we only support batch size=1'
        # image size
        img_size = (x.shape[2], x.shape[3]) # height & width
        # feature extract
        feat = self.backbone(x)

        # get shifted anchor boxes for every point in feature map
        n, ch, h, w = feat.shape
        pre_defined_anchor_boxes = shift_anchor_boxes(self.base_anchor_boxes, h, w, self.feat_stride) # shape(n, 4), n=h*w*9

        # rpn, 
        # rpn_score & rpn_reg_loc with shape (n,1) & (n,2), n=feat.h*feat.w*9
        # rois with shape (n,4), n is at most 6000(at train mode) or 300(at test mode), rois is np.array
        rpn_score, rpn_reg_loc, rois = self.rpn(feat, pre_defined_anchor_boxes, img_size, scale)
        # note batch size == 1
        rpn_score = rpn_score[0]
        rpn_reg_loc = rpn_reg_loc[0]
        rois = rois[0]

        # anchor target creator
        if self.training:
            # loc shape is still n*4, labels shape is still n*1
            # n's typical value is w//16*h//16*9, w&h is input image size, 16 is scaler, 9 is anchor box number
            # ***but note, there are only 256 sample is useful***, except that, in other position the value is -1 for label, 0 for loc
            # gt_boxes' shape [0] == 1, we only support one batch, loc & labels with type of np.array
            loc, labels = self.anchor_target_creator(pre_defined_anchor_boxes, gt_boxes[0].numpy(), img_size)
            # the type of loc & labels is np.array, so we need to change it into tensor
            loc = t.from_numpy(loc).cuda() if t.cuda.is_available() else t.from_numpy(loc)
            labels = t.from_numpy(labels).long().cuda() if t.cuda.is_available() else t.from_numpy(labels).long()
            # rpn loss
            # score cross entropy loss, ignore index = -1, labels only contain 0(bg),1(fg),-1(ignore)
            rpn_score_loss = F.cross_entropy(rpn_score, labels, ignore_index=-1)
            # smooth l1 loss
            rpn_reg_loss = smooth_l1_loss(rpn_reg_loc, loc, labels, running_args.rpn_sigma)
        else:
            rpn_score_loss = 0
            rpn_reg_loss = 0
        
        # proposal target creator, all its outputs 'type are np.array
        # choose samples for roi head network
        # sample_rois 128*4(xy,xy), rois
        # gt_sample_locs: 128*4(xywh scale & offset), delta between sample rois and gt boxes 
        # gt_sample_labels: 128*1 labels including background as 0
        # gt_sample_roi_indices: 128*1, the indices for roi, default is 0
        if self.training:
            sample_rois, gt_sample_locs, gt_sample_labels, gt_sample_roi_indices = self.proposal_target_creator(rois, gt_boxes[0].numpy(), gt_labels[0].numpy())
        else:
            sample_rois = rois
            gt_sample_roi_indices = np.zeros(sample_rois.shape[0], dtype=np.float32)
        # head network output, classes
        # roi_reg_locs shape:[128,classes*4]
        # roi_scores shape:[128,classes]
        sample_rois = t.from_numpy(sample_rois).cuda() if t.cuda.is_available() else t.from_numpy(sample_rois)
        gt_sample_roi_indices = t.from_numpy(gt_sample_roi_indices).cuda() if t.cuda.is_available() else t.from_numpy(gt_sample_roi_indices)
        roi_reg_locs, roi_scores = self.head(feat, sample_rois, gt_sample_roi_indices)
        
        if self.training:
            gt_sample_labels = t.from_numpy(gt_sample_labels).long().cuda() if t.cuda.is_available() else t.from_numpy(gt_sample_labels).long()
            
            n = roi_reg_locs.shape[0]
            roi_locs = roi_reg_locs.view(n, -1, 4)[:,:,[1,0,3,2]]
            roi_locs = roi_locs[t.arange(n).cuda() if t.cuda.is_available() else t.arange(n), gt_sample_labels].contiguous()
            gt_sample_locs = t.from_numpy(gt_sample_locs).cuda() if t.cuda.is_available() else t.from_numpy(gt_sample_locs)

            roi_reg_loss = smooth_l1_loss(roi_locs, gt_sample_locs, gt_sample_labels, running_args.roi_sigma)
            roi_score_loss = F.cross_entropy(roi_scores, gt_sample_labels)
        else:
            roi_score_loss = 0
            roi_reg_loss = 0

        return rpn_score_loss, rpn_reg_loss, roi_score_loss, roi_reg_loss, roi_reg_locs, roi_scores, sample_rois  #, gt_sample_locs, gt_sample_labels

    def _suppress(self, raw_cls_bbox, raw_prob):
        bbox = list()
        label = list()
        score = list()
        # skip cls_id = 0 because it is the background class
        for one_cls in range(1, self.n_class):
            # extract the one_cls's boxes & scores
            cls_bbox_l = raw_cls_bbox[:, one_cls, :]
            prob_l = raw_prob[:, one_cls]
            # get the target whose iou bigger than thresh
            mask = prob_l > self.score_thresh
            cls_bbox_l = cls_bbox_l[mask]
            prob_l = prob_l[mask]
            # do nms
            keep = nms(cls_bbox_l, prob_l, self.nms_thresh)

            bbox.append(cls_bbox_l[keep])
            label.append(t.tensor(one_cls - 1).repeat(len(keep)))  # set class label index back to 0 based, 
            score.append(prob_l[keep])

        bbox = t.cat(bbox, dim=0).float()
        label = t.cat(label, dim=0).int()
        score = t.cat(score, dim=0).float()
        return bbox, label, score

    def predict(self, img, gt_boxes, gt_labels, scale, present='evaluate'):
        if present == 'visualize':
            self.nms_thresh = 0.3
            self.score_thresh = 0.7
        elif present == 'evaluate':
            self.nms_thresh = 0.3
            self.score_thresh = 0.05

        n, c, h, w = img.shape
        assert n==1, 'predict function only support one image at once'
        # roi_reg_locs shape [128,(classes+1)*4]
        # roi_scores shape [128, classes+1]
        # sample_rois shape [128, 4]
        _, _, _, _, roi_reg_locs, roi_scores, sample_rois = self(img, gt_boxes, gt_labels, scale)

        loc_std = t.tensor(running_args.loc_normalize_std).repeat(self.n_class)
        loc_mean = t.tensor(running_args.loc_normalize_mean).repeat(self.n_class)
        roi_reg_locs = roi_reg_locs*loc_std+loc_mean
        roi_reg_locs = roi_reg_locs.view(-1, 4)[:,[1,0,3,2]]

        rois = sample_rois.view(sample_rois.shape[0], 1, 4).expand((sample_rois.shape[0], self.n_class, 4))
        rois = rois.contiguous().view(-1, 4)

        pred_boxes = delta2box(rois.numpy(), roi_reg_locs.numpy())  # shape [sample_roi.shape[0]*n_class, 4], note n_class=fg_class+1
        pred_boxes[:, 0::2] = (pred_boxes[:, 0::2]).clip(min=0, max=w)
        pred_boxes[:, 1::2] = (pred_boxes[:, 1::2]).clip(min=0, max=h)
        pred_boxes = pred_boxes.reshape(-1, self.n_class, 4)
        pred_boxes = t.from_numpy(pred_boxes).cuda() if t.cuda.is_available() else t.from_numpy(pred_boxes)

        pred_socres = F.softmax(roi_scores, dim=1)
        
        bbox, label, score = self._suppress(pred_boxes, pred_socres)
        
        return bbox, label, score






        