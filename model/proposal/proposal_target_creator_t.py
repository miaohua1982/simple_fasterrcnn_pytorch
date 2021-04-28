from __future__ import  absolute_import

import torch as t
from model.util.bbox_opt_t import delta2box, box2delta
from model.util.iou_t import calc_iou
from functools import wraps 

def nograd(f):
    @wraps
    def new_f(*args,**kwargs):
        with t.no_grad():
           return f(*args,**kwargs)
    return new_f

class ProposalTargetCreator:
    def __init__(self, n_sample, pos_ratio, pos_iou_thresh, neg_iou_thresh_hi, neg_iou_thresh_lo, loc_normalize_mean, loc_normalize_std):
        self.n_sample = n_sample                     # default value is 128, you can change it in config.py
        self.pos_ratio = pos_ratio                   # default value is 0.25, you can change it in config.py
        self.pos_iou_thresh = pos_iou_thresh         # default value is 0.5, you can change it in config.py
        self.neg_iou_thresh_hi = neg_iou_thresh_hi   # default value is 0.5, you can change it in config.py
        self.neg_iou_thresh_lo = neg_iou_thresh_lo   # default value is 0.0, you can change it in config.py
        self.loc_normalize_mean = t.tensor(loc_normalize_mean, dtype=t.float) # default value is [0,0,0,0], you can change it in config.py
        self.loc_normalize_std = t.tensor(loc_normalize_std, dtype=t.float)   # default value is [0.1, 0.1, 0.2, 0.2], you can change it in config.py

    @nograd
    def __call__(self, rois, gt_boxes, gt_labels):
        # 1. dataset for choosing *n_sample* samples, note **gt_boxes** can also be choosen, 将roi与gt box合并作为备选对象，统一记录为roi。（这里注意，128的可选范围不仅仅是roi）
        rois = t.cat([rois, gt_boxes], dim=0)
        
        # 2. calc iou between roi & gt_boxes, note here rois including gt, 计算roi与每个ground truth box之间的IOU，形成n*m的矩阵（n为roi数量，m为ground truth box数量）
        iou = calc_iou(rois, gt_boxes) # iou shape is n*m
        target_gt_max_iou_val, target_gt_max_iou_arg = iou.max(axis=1)
        target_gt_labels = gt_labels[target_gt_max_iou_arg]+1 #label index to add one, make the bg to be index 0

        # 3. choose max iou >= pos_iou_thresh, and check it if more than n_sample*pos_ratio
        # 选择所有max_iou>pos_iou_thresh=0.5的样本作为正例，若正例数量超过pos_num = n_sample*pos_ratio(=0.25)，则在正样本中随机选择pos_num个例子
        pos_keep_idx = t.where(target_gt_max_iou_val>=self.pos_iou_thresh)[0]
        pos_need_size = int(self.n_sample*self.pos_ratio)
        if pos_keep_idx.shape[0] > pos_need_size:
            sel_idx = t.randperm(pos_keep_idx.shape[0])[:pos_need_size]
            pos_keep_idx = pos_keep_idx[sel_idx]
        pos_size = min(pos_need_size, pos_keep_idx.shape[0])
        # 4. choose max iou >= neg_iou_thresh_lo & max iou < neg_iou_thresh_hi, and check it if more than n_sample-n_sample*pos_ratio
        # 选择所有max_iou<neg_iou_thresh_hi=0.5且max_iou>=neg_iou_thresh_lo=0.0的样本作为负例，若负例数量超过neg_num=n_sample-pos_need_size，则在负样本中随机选择neg_num个例子
        neg_keep_idx = t.where((target_gt_max_iou_val>=self.neg_iou_thresh_lo) & (target_gt_max_iou_val<self.neg_iou_thresh_hi))[0]
        neg_need_size = self.n_sample-pos_size
        if neg_keep_idx.shape[0] > neg_need_size:
            sel_idx = t.randperm(neg_keep_idx.shape[0])[:neg_need_size]
            neg_keep_idx = neg_keep_idx[sel_idx]
            target_gt_labels[neg_keep_idx] = 0   # it is background
        
        keep_idx = t.cat([pos_keep_idx, neg_keep_idx], dim=0)
        sample_rois = rois[keep_idx]
        gt_sample_boxes = gt_boxes[target_gt_max_iou_arg[keep_idx]]

        gt_sample_locs = box2delta(sample_rois, gt_sample_boxes)
        gt_sample_labels = target_gt_labels[keep_idx]

        gt_sample_locs = (gt_sample_locs - self.loc_normalize_mean) / self.loc_normalize_std

        gt_sample_roi_indices = t.zeros(self.n_sample, dtype=t.float32)

        return sample_rois, gt_sample_locs, gt_sample_labels, gt_sample_roi_indices


