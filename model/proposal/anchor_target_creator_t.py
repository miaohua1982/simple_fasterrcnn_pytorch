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

class AnchorTargetCreator:
    def __init__(self, n_sample, pos_ratio, neg_iou_thresh, pos_iou_thresh):
        self.n_sample = n_sample                  # default value is 256, you can change it in config.py
        self.pos_ratio = pos_ratio
        self.neg_iou_thresh = neg_iou_thresh
        self.pos_iou_thresh = pos_iou_thresh

    @nograd
    def __call__(self, anchor_boxes, gt_boxes, img_size):
        '''
        anchor_boxes(np.array): the predefined anchor boxes in feature map 
        gt_boxes(np.array): the gt boxes in network input image(after scale)
        image((h,w)): the height & width for input image
        '''
        # 0. the orignal number of anchor boxes
        n_anchor = anchor_boxes.shape[0]
        # 1. remove the position beyond 0~h&w
        h, w = img_size
        keep_idx = t.where((anchor_boxes[:,0]>=0) & (anchor_boxes[:,1]>=0) & (anchor_boxes[:,2]<=w) & (anchor_boxes[:,3]<=h))[0]
        inside_boxes = anchor_boxes[keep_idx]
        # 2. calc iou
        iou_mat = calc_iou(inside_boxes, gt_boxes)
        target_gt_max_iou_val, target_gt_max_iou_arg = iou_mat.max(axis=1)
        # note: here we do not directly use argmax, if argmax, there is only len(gt_boxes) max arg, but sometimes maybe more
        # for example, more than one have the max val in dim=0 direction
        gt_target_max_iou_val = iou_mat.max(axis=0)[0]
        gt_target_max_iou_arg = t.where(iou_mat == gt_target_max_iou_val)[0]
        # 3. setup the label
        labels = t.tensor([-1]*inside_boxes.shape[0])
        # 4. set label to be 0, if the max_iou <= neg_iou_thresh, 将max_iou小于neg_iou_thresh（=0.3）的，其标签置为0
        labels[target_gt_max_iou_val<self.neg_iou_thresh]=0
        # 5. set every gt box's max iou anchor target to be 1, 对于每个gt box，与其有最大IOU值的anchor box对应标签设置为1
        labels[gt_target_max_iou_arg]=1
        # 6. set label to be 1 if the max_iou >= pos_iou_thresh, 将max_iou大于pos_iou_thresh（=0.7）的，其标签置为1
        labels[target_gt_max_iou_val>=self.pos_iou_thresh]=1
        # 7. check wether the count of label=1 is greater than pos_ratio*n_sample, if yes, set the remainder to be -1 randomly
        # 检查标签列表中为1的数量，如果大于pos_ratio（=0.5）*n_sample（256）的，随机选择多余数量的部分设置为-1
        pos_count = t.sum(labels==1)
        if pos_count > self.pos_ratio*self.n_sample:
            reset_count = int(pos_count-self.pos_ratio*self.n_sample)
            reset_index = t.randperm(pos_count)[:reset_count]
            labels[t.where(labels==1)[0][reset_index]]=-1
        # 8. check wether the count of label=0 is greater than n_sample-pos_count, if yes set the remainder to be -1 randomly
        # 检查列表中为0的数量，，如果大于n_sample（256）-pos_ratio（=0.5）*n_sample（256），那么随机选择多余部分设置为-1
        pos_count = t.sum(labels==1)
        neg_count = t.sum(labels==0)
        if neg_count > self.n_sample-pos_count:
            reset_count = neg_count+pos_count-self.n_sample
            reset_index = t.randperm(neg_count)[:reset_count]
            labels[t.where(labels==0)[0][reset_index]]=-1
        
        # 9.result, we only care about fg object location
        anchor_target = inside_boxes[labels>0]
        argmax_boxes = gt_boxes[target_gt_max_iou_arg][labels>0]
        loc = box2delta(anchor_target, argmax_boxes)

        # 10. pack result to shape (n*), n's typical value is w//16*h//16*9, w&h is input image size, 16 is scaler, 9 is anchor box number
        new_labels = t.tensor([-1]*n_anchor)
        new_labels[keep_idx] = labels
        new_loc = t.zeros((n_anchor,4))
        new_loc[keep_idx[labels>0]] = loc

        return new_loc, new_labels