from __future__ import  absolute_import

import numpy as np
from model.util.bbox_opt import delta2box, box2delta
from model.util.iou import calc_iou


class AnchorTargetCreator:
    def __init__(self, n_sample, pos_ratio, neg_iou_thresh, pos_iou_thresh):
        self.n_sample = n_sample                  # default value is 256, you can change it in config.py
        self.pos_ratio = pos_ratio                # default value is 0.5, means that in n_sample samples, half is positive, half is negative
        self.neg_iou_thresh = neg_iou_thresh
        self.pos_iou_thresh = pos_iou_thresh

    def __call__(self, anchor_boxes, gt_boxes, img_size):
        '''
        Args:
            anchor_boxes(np.array): the predefined anchor boxes in feature map 
            gt_boxes(np.array): the gt boxes in network input image(after scale)
            image((h,w)): the height & width for input image
        Returns:
            roi delta loc: region of intresting boxes, have same shape of anchor boxes, only **n_sample** area validate, used to train rpn network loc regression
            roi label: label of roi, only indicates foreground & background, in shape [R,4], where R is the number of anchor boxes
        '''
        # 0. the original number of anchor boxes
        n_anchor = anchor_boxes.shape[0]
        
        # 1. clip the anchor boxes beyond 0~h&w
        h, w = img_size
        keep_idx = np.where((anchor_boxes[:,0]>=0) & (anchor_boxes[:,1]>=0) & (anchor_boxes[:,2]<=w) & (anchor_boxes[:,3]<=h))[0]
        inside_boxes = anchor_boxes[keep_idx]

        # 1.1 I copy the code from https://github.com/matterport/Mask_RCNN, but I don't know why
        # Handle COCO crowds
        # A crowd box in COCO is a bounding box around several instances. Exclude
        # them from training. A crowd box is given a negative class ID.
        # crowd_ix = np.where(gt_class_ids < 0)[0]
        # if crowd_ix.shape[0] > 0:
        #     # Filter out crowds from ground truth class IDs and boxes
        #     non_crowd_ix = np.where(gt_class_ids > 0)[0]
        #     crowd_boxes = gt_boxes[crowd_ix]
        #     gt_class_ids = gt_class_ids[non_crowd_ix]
        #     gt_boxes = gt_boxes[non_crowd_ix]
        #     # Compute overlaps with crowd boxes [anchor_boxes, crowds]
        #     crowd_overlaps = calc_iou(anchor_boxes, crowd_boxes)
        #     crowd_iou_max = np.amax(crowd_overlaps, axis=1)
        #     no_crowd_bool = (crowd_iou_max < 0.001)
        # else:
        #     # All anchors don't intersect a crowd
        #     no_crowd_bool = np.ones([anchor_boxes.shape[0]], dtype=bool)

        # 2. calc iou
        iou_mat = calc_iou(inside_boxes, gt_boxes)
        target_gt_max_iou_val = iou_mat.max(axis=1)
        target_gt_max_iou_arg = iou_mat.argmax(axis=1)
        # note: here we do not directly use argmax, if argmax, there is only len(gt_boxes) max arg, but sometimes maybe more
        # for example, more than one have the max val in dim=0 direction
        gt_target_max_iou_val = iou_mat.max(axis=0)
        gt_target_max_iou_arg = np.where(iou_mat == gt_target_max_iou_val)[0]
        # 3. setup the label
        labels = np.array([-1]*inside_boxes.shape[0])
        # 4. set label to be 0, if the max_iou <= neg_iou_thresh, 将max_iou小于neg_iou_thresh（=0.3）的，其标签置为0
        labels[target_gt_max_iou_val<self.neg_iou_thresh]=0
        # 5. set every gt box's max iou anchor target to be 1, 对于每个gt box，与其有最大IOU值的anchor box对应标签设置为1
        labels[gt_target_max_iou_arg]=1
        # 6. set label to be 1 if the max_iou >= pos_iou_thresh, 将max_iou大于pos_iou_thresh（=0.7）的，其标签置为1
        labels[target_gt_max_iou_val>=self.pos_iou_thresh]=1
        # 7. check wether the count of label=1 is greater than pos_ratio*n_sample, if yes, set the remainder to be -1 randomly
        # 检查标签列表中为1的数量，如果大于pos_ratio（=0.5）*n_sample（256）的，随机选择多余数量的部分设置为-1
        pos_count = np.sum(labels==1)
        if pos_count > self.pos_ratio*self.n_sample:
            reset_count = int(pos_count-self.pos_ratio*self.n_sample)
            reset_index = np.random.choice(pos_count, size=reset_count, replace=False)
            labels[np.where(labels==1)[0][reset_index]]=-1
        # 8. check wether the count of label=0 is greater than n_sample-pos_count, if yes set the remainder to be -1 randomly
        # 检查列表中为0的数量，，如果大于n_sample（256）-pos_ratio（=0.5）*n_sample（256），那么随机选择多余部分设置为-1
        pos_count = np.sum(labels==1)
        neg_count = np.sum(labels==0)
        if neg_count > self.n_sample-pos_count:
            reset_count = neg_count+pos_count-self.n_sample
            reset_index = np.random.choice(neg_count, size=reset_count, replace=False)
            labels[np.where(labels==0)[0][reset_index]]=-1
        
        # 9.result, we only care about fg object location
        anchor_target = inside_boxes[labels>0]
        argmax_boxes = gt_boxes[target_gt_max_iou_arg][labels>0]
        loc = box2delta(anchor_target, argmax_boxes)

        # 10. pack result to shape (n*),
        # fastercnn: n's typical value is w//16*h//16*9, w&h is input image size, 16 is scaler, 9 is anchor box number per feature map cell
        # mask rcnn: n's typical value is (feat_height/anchor_stride)*(feat_width/anchor_stride)*15*number of feat map
        # number of feat map is 5(p2, p3, p4, p5, p6), anchor stride is 1(means generat anchor box for every feature map cell, can be other values)
        # different feat map has different feat_height & feat_width             
        new_labels = np.array([-1]*n_anchor)
        new_labels[keep_idx] = labels
        new_loc = np.zeros((n_anchor,4))
        new_loc[keep_idx[labels>0]] = loc

        # new_loc : n*4
        # new_labels: n*1
        # only **n_sample** is validate, pos_ratio*n_sample are positive samples, remainder are negative samples
        return new_loc, new_labels