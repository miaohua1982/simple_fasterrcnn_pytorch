import numpy as np
import torch as t

def gen_anchor_boxes(ratios=[0.5, 1, 2], anchor_scales=[8, 16, 32], base_size=16):
    base_box = []
    area = base_size*base_size
    ctx, cty = base_size/2.0, base_size/2.0

    for r in ratios:
        w = np.sqrt(area/r)
        h = w*r
        for s in anchor_scales:
            sw = w*s
            sh = h*s
            new_box = [int(ctx-sw/2.0), int(cty-sh/2.0), int(ctx+sw/2.0), int(cty+sh/2.0)]
            base_box.append(new_box)

    base_box = np.array(base_box, dtype=np.float32)
    return base_box

def shift_anchor_boxes(base_anchor_boxes, h, w, feat_stride): 
    '''
    Args:
       base_anchor_boxes: the base anchor boxes, shape is 9*4
       h: the feature map height, typical value is 50
       w: the feature map width, typical value is 37
       feat_stride: the feature scale from input image to current feature map, typical value is 16
    Outputs:
       pre_defined_anchor_boxes: the shifted base anchor boxes, shape is (h*w*base_anchor_boxes.shape[0],4) 
    '''
    x = np.arange(0, w*feat_stride, feat_stride, np.float32)
    y = np.arange(0, h*feat_stride, feat_stride, np.float32)

    grid_x, grid_y = np.meshgrid(x,y)
    
    shift = np.stack([grid_x.flatten(), grid_y.flatten(), grid_x.flatten(), grid_y.flatten()], axis=1)
    
    A = base_anchor_boxes.shape[0]
    K = shift.shape[0]

    pre_defined_anchor_boxes = base_anchor_boxes.reshape((1,A,4))+shift.reshape((K,1,4))
    pre_defined_anchor_boxes = pre_defined_anchor_boxes.reshape((K*A, 4))
    return pre_defined_anchor_boxes.astype(np.float32)

def calc_iou(proposal_boxes, gt_boxes):
    lt_x = np.maximum(proposal_boxes[:,[0]], gt_boxes[:,0])
    lt_y = np.maximum(proposal_boxes[:,[1]], gt_boxes[:,1])

    rd_x = np.minimum(proposal_boxes[:,[2]], gt_boxes[:,2])
    rd_y = np.minimum(proposal_boxes[:,[3]], gt_boxes[:,3])

    ws = np.clip(rd_x-lt_x, a_min=0.0, a_max=None)
    hs = np.clip(rd_y-lt_y, a_min=0.0, a_max=None)

    proposal_area = (proposal_boxes[:,2]-proposal_boxes[:,0])*(proposal_boxes[:,3]-proposal_boxes[:,1])
    gt_area = (gt_boxes[:,2]-gt_boxes[:,0])*(gt_boxes[:,3]-gt_boxes[:,1])
    inter_area = ws*hs
    
    proposal_area = proposal_area.reshape(-1,1)
    gt_area = gt_area.reshape(1,-1)

    iou = inter_area/(proposal_area+gt_area-inter_area)
    # the shape of iou is num_proposal*num_gt
    return iou

def calc_iou_torch(proposal_boxes, gt_boxes):
    lt_x = t.max(proposal_boxes[:,[0]], gt_boxes[:,0])
    lt_y = t.max(proposal_boxes[:,[1]], gt_boxes[:,1])

    rd_x = t.min(proposal_boxes[:,[2]], gt_boxes[:,2])
    rd_y = t.min(proposal_boxes[:,[3]], gt_boxes[:,3])

    ws = t.clamp(rd_x-lt_x, min=0.0)
    hs = t.clamp(rd_y-lt_y, min=0.0)

    proposal_area = (proposal_boxes[:,2]-proposal_boxes[:,0])*(proposal_boxes[:,3]-proposal_boxes[:,1])
    gt_area = (gt_boxes[:,2]-gt_boxes[:,0])*(gt_boxes[:,3]-gt_boxes[:,1])
    inter_area = ws*hs
    
    proposal_area = proposal_area.reshape(-1,1)
    gt_area = gt_area.reshape(1,-1)

    iou = inter_area/(proposal_area+gt_area-inter_area)
    # the shape of iou is num_proposal*num_gt
    return iou

def delta2box(base_boxes, deltas):
    xywh_boxes = xxyy2xywh(base_boxes)

    w = np.exp(deltas[:,[2]])*xywh_boxes[:,[2]]
    h = np.exp(deltas[:,[3]])*xywh_boxes[:,[3]]
    x = xywh_boxes[:,[2]]*deltas[:,[0]]+xywh_boxes[:,[0]]
    y = xywh_boxes[:,[3]]*deltas[:,[1]]+xywh_boxes[:,[1]]

    boxes = np.concatenate([x, y, w, h], axis=1)

    xxyy_boxes = xywh2xxyy(boxes)

    return xxyy_boxes

def box2delta(predict_boxes, gt_boxes):
    predict_boxes_xywh = xxyy2xywh(predict_boxes)
    gt_boxes_xywh = xxyy2xywh(gt_boxes)

    dx = (gt_boxes_xywh[:,[0]]-predict_boxes_xywh[:,[0]])/predict_boxes_xywh[:,[2]]
    dy = (gt_boxes_xywh[:,[1]]-predict_boxes_xywh[:,[1]])/predict_boxes_xywh[:,[3]]

    scale_w = np.log(gt_boxes_xywh[:,[2]]/predict_boxes_xywh[:,[2]])
    scale_h = np.log(gt_boxes_xywh[:,[3]]/predict_boxes_xywh[:,[3]])

    deltas = np.concatenate([dx, dy, scale_w, scale_h], axis=1).astype(np.float32)
    return deltas

def xxyy2xywh(box):
    w = box[:,[2]]-box[:,[0]]
    h = box[:,[3]]-box[:,[1]]

    crt_x = box[:,[0]]+w/2.0
    crt_y = box[:,[1]]+h/2.0

    new_box = np.concatenate([crt_x, crt_y, w, h], axis=1).astype(np.float32)
    return new_box

def xywh2xxyy(box):
    lp_x = box[:,[0]]-box[:,[2]]/2.0
    lp_y = box[:,[1]]-box[:,[3]]/2.0

    rd_x = box[:,[0]]+box[:,[2]]/2.0
    rd_y = box[:,[1]]+box[:,[3]]/2.0

    new_box = np.concatenate([lp_x, lp_y, rd_x, rd_y], axis=1).astype(np.float32)
    return new_box



if __name__ == '__main__':
    print(gen_anchor_boxes())