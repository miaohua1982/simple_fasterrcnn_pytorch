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

    base_box = t.tensor(base_box, dtype=t.float32)
    return base_box

def shift_anchor_boxes(base_anchor_boxes, h, w, feat_stride): 
    '''
    Args:
       base_anchor_boxes(tensor): the base anchor boxes, shape is 9*4
       h: the feature map height, typical value is 50
       w: the feature map width, typical value is 37
       feat_stride: the feature scale from input image to current feature map, typical value is 16
    Outputs:
       pre_defined_anchor_boxes: the shifted base anchor boxes, shape is (h*w*base_anchor_boxes.shape[0],4) 
    '''
    x = t.arange(0, w*feat_stride, feat_stride, dtype=t.float32)
    y = t.arange(0, h*feat_stride, feat_stride, dtype=t.float32)

    grid_y, grid_x = t.meshgrid(y,x)
    
    shift = t.cat([grid_x.flatten().view(-1,1), grid_y.flatten().view(-1,1), grid_x.flatten().view(-1,1), grid_y.flatten().view(-1,1)], dim=1)
    
    A = base_anchor_boxes.shape[0]
    K = shift.shape[0]

    pre_defined_anchor_boxes = base_anchor_boxes.view((1,A,4))+shift.view((K,1,4))
    pre_defined_anchor_boxes = pre_defined_anchor_boxes.view((K*A, 4))
    return pre_defined_anchor_boxes.float()


def delta2box(base_boxes, deltas):
    xywh_boxes = xxyy2xywh(base_boxes)

    w = t.exp(deltas[:,[2]])*xywh_boxes[:,[2]]
    h = t.exp(deltas[:,[3]])*xywh_boxes[:,[3]]
    x = xywh_boxes[:,[2]]*deltas[:,[0]]+xywh_boxes[:,[0]]
    y = xywh_boxes[:,[3]]*deltas[:,[1]]+xywh_boxes[:,[1]]

    boxes = t.cat([x, y, w, h], dim=1)

    xxyy_boxes = xywh2xxyy(boxes)

    return xxyy_boxes

def box2delta(predict_boxes, gt_boxes):
    predict_boxes_xywh = xxyy2xywh(predict_boxes)
    gt_boxes_xywh = xxyy2xywh(gt_boxes)

    dx = (gt_boxes_xywh[:,[0]]-predict_boxes_xywh[:,[0]])/predict_boxes_xywh[:,[2]]
    dy = (gt_boxes_xywh[:,[1]]-predict_boxes_xywh[:,[1]])/predict_boxes_xywh[:,[3]]

    scale_w = t.log(gt_boxes_xywh[:,[2]]/predict_boxes_xywh[:,[2]])
    scale_h = t.log(gt_boxes_xywh[:,[3]]/predict_boxes_xywh[:,[3]])

    deltas = t.cat([dx, dy, scale_w, scale_h], dim=1).astype(t.float32)
    return deltas

def xxyy2xywh(box):
    w = box[:,[2]]-box[:,[0]]
    h = box[:,[3]]-box[:,[1]]

    crt_x = box[:,[0]]+w/2.0
    crt_y = box[:,[1]]+h/2.0

    new_box = t.cat([crt_x, crt_y, w, h], dim=1).astype(np.float32)
    return new_box

def xywh2xxyy(box):
    lp_x = box[:,[0]]-box[:,[2]]/2.0
    lp_y = box[:,[1]]-box[:,[3]]/2.0

    rd_x = box[:,[0]]+box[:,[2]]/2.0
    rd_y = box[:,[1]]+box[:,[3]]/2.0

    new_box = t.cat([lp_x, lp_y, rd_x, rd_y], dim=1).astype(np.float32)
    return new_box

if __name__ == '__main__':
    print(gen_anchor_boxes_t())