import numpy as np

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
       base_anchor_boxes(np.array): the base anchor boxes, shape is 9*4
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

def gen_pyramid_anchors(scales, ratios, image_size, feat_strides, anchor_stride):
    '''
    generate anchor boxes for every feature map
    Args:
        scales:the area for boxes, a typical value is [32,64,128,256,512]
        ratios:the different ratio for height/width, a typical value is [0.5,1,2]
        image_size: a tuple, (height, width), stands for input image size
        feat_strides:stride for different scale backbone layers output, a typical values is [4,8,16,32,64]
        anchor_stride: wether to generate anchor boxes for every *stride* cell, default value is 1, for every cell
    Returns:
        anchor_boxes:predefined anchor boxes, in shape [R, 4], 
        R=len(scales)*len(ratios)*len(feat_stride)*int(feat_height/anchor_stride)*int(feat_width/anchor_stride)
    '''
    anchor_boxes = []
    for i in range(len(scales)):
        one_scale = scales[i]
        feat_stride = feat_strides[i]
        feat_size = [int(np.ceil(image_size[0]/feat_stride)), int(np.ceil(image_size[1]/feat_stride))]
        x = np.arange(0, feat_size[1], anchor_stride, dtype=np.float32)*feat_stride
        y = np.arange(0, feat_size[0], anchor_stride, dtype=np.float32)*feat_stride

        shift_x, shift_y = np.meshgrid(x, y)
        shifts = np.stack([shift_x.flatten(), shift_y.flatten(), shift_x.flatten(), shift_y.flatten()], axis=1)

        for one_ratio in ratios:
            h = one_scale/np.sqrt(one_ratio)
            w = one_scale*np.sqrt(one_ratio)
            ctx, cty = w/2, h/2

            base_box = np.array([ctx-w/2, cty-h/2, ctx+w/2, cty+h/2])
            boxes = shifts+base_box # R*4

            anchor_boxes.append(boxes)

    anchor_boxes = np.concatenate(anchor_boxes, axis=0)            
    return anchor_boxes
            


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