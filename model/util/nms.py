import numpy as np
import torch as t
import nms_mh as mh
from model.util.iou import calc_iou, calc_iou_torch

def nms(boxes, scores, iou_thresh=0.5):
    '''
    non maximum supress: the boxes which have lower socre than the biggest one, and iou value bigger than iou_thresh will be supressed
    note: well, the function's result is the same with from torchvision.ops import nms, but the speed is so not so good, 6x slower than library function
    Args:
        boxes(numpy.array): must be in shape (R, 4), where R denotes the number of boxes, and has left-top point & right-down point with(x1,y1,x2,y2)
        scores(numpy.array): in shape(N,), here N==R, and every score show the confidence wether the corresponding box has objects
        iou_thresh(float):the thresh value, scores bigger than this value will be supressed

    Returns:
     * **keep_idx(numpy.array)**: the indexes which boxes should be reserved
    '''
    result = []
    arg_idx = scores.argsort()[::-1]
    
    while(arg_idx.shape[0]>=2):
        test_boxes = boxes[arg_idx]
        boxes_iou = calc_iou(test_boxes[:1], test_boxes[1:])[0]   # the shape is 1*boxes.shape[0]-1
        n = test_boxes.shape[0]
        keep_list = np.arange(1,n)[boxes_iou<=iou_thresh]
        result.append(arg_idx[0])
        arg_idx = arg_idx[keep_list]
    # add last one
    result.extend(arg_idx)
    return np.array(result, dtype=np.int32)


def nms_torch(boxes, scores, iou_thresh=0.5):
    '''

    non maximum supress: the boxes which have lower socre than the biggest one, and iou value bigger than iou_thresh will be supressed
    torch version
    Args:
        boxes(torch.tensor): must be in shape (R, 4), where R denotes the number of boxes, and has left-top point & right-down point with(x1,y1,x2,y2)
        scores(torch.tensor): in shape(N,), here N==R, and every score show the confidence wether the corresponding box has objects
        iou_thresh(float):the thresh value, scores bigger than this value will be supressed

    Returns:
     * **keep_idx(torch.tensor)**: the indexes which boxes should be reserved
    '''
    result = []
    arg_idx = scores.argsort(descending=True)
    
    while(arg_idx.shape[0]>=2):
        test_boxes = boxes[arg_idx]
        boxes_iou = calc_iou_torch(test_boxes[:1], test_boxes[1:])[0]   # the shape is 1*boxes.shape[0]-1
        n = test_boxes.shape[0]
        keep_list = t.arange(1,n)[boxes_iou<=iou_thresh]
        result.append(arg_idx[0].item())
        arg_idx = arg_idx[keep_list]
    # add last one
    result.extend(arg_idx)
    return t.tensor(result, dtype=t.long)

def nms2(boxes, scores, iou_thresh=0.5):
    '''
    non maximum supress v2: the boxes which have lower socre than the biggest one, and iou value bigger than iou_thresh will be supressed
    note: v2 will return less boxes than nms v1, this is because some boxes will be removed in previous round in v1 function which will not
    interfact the latter boxes
    
    Args:
        boxes(np.array): must be in shape (R, 4), where R denotes the number of boxes, and has left-top point & right-down point with(x1,y1,x2,y2)
        scores(np.array): in shape(N,), here N==R, and every score show the confidence wether the corresponding box has objects
        iou_thresh(float):the thresh value, scores bigger than this value will be supressed

    Returns:
     * **keep_idx**: the indexes which boxes should be reserved
    '''
    arg_idx = scores.argsort()[::-1]
    n = len(arg_idx)
    if n<2:
        return arg_idx
    
    v = np.arange(n)
    test_boxes = boxes[arg_idx]
    boxes_mask = np.zeros((n,n), dtype=np.int32)
    
    for i in range(1,n):
        row = mh.calc_iou(test_boxes[(i-1):i,:], test_boxes[i:,:])[0]
        boxes_mask[i-1,i:]=row<=iou_thresh
    
    k = boxes_mask.sum(axis=0)
    keep_list = arg_idx[k==v]

    return keep_list
