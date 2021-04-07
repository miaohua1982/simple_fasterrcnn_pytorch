import numpy as np
import torch as t
from model.util.bbox_opt import calc_iou

def nms(boxes, scores, iou_thresh=0.5):
    '''
    non maximum supress: the boxes which have lower socre than the biggest one, and iou value bigger than iou_thresh will be supressed
    
    Args:
        boxes(numpy.array): must be in shape (R, 4), where R denotes the number of boxes, and has left-top point & right-down point with(x1,y1,x2,y2)
        scores(numpy.array): in shape(N,), here N==R, and every score show the confidence wether the corresponding box has objects
        iou_thresh(float):the thresh value, scores bigger than this value will be supressed

    Returns:
     * **keep_idx(numpy.array)**: the indexes which boxes should be reserved
    '''
    result = []
    arg_idx = scores.argsort()[::-1]

    # only one box, do nothing
    if len(arg_idx) < 2:
        return arg_idx
    
    while(arg_idx.shape[0]>=2):
        test_boxes = boxes[arg_idx]
        boxes_iou = calc_iou(test_boxes[:1], test_boxes[1:])[0]   # the shape is 1*boxes.shape[0]-1
        n = boxes_iou.shape[0] # the n = boxes.shape[0]-1
        keep_list = np.arange(1,n+1)[boxes_iou<=iou_thresh]
        result.append(arg_idx[0])
        arg_idx = arg_idx[keep_list]
    # add last one
    result.append(arg_idx[0])
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

    # only one box, do nothing
    if len(arg_idx) < 2:
        return arg_idx
    
    while(arg_idx.shape[0]>=2):
        test_boxes = boxes[arg_idx]
        boxes_iou = calc_iou(test_boxes[:1].detach().numpy(), test_boxes[1:].detach().numpy())[0]   # the shape is 1*boxes.shape[0]-1
        n = boxes_iou.shape[0] # the n = boxes.shape[0]-1
        keep_list = t.arange(1,n+1)[boxes_iou<=iou_thresh]
        result.append(arg_idx[0].item())
        arg_idx = arg_idx[keep_list]
    # add last one
    result.append(arg_idx[0])
    return t.tensor(result, dtype=t.long)
