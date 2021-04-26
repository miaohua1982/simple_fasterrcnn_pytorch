import numpy as np
import nms_mh as mh
from model.util.iou import calc_iou

def nms(boxes, scores, iou_thresh=0.5, iou_algo='iou'):
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
        # use c++ implementation to accelerate
        boxes_iou = mh.calc_iou(test_boxes[:1], test_boxes[1:], iou_algo)[0]   # the shape is 1*boxes.shape[0]-1
        n = test_boxes.shape[0]
        keep_list = np.arange(1,n)[boxes_iou<=iou_thresh]
        result.append(arg_idx[0])
        arg_idx = arg_idx[keep_list]
    # add last one
    result.extend(arg_idx)
    return np.array(result, dtype=np.int32)