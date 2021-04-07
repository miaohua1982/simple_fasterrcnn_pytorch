import numpy as np
from model.util.bbox_opt import calc_iou

def nms(boxes, scores, iou_thresh=0.5):
    '''
    non maximum supress: the boxes which have lower socre than the biggest one, and iou value bigger than iou_thresh will be supressed
    
    Args:
        boxes(np.array): must be in shape (R, 4), where R denotes the number of boxes, and has left-top point & right-down point with(x1,y1,x2,y2)
        scores(np.array): in shape(N,), here N==R, and every score show the confidence wether the corresponding box has objects
        iou_thresh(float):the thresh value, scores bigger than this value will be supressed

    Returns:
     * **keep_idx**: the indexes which boxes should be reserved
    '''
    result = []
    arg_idx = scores.argsort()[::-1]

    # only one box, do nothing
    if len(arg_idx) < 2:
        return arg_idx
    
    while(arg_idx.shape[0]>=2):
        scores = scores[arg_idx]
        boxes = boxes[arg_idx]
        boxes_iou = calc_iou(boxes[:1], boxes[1:])
        n = boxes_iou.shape[0] # the n = boxes.shape[0]-1
        keep_list = []
        for i in range(n):
            if boxes_iou[i]<=iou_thresh:
                keep_list.append(i+1)  #0 is for the biggest score one
        
        result.append(arg_idx[0])
        arg_idx = arg_idx[keep_list]

    return np.array(result, dtype=np.int32)
        

    