import torch as t
from model.util.iou_t import calc_iou

def nms(boxes, scores, iou_thresh=0.5, iou_algo='iou'):
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
        boxes_iou = calc_iou(test_boxes[:1], test_boxes[1:], iou_algo)[0]   # the shape is 1*boxes.shape[0]-1
        n = test_boxes.shape[0]
        keep_list = t.arange(1,n)[boxes_iou<=iou_thresh]
        result.append(arg_idx[0].item())
        arg_idx = arg_idx[keep_list]
    # add last one
    result.extend(arg_idx)
    return t.tensor(result, dtype=t.long)