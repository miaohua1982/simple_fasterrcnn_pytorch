import torch as t
import numpy as np
import nms_mh as m
from torchvision.ops import nms

assert m.add(1, 2) == 3


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
    
def py_nms(boxes, scores, iou_thresh=0.5):
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
    
def py_nms_cbox(boxes, scores, iou_thresh=0.5):
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
        boxes_iou = m.calc_iou(test_boxes[:1], test_boxes[1:])[0]   # the shape is 1*boxes.shape[0]-1
        n = test_boxes.shape[0]
        keep_list = np.arange(1,n)[boxes_iou<=iou_thresh]
        result.append(arg_idx[0])
        arg_idx = arg_idx[keep_list]
    # add last one
    result.extend(arg_idx)
    return np.array(result, dtype=np.int32)

# test nms  
rois = np.random.rand(12000,4)
scores = np.random.rand(12000)

keep1 = py_nms(rois, scores, 0.7)
keep2 = py_nms_cbox(rois, scores, 0.7)
keep3 = m.nms(rois, scores, 0.7)
keep4 = nms(t.from_numpy(rois), t.from_numpy(scores), 0.7)

assert np.all(keep1==keep2)
assert np.all(keep2==keep3)
assert np.all(keep3==keep4.numpy())
# test speed in ipython
#%timeit keep1 = py_nms(rois, scores, 0.7)
#%timeit keep2 = py_nms_cbox(rois, scores, 0.7)
#%timeit keep3 = m.nms(rois, scores, 0.7)
#%timeit keep4 = nms(t.from_numpy(rois), t.from_numpy(scores), 0.7)


pb = np.random.rand(12000,4)
gb = np.random.rand(12000,4)
iou1 = calc_iou(pb, gb)
iou2 = m.calc_iou(pb, gb)
assert iou1.shape == iou2.shape
# can not do assert np.all(iou1==iou2), because nms_mh lib use float, but python code is double
