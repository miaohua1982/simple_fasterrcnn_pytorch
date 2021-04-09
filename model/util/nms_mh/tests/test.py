import nms_mh as m
import numpy as np

assert m.__version__ == '0.0.1'
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

pb = np.random.rand(12000,4)
gb = np.random.rand(12000,4)
iou1 = calc_iou(pb, gb)
iou2 = m.calc_iou(pb, gb)
assert iou1.shape == iou2.shape
