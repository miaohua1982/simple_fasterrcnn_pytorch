import torch as t
import math

def get_center_xy(boxes):
    ws = boxes[:,[2]]-boxes[:,[0]]
    hs = boxes[:,[3]]-boxes[:,[1]]

    x = boxes[:,[0]]+ws/2.0
    y = boxes[:,[1]]+hs/2.0

    center = t.cat([x, y], dim=1)

    return center

def calc_iou_helper(proposal_boxes, gt_boxes):
    lt_x = t.max(proposal_boxes[:,[0]], gt_boxes[:,0])
    lt_y = t.max(proposal_boxes[:,[1]], gt_boxes[:,1])

    rd_x = t.min(proposal_boxes[:,[2]], gt_boxes[:,2])
    rd_y = t.min(proposal_boxes[:,[3]], gt_boxes[:,3])

    ws = t.clamp(rd_x-lt_x, min=0.0)
    hs = t.clamp(rd_y-lt_y, min=0.0)

    proposal_area = (proposal_boxes[:,2]-proposal_boxes[:,0])*(proposal_boxes[:,3]-proposal_boxes[:,1])
    gt_area = (gt_boxes[:,2]-gt_boxes[:,0])*(gt_boxes[:,3]-gt_boxes[:,1])
    inter_area = ws*hs
    
    proposal_area = proposal_area.view(-1,1)
    gt_area = gt_area.view(1,-1)

    iou = inter_area/(proposal_area+gt_area-inter_area+1e-7)
    # the shape of iou is num_proposal*num_gt
    return iou, inter_area, proposal_area+gt_area-inter_area

def calc_iou_iou(proposal_boxes, gt_boxes):
    iou, _, _ = calc_iou_helper(proposal_boxes, gt_boxes)
    # the shape of iou is num_proposal*num_gt
    # iou ~ [0,1]
    return iou

def calc_iou_giou(proposal_boxes, gt_boxes):
    '''
    to calc the giou, giou = iou-(A-U)/A, where A is the minmum box area tha contains the pb & gb, U is area for pb and gb minus inter-area
    Args:
       proposal_boxes(tensor): the region for proposals, in shape `[R, 4]`, R denotes the number of proposal boxes
       gt_boxes(tensor): the region for ground truth, in shape `[T, 4]`, T denotes the number of ground truth boxes
    Returns:
       giou(tensor): the global iou values for proposal boxes & gt boxes, in shape [R,T]
    '''
    iou, inter_area, union_area = calc_iou_helper(proposal_boxes, gt_boxes)
    
    # calc minimum containing box
    lt_x = t.min(proposal_boxes[:,[0]], gt_boxes[:,0])
    lt_y = t.min(proposal_boxes[:,[1]], gt_boxes[:,1])

    rd_x = t.max(proposal_boxes[:,[2]], gt_boxes[:,2])
    rd_y = t.max(proposal_boxes[:,[3]], gt_boxes[:,3])

    ws = rd_x-lt_x
    hs = rd_y-lt_y
    contain_area = ws*hs
    
    # calc
    giou = iou - (contain_area-union_area)/contain_area
    # the shape of giou is num_proposal*num_gt
    # giou ~ [-1,1]
    return giou

def calc_iou_diou(proposal_boxes, gt_boxes):
    '''
    to calc the diou, diou = iou-d(pb, gb)/d(diagonal of containing box), where d(pb,gb) is the distance between proposal boxes and ground truth boxes,
    containing box is the mimimum boxes which containing the proposal box & ground truth
    Args:
       proposal_boxes(tensor): the region for proposals, in shape `[R, 4]`, R denotes the number of proposal boxes
       gt_boxes(tensor): the region for ground truth, in shape `[R, 4]`, T denotes the number of ground truth boxes
    Returns:
       diou(tensor): the distance iou values for proposal boxes & gt boxes, in shape [R,T]
    '''
    iou, inter_area, union_area = calc_iou_helper(proposal_boxes, gt_boxes)
    
    # calc minimum containing box diagonal distance
    lt_x = t.min(proposal_boxes[:,[0]], gt_boxes[:,0])
    lt_y = t.min(proposal_boxes[:,[1]], gt_boxes[:,1])

    rd_x = t.max(proposal_boxes[:,[2]], gt_boxes[:,2])
    rd_y = t.max(proposal_boxes[:,[3]], gt_boxes[:,3])

    dis_con = t.pow(rd_x-lt_x,2)+t.pow(rd_y-lt_y,2)

    # calc center of proposal boxes & ground truth boxes
    pb_center = get_center_xy(proposal_boxes)
    gb_center = get_center_xy(gt_boxes)

    wd = pb_center[:,[0]] - gb_center[:,0]
    hd = pb_center[:,[1]] - gb_center[:,1]
    
    dis_boxes = t.pow(wd,2)+t.pow(hd,2)

    diou = iou - dis_boxes/dis_con
    # the shape of diou is num_proposal*num_gt
    # diou ~ [-1,1]
    return diou

def calc_iou_ciou(proposal_boxes, gt_boxes):
    '''
    to calc the ciou, ciou = iou-d(pb, gb)/d(diagonal of containing box)-alpha*v, where d(pb,gb) is the distance between proposal boxes and ground truth boxes,
    containing box is the mimimum boxes which containing the proposal box & ground truth,
    arctan = pow((arctan(w_gt/h_gt) - arctan(w/h)), 2)
    v = 4/np.pi**2*arctan
    alpha = v/(1-iou+v)

    Args:
       proposal_boxes(tensor): the region for proposals, in shape `[R, 4]`, R denotes the number of proposal boxes
       gt_boxes(tensor): the region for ground truth, in shape `[R, 4]`, T denotes the number of ground truth boxes
    Returns:
       ciou(tensor): the complete iou values for proposal boxes & gt boxes, in shape [R,T]
    '''
    iou, inter_area, union_area = calc_iou_helper(proposal_boxes, gt_boxes)
    
    # calc minimum containing box diagonal distance
    lt_x = t.min(proposal_boxes[:,[0]], gt_boxes[:,0])
    lt_y = t.min(proposal_boxes[:,[1]], gt_boxes[:,1])

    rd_x = t.max(proposal_boxes[:,[2]], gt_boxes[:,2])
    rd_y = t.max(proposal_boxes[:,[3]], gt_boxes[:,3])

    dis_con = t.pow(rd_x-lt_x,2)+t.pow(rd_y-lt_y,2)

    # calc center of proposal boxes & ground truth boxes
    pb_center = get_center_xy(proposal_boxes)
    gb_center = get_center_xy(gt_boxes)

    wd = pb_center[:,[0]] - gb_center[:,0]
    hd = pb_center[:,[1]] - gb_center[:,1]
    
    dis_boxes = t.pow(wd,2)+t.pow(hd,2)

    # calc alpha*v, which considers the scale of weight and height
    w_gt = gt_boxes[:,2]-gt_boxes[:,0]
    h_gt = gt_boxes[:,3]-gt_boxes[:,1]
    w = proposal_boxes[:,[2]]-proposal_boxes[:,[0]]
    h = proposal_boxes[:,[3]]-proposal_boxes[:,[1]]
    arctan = t.pow((t.atan(w_gt/h_gt) - t.atan(w/h)), 2)
    v = 4/math.pi**2*arctan
    alpha = v/(1-iou+v)

    # get the ciou
    ciou = iou - dis_boxes/dis_con-alpha*v
    # the shape of diou is num_proposal*num_gt
    # ciou ~ [-1,1]
    return ciou
    
iou_routine_map = {'iou':calc_iou_iou, 'giou':calc_iou_giou, 'diou':calc_iou_diou, 'ciou':calc_iou_ciou}
def calc_iou(proposal_boxes, gt_boxes, iou_algo='iou'):
    iou = iou_routine_map[iou_algo](proposal_boxes, gt_boxes)
    # the shape of iou is num_proposal*num_gt
    return iou
