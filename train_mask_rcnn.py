from __future__ import  absolute_import

import matplotlib
matplotlib.use('Agg')

import os
import ipdb
import time
import torch as t
import numpy as np
from torch.utils import data as data_
from data.coco_dataset import Coco_Dataset
from data.util import inverse_normalize
from config.config import mask_running_args
from model.util.coco_eval import CocoEvaluator
from model.backbone.resnet import ResnetBackbone
from model.maskrcnn.maskrcnn import MaskRCNN

from model.util.vis_tool import Visualizer, visdom_bbox, visdom_mask_gt, visdom_mask_pred
from tqdm import tqdm

def set_seed_everywhere(seed, cuda):
    np.random.seed(seed)
    t.manual_seed(seed)
    if cuda:
        t.cuda.manual_seed_all(seed)

def get_optimizer(opt, model):
    """
    return optimizer, It could be overwritten if you want to specify 
    special optimizer
    """
    lr = opt.learning_rate
    params = []
    for key, value in dict(model.named_parameters()).items():
        if value.requires_grad:
            if 'bias' in key:
                params += [{'params': [value], 'lr': lr * 2, 'weight_decay': 0}]
            else:
                params += [{'params': [value], 'lr': lr, 'weight_decay': opt.weight_decay}]
    if opt.use_adam:
        optimizer = t.optim.Adam(params)
    else:
        optimizer = t.optim.SGD(params, momentum=0.9)
    return optimizer

def scale_lr(optimizer, decay=0.1):
    for param_group in optimizer.param_groups:
        param_group['lr'] *= decay
    return optimizer

def eval(model, opt, epoch_idx, test_num=5000):
    # load eval dataset
    dataset = Coco_Dataset(opt.dataset_base_path,min_size=opt.min_img_size, max_size=opt.max_img_size, split='val2017')
    test_dataset = data_.DataLoader(dataset, batch_size=1, shuffle=True)
    # setup coco eval class
    iou_types = ["bbox", "segm"]
    coco_evaluator = CocoEvaluator(dataset.coco, dataset.coco.classes_id_map, iou_types)

    # set to eval mode
    model.eval()

    # eval
    for idx, one_obj_ds in tqdm(enumerate(test_dataset)):
        img, prop = one_obj_ds
        boxes = prop['gt_boxes']
        labels = prop['gt_labels']
        masks = prop['gt_masks']
        scale = prop['scale']
        img_id = prop['image_id']

        if t.cuda.is_available():
            img, boxes, labels, masks = img.cuda(), boxes.cuda(), labels.cuda(), masks.cuda()

        pboxes, plabels, pscores, pmasks = model.predict(img, boxes, labels, masks, scale.item())
        
        coco_evaluator.prepare_for_one_coco(img_id.item(), pboxes, pscores, plabels, pmasks)

        if idx == test_num:
            break
    
    # save result
    coco_evaluator.finish_eval(mask_running_args.eval_result_path.format(epoch_idx))
    
    # eval
    coco_evaluator.accumulate()
    result = coco_evaluator.summarize()
    
    # set back to train mode
    model.train()

    # return result
    return result

def train(opt):
    # load dataset
    dataset = Coco_Dataset(opt.dataset_base_path, min_size=opt.min_img_size, max_size=opt.max_img_size, split='train2017')
    train_dataset = data_.DataLoader(dataset, batch_size=1, shuffle=False)

    # model
    # backbone
    backbone = ResnetBackbone(opt.backbone)
    maskrcnn = MaskRCNN(opt.num_classes, backbone, backbone_channels=256)
    if opt.load_model_path is not None:
        maskrcnn.load_state_dict(t.load(opt.load_model_path))
    maskrcnn = maskrcnn.cuda() if t.cuda.is_available() else maskrcnn

    # optimizer
    optimizer = get_optimizer(opt, maskrcnn)

    # visdom
    vis = Visualizer(env=opt.vis_env)
  
    # eval map
    cur_eval_map = 0.0
    # trace learning rate
    cur_lr = opt.learning_rate

    # set train mode
    maskrcnn.train()
    # train
    t.autograd.set_detect_anomaly(True)
    for epoch in range(opt.num_epochs):
        # rest loss meter
        avg_rpn_reg_loss = 0.0
        avg_roi_reg_loss = 0.0
        avg_rpn_cls_loss = 0.0
        avg_roi_cls_loss = 0.0
        avg_roi_mask_loss = 0.0
        avg_total_loss = 0.0
        # one epoch
        for idx, one_obj_ds in tqdm(enumerate(train_dataset)):
            img, prop = one_obj_ds
            gt_boxes = prop['gt_boxes']
            gt_labels = prop['gt_labels']
            gt_masks = prop['gt_masks']
            scale = prop['scale']
            img_id = prop['image_id']

            if t.cuda.is_available():
                img, gt_boxes, gt_labels, gt_masks = img.cuda(), gt_boxes.cuda(), gt_labels.cuda(), gt_masks.cuda()

            optimizer.zero_grad()
            rpn_cls_loss, rpn_reg_loss, roi_cls_loss, roi_reg_loss, roi_mask_loss, _, _, _, _ = maskrcnn(img, gt_boxes, gt_labels, gt_masks, scale.item())
            total_loss = rpn_cls_loss+rpn_reg_loss+roi_cls_loss+roi_reg_loss+roi_mask_loss
            total_loss.backward()
            optimizer.step()
            
            avg_rpn_reg_loss += (rpn_reg_loss.item()-avg_rpn_reg_loss)/(idx+1) 
            avg_roi_reg_loss += (roi_reg_loss.item()-avg_roi_reg_loss)/(idx+1) 
            avg_rpn_cls_loss += (rpn_cls_loss.item()-avg_rpn_cls_loss)/(idx+1)
            avg_roi_cls_loss += (roi_cls_loss.item()-avg_roi_cls_loss)/(idx+1)
            avg_roi_mask_loss += (roi_mask_loss.item()-avg_roi_mask_loss)/(idx+1)
            avg_total_loss = avg_rpn_reg_loss+avg_roi_reg_loss+avg_rpn_cls_loss+avg_roi_cls_loss+avg_roi_mask_loss

            if (idx+1)%opt.plot_spot == 0:
                if os.path.exists(opt.debug_file):
                    ipdb.set_trace()

                # plot loss
                vis.plot('rpn_loc_loss', avg_rpn_reg_loss)
                vis.plot('rpn_cls_loss', avg_rpn_cls_loss)
                vis.plot('roi_loc_loss', avg_roi_reg_loss)
                vis.plot('roi_cls_loss', avg_roi_cls_loss)
                vis.plot('roi_mask_loss', avg_roi_mask_loss)
                vis.plot('total_loss', avg_total_loss)

                # plot ground truth boxes
                ori_img = inverse_normalize(img[0].cpu().numpy())
                gt_img = visdom_bbox(ori_img, gt_boxes[0].cpu().numpy(), gt_labels[0].cpu().numpy())
                vis.img('gt_img', gt_img)

                # plot ground truth mask
                gt_mask_img = visdom_mask_gt(ori_img, img_id.item(), dataset.coco)
                vis.img('gt_mask_img', gt_mask_img)

                # plot predict boxes
                maskrcnn.eval()
                pred_boxes, pred_labels, pred_scores, pred_masks = maskrcnn.predict(img, gt_boxes, gt_labels, gt_masks, scale.item(), present='visualize')
                maskrcnn.train()
                # show predict img
                pred_img = visdom_bbox(ori_img, pred_boxes.cpu().numpy(), pred_labels.cpu().numpy(), pred_scores.cpu().numpy())
                vis.img('pred_img', pred_img)
                # show predict mask img
                pred_mask_img = visdom_mask_pred(ori_img, img_id.item(), dataset.coco, pred_masks.cpu().numpy(), pred_labels.cpu().numpy())
                vis.img('pred_mask_img', pred_mask_img)

        # eval
        result = eval(maskrcnn, epoch, opt)
        ap = result.get_AP()
        # py visdom plot
        vis.plot('eval_mask_ap', ap['mask AP'])
        vis.plot('eval_box_ap', ap['bbox AP'])

        # log
        log_info = 'In round %d, lr:%.6f, avg rpn regloss is %.6f, avg roi regloss is %.6f, avg rpn scoreloss is %.6f, avg roi scoreloss is %.6f, avg maskloss is %.6f, eval result is %s' % \
            (epoch, cur_lr, avg_rpn_reg_loss, avg_roi_reg_loss, avg_rpn_score_loss, avg_roi_score_loss, avg_roi_mask_loss, str(result))
        print(log_info)
        vis.log(log_info)

        # save model if needed
        if ap['mask AP'] > cur_eval_map:
            cur_eval_map = ap['mask AP']
            save_path = mask_running_args.save_model_path % (time.strftime("%m%d_%H%M"), cur_eval_map)
            t.save(maskrcnn.state_dict(), save_path)
            opt.load_model_path = save_path
        elif ap['mask AP'] < cur_eval_map:     # make sure current one is the best one
            maskrcnn.load_state_dict(t.load(opt.load_model_path))

        # decay the model's learning rate
        if epoch == 9:  # it is a trick
            optimizer = scale_lr(optimizer, mask_running_args.lr_decay)
            cur_lr *= mask_running_args.lr_decay
        
if __name__ == '__main__':
    set_seed_everywhere(mask_running_args.seed, t.cuda.is_available())
    train(mask_running_args)
