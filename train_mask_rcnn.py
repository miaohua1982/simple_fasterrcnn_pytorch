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
from model.util.eval_ap import eval_detection_voc
from model.backbone.resnet import ResnetBackbone
from model.maskrcnn.maskrcnn import MaskRCNN

from model.util.vis_tool import Visualizer, visdom_bbox, visdom_mask
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

def eval(model, opt, test_num=5000):
    # load eval dataset
    dataset = Coco_Dataset(opt.dataset_base_path,min_size=opt.min_img_size, max_size=opt.max_img_size, split='val2017')
    test_dataset = data_.DataLoader(dataset, batch_size=1, shuffle=True)

    # set to eval mode
    model.eval()

    # test 
    pred_boxes = []
    pred_labels = []
    pred_scores = []
    gt_boxes = []
    gt_labels = []
    gt_difficults = []
    for idx, one_obj_ds in tqdm(enumerate(test_dataset)):
        img, boxes, labels, difficults, scale = one_obj_ds
        
        if t.cuda.is_available():
            img, boxes, labels = img.cuda(), boxes.cuda(), labels.cuda()

        pboxes, plabels, pscores = model.predict(img, boxes, labels, scale.item())
        
        pred_boxes.append(pboxes.cpu().numpy())
        pred_labels.append(plabels.cpu().numpy())
        pred_scores.append(pscores.cpu().numpy())
        gt_boxes.append(boxes[0].cpu().numpy())
        gt_labels.append(labels[0].cpu().numpy())
        gt_difficults.append(difficults[0].cpu().numpy())

        if idx == test_num:
            break
    
    # result including ap & map
    result = eval_detection_voc(pred_boxes, pred_labels, pred_scores, gt_boxes, gt_labels, gt_difficults, iou_thresh=0.5, use_07_metric=True)
    
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
        avg_rpn_score_loss = 0.0
        avg_roi_score_loss = 0.0
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
                img, gt_boxes, gt_labels = img.cuda(), gt_boxes.cuda(), gt_labels.cuda()

            optimizer.zero_grad()
            rpn_score_loss, rpn_reg_loss, roi_score_loss, roi_reg_loss, roi_mask_loss = maskrcnn(img, gt_boxes, gt_labels, scale.item())
            total_loss = rpn_score_loss+rpn_reg_loss+roi_score_loss+roi_reg_loss+roi_mask_loss
            total_loss.backward()
            optimizer.step()
            
            avg_rpn_reg_loss += (rpn_reg_loss.item()-avg_rpn_reg_loss)/(idx+1) 
            avg_roi_reg_loss += (roi_reg_loss.item()-avg_roi_reg_loss)/(idx+1) 
            avg_rpn_score_loss += (rpn_score_loss.item()-avg_rpn_score_loss)/(idx+1)
            avg_roi_score_loss += (roi_score_loss.item()-avg_roi_score_loss)/(idx+1)
            avg_roi_mask_loss += (roi_mask_loss.item()-avg_roi_mask_loss)/(idx+1)
            avg_total_loss = avg_rpn_reg_loss+avg_roi_reg_loss+avg_rpn_score_loss+avg_roi_score_loss+avg_roi_mask_loss

            if (idx+1)%opt.plot_spot == 0:
                if os.path.exists(opt.debug_file):
                    ipdb.set_trace()

                # plot loss
                vis.plot('rpn_loc_loss', avg_rpn_reg_loss)
                vis.plot('rpn_cls_loss', avg_rpn_score_loss)
                vis.plot('roi_loc_loss', avg_roi_reg_loss)
                vis.plot('roi_cls_loss', avg_roi_score_loss)
                vis.plot('roi_mask_loss', avg_roi_score_loss)
                vis.plot('total_loss', avg_total_loss)

                # plot ground truth bboxes
                ori_img = inverse_normalize(img[0].cpu().numpy())
                gt_img = visdom_bbox(ori_img,
                                     gt_boxes[0].cpu().numpy(),
                                     gt_labels[0].cpu().numpy())
                vis.img('gt_img', gt_img)

                # plot ground truth mask
                gt_mask_img = visdom_mask(ori_img, img_id, dataset.coco)
                vis.img('gt_mask_img', gt_mask_img)

                # plot predict bboxes
                maskrcnn.eval()
                pre_boxes, pre_labels, pre_scores = maskrcnn.predict(img, gt_boxes, gt_labels, scale.item(), present='visualize')
                maskrcnn.train()
                # we need scale back
                # show predict img
                pred_img = visdom_bbox(ori_img,
                                       pre_boxes.cpu().numpy(),
                                       pre_labels.cpu().numpy(),
                                       pre_scores.cpu().numpy())
                vis.img('pred_img', pred_img)
                # show predict mask img


        # eval
        result = eval(maskrcnn, opt)
        # py visdom plot
        vis.plot('eval_map', result['map'])
        # log
        log_info = 'In round %d, lr:%.6f, test map:%.6f, avg rpn regloss is %.6f, avg roi regloss is %.6f, avg rpn scoreloss is %.6f, avg roi scoreloss is %.6f' % \
            (epoch, cur_lr, result['map'], avg_rpn_reg_loss, avg_roi_reg_loss, avg_rpn_score_loss, avg_roi_score_loss)
        print(log_info)
        vis.log(log_info)

        # save model if needed
        if result['map'] > cur_eval_map:
            cur_eval_map = result['map']
            save_path = mask_running_args.save_model_path % (time.strftime("%m%d_%H%M"), cur_eval_map)
            t.save(maskrcnn.state_dict(), save_path)
            opt.load_model_path = save_path
        elif result['map'] < cur_eval_map:     # make sure current one is the best one
            maskrcnn.load_state_dict(t.load(opt.load_model_path))

        # decay the model's learning rate
        if epoch == 9:  # it is a trick
            optimizer = scale_lr(optimizer, mask_running_args.lr_decay)
            cur_lr *= mask_running_args.lr_decay
        
if __name__ == '__main__':
    set_seed_everywhere(mask_running_args.seed, t.cuda.is_available())
    train(mask_running_args)
