from __future__ import  absolute_import

import os
import ipdb
import time
import torch as t
import numpy as np
from torch.utils import data as data_
from data.voc_dataset import Voc_Dataset
from data.util import inverse_normalize
from config.config import running_args
from model.util.eval_ap import eval_detection_voc
from model.backbone.vgg16 import decom_vgg16
from model.backbone.resnet import decom_resnet
from model.fasterrcnn.fasterrcnn import FasterRCNN
from model.util.vis_tool import Visualizer, visdom_bbox
from tqdm import tqdm

def set_seed_everywhere(seed, cuda):
    np.random.seed(seed)
    t.manual_seed(seed)
    if cuda:
        t.cuda.manual_seed_all(seed)

def get_optimizer(opt, model):
    """
    return optimizer, It could be overwriten if you want to specify 
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
    dataset = Voc_Dataset(opt.dataset_base_path,min_size=opt.min_img_size, max_size=opt.max_img_size, split='test')
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

        with t.no_grad():
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
    dataset = Voc_Dataset(opt.dataset_base_path, min_size=opt.min_img_size, max_size=opt.max_img_size, split='trainval')
    train_dataset = data_.DataLoader(dataset, batch_size=1, shuffle=False)

    # model
    if opt.backbone == 'vgg16':
        backbone, classifier = decom_vgg16(opt)
    if opt.backbone == 'resnet':
        backbone, classifier = decom_resnet(opt)
    fasterrcnn = FasterRCNN(opt.num_classes, backbone, classifier)
    fasterrcnn = fasterrcnn.cuda() if t.cuda.is_available() else fasterrcnn

    # optimizer
    optimizer = get_optimizer(opt, fasterrcnn)

    # visdom
    vis = Visualizer(env=opt.vis_env)
  
    # eval map
    cur_eval_map = 0.0
    # trace learning rate
    cur_lr = opt.learning_rate

    # set train mode
    fasterrcnn.train()

    # train
    t.autograd.set_detect_anomaly(True)
    for epoch in range(opt.num_epochs):
        # rest loss meter
        avg_rpn_reg_loss = 0.0
        avg_roi_reg_loss = 0.0
        avg_rpn_score_loss = 0.0
        avg_roi_score_loss = 0.0
        avg_total_loss = 0.0
        # one epoch
        for idx, one_obj_ds in tqdm(enumerate(train_dataset)):
            img, gt_boxes, gt_labels, _, scale = one_obj_ds

            if t.cuda.is_available():
                img, gt_boxes, gt_labels = img.cuda(), gt_boxes.cuda(), gt_labels.cuda()

            optimizer.zero_grad()

            rpn_score_loss, rpn_reg_loss, roi_score_loss, roi_reg_loss, roi_reg_locs, roi_scores, \
            sample_rois = fasterrcnn(img, gt_boxes, gt_labels, scale.item())
            #print('rpn loc loss %.6f, rpn cls loss %.6f, roi loc loss %.6f, roi cls loss %.6f'%(rpn_reg_loss, rpn_score_loss, roi_reg_loss, roi_score_loss))

            total_loss = rpn_score_loss+rpn_reg_loss+roi_score_loss+roi_reg_loss
            
            total_loss.backward()

            optimizer.step()
            
            avg_rpn_reg_loss += (rpn_reg_loss.item()-avg_rpn_reg_loss)/(idx+1) 
            avg_roi_reg_loss += (roi_reg_loss.item()-avg_roi_reg_loss)/(idx+1) 
            avg_rpn_score_loss += (rpn_score_loss.item()-avg_rpn_score_loss)/(idx+1)
            avg_roi_score_loss += (roi_score_loss.item()-avg_roi_score_loss)/(idx+1)
            avg_total_loss = avg_rpn_reg_loss+avg_roi_reg_loss+avg_rpn_score_loss+avg_roi_score_loss


            if (idx+1)%opt.plot_spot == 0:
                if os.path.exists(opt.debug_file):
                    ipdb.set_trace()

                # plot loss
                vis.plot('rpn_loc_loss', avg_rpn_reg_loss)
                vis.plot('rpn_cls_loss', avg_rpn_score_loss)
                vis.plot('roi_loc_loss', avg_roi_reg_loss)
                vis.plot('roi_cls_loss', avg_roi_score_loss)
                vis.plot('total_loss', avg_total_loss)

                # plot groud truth bboxes
                ori_img = inverse_normalize(img[0].cpu().numpy())
                gt_img = visdom_bbox(ori_img,
                                     gt_boxes[0].cpu().numpy(),
                                     gt_labels[0].cpu().numpy())
                vis.img('gt_img', gt_img)

                # plot predicti bboxes
                fasterrcnn.eval()
                with t.no_grad():
                    pboxes, plabels, pscores = fasterrcnn.predict(img, gt_boxes, gt_labels, scale.item(), present='visualize')
                fasterrcnn.train()
                # we need scale back
                #pboxes = pboxes/scale
                pred_img = visdom_bbox(ori_img,
                                       pboxes.cpu().numpy(),
                                       plabels.cpu().numpy(),
                                       pscores.cpu().numpy())
                vis.img('pred_img', pred_img)

        # eval
        result = eval(fasterrcnn, opt)
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
            save_path = running_args.save_model_path % (time.strftime("%m%d_%H%M"), cur_eval_map)
            t.save(fasterrcnn.state_dict(), save_path)

        # decay the model's learning rate
        if epoch == 9:  # it is a trick
            optimizer = scale_lr(optimizer, running_args.lr_decay)
            cur_lr *= running_args.lr_decay
        

if __name__ == '__main__':
    set_seed_everywhere(running_args.seed, t.cuda.is_available())
    train(running_args)
