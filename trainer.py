import torch as t
from tqdm import tqdm
import os
import ipdb

def train_one_epoch(model, optimizer, train_dataset, lr, preprocess_data, plot_step=100):
    # eval map
    cur_eval_map = 0.0
    # trace learning rate
    cur_lr = lr

    # set train mode
    model.train()

    # rest loss meter
    avg_rpn_reg_loss = 0.0
    avg_roi_reg_loss = 0.0
    avg_rpn_score_loss = 0.0
    avg_roi_score_loss = 0.0
    avg_total_loss = 0.0
    # one epoch
    for idx, one_obj_ds in tqdm(enumerate(train_dataset)):
        if t.cuda.is_available():
            one_obj_ds = [one_rec.cuda() for one_rc in one_obj_ds]
        else:
            one_obj_ds = list(one_obj_ds)

        optimizer.zero_grad()

        loss = model(one_obj_ds)

        total_loss = rpn_score_loss+rpn_reg_loss+roi_score_loss+roi_reg_loss
        
        total_loss.backward()

        optimizer.step()
        
        avg_rpn_reg_loss += (rpn_reg_loss.item()-avg_rpn_reg_loss)/(idx+1) 
        avg_roi_reg_loss += (roi_reg_loss.item()-avg_roi_reg_loss)/(idx+1) 
        avg_rpn_score_loss += (rpn_score_loss.item()-avg_rpn_score_loss)/(idx+1)
        avg_roi_score_loss += (roi_score_loss.item()-avg_roi_score_loss)/(idx+1)
        avg_total_loss = avg_rpn_reg_loss+avg_roi_reg_loss+avg_rpn_score_loss+avg_roi_score_loss


        if (idx+1)%plot_step == 0:
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
            pboxes, plabels, pscores = fasterrcnn.predict(img, gt_boxes, gt_labels, scale.item(), present='visualize')
            fasterrcnn.train()
            # we need scale back
            #pboxes = pboxes/scale
            pred_img = visdom_bbox(ori_img,
                                    pboxes.cpu().numpy(),
                                    plabels.cpu().numpy(),
                                    pscores.cpu().numpy())
            vis.img('pred_img', pred_img)

