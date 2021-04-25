from argparse import Namespace


running_args = Namespace(
                 #------------------------------
                 # random seed setup
                 seed=1337,
                 #-----------------------------
                 # model laod & save
                 load_model_path=None,
                 save_model_path="model_storage/model_%s_%.6f.pth",
                 # param for optimizer
                 # 0.0005 in origin paper but 0.0001 in tf-faster-rcnn
                 use_adam = False,
                 weight_decay = 0.0005,
                 lr_decay = 0.1,  # 1e-3 -> 1e-4
                 learning_rate=1e-3,
                 #------------------------------
                 #train set
                 batch_size=1,
                 num_classes=20,
                 num_epochs=13,
                 use_drop=False,
                 feat_stride=16,
                 plot_spot=100,
                 debug_file='/tmp/debug',
                 vis_env='fastercnn_mh',
                 #------------------------------
                 dataset_base_path="./data/VOCdevkit/VOC2007/",
                 min_img_size = 600,
                 max_img_size = 1000,
                 #------------------------------
                 #proposal creator for choosing good samples to train roi head network
                 pre_train_num=12000,
                 post_train_num=6000,
                 pre_test_num=2000,
                 post_test_num=300,
                 min_roi_size=16,
                 proposal_nms_thresh=0.7,
                 #-----------------------------
                 #anchor target creator for rpn training
                 n_sample=256,
                 pos_ratio=0.5,
                 neg_iou_thresh=0.3,
                 pos_iou_thresh=0.7,
                 # sigma for l1_smooth_loss
                 rpn_sigma = 3.,
                 #----------------------------
                 #proposal target creator for roi head network training
                 n_roi_sample=128,
                 pos_roi_ratio=0.25,
                 pos_roi_iou_thresh=0.5,
                 neg_iou_thresh_hi=0.5,
                 neg_iou_thresh_lo=0.0,
                 loc_normalize_mean=[0., 0., 0., 0.],
                 loc_normalize_std=[0.1, 0.1, 0.2, 0.2],
                 #----------------------------
                 #region proposal network
                 n_base_anchors_num=9,
                 #----------------------------
                 #roi header
                 roi_size=7,
                 spatial_scale=1./16,
                 #----------------------------
                 #smooth l1 loss calc
                 roi_sigma=1.0
                 )
