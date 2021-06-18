class ConfigBase(object):
    #------------------------------
    # random seed setup
    seed=1337
    #-----------------------------
    #the flag for wether using torch version anchor target creator & proposal creator & proposal target creator or numpy version
    #when True means all operations are done on gpu
    #which is a little bit speedy than False
    all_torch=False
    #-----------------------------
    # model load & save
    load_model_path=None
    save_model_path="model_storage/model_%s_%.6f.pth"
    # param for optimizer
    # 0.0005 in origin paper but 0.0001 in tf-faster-rcnn
    use_adam = False
    weight_decay = 0.0005
    lr_decay = 0.1  # 1e-3 -> 1e-4
    learning_rate=1e-3
    # backbone model selection vgg16 or resnet
    backbone = "vgg16"
    #------------------------------
    resnet_layers=101
    #------------------------------
    #train set
    batch_size=1
    num_classes=20
    num_epochs=14
    use_drop=False
    feat_stride=16
    plot_spot=100
    debug_file='/tmp/debug'
    vis_env='fastercnn_mh'
    #------------------------------
    dataset_base_path="./data/VOCdevkit/VOC2007/"
    min_img_size = 600
    max_img_size = 1000
    #------------------------------
    #proposal creator for choosing good samples to train roi head network
    pre_train_num=12000
    post_train_num=6000
    pre_test_num=2000
    post_test_num=300
    min_roi_size=16
    proposal_nms_thresh=0.7
    #-----------------------------
    #anchor target creator for rpn training
    n_sample=256
    pos_ratio=0.5
    neg_iou_thresh=0.3
    pos_iou_thresh=0.7
    # sigma for l1_smooth_loss
    rpn_sigma = 3.
    #----------------------------
    #proposal target creator for roi head network training
    n_roi_sample=128
    pos_roi_ratio=0.25
    pos_roi_iou_thresh=0.5
    neg_iou_thresh_hi=0.5
    neg_iou_thresh_lo=0.0
    loc_normalize_mean=[0.,0.,0.,0.]
    loc_normalize_std=[0.1,0.1,0.2,0.2]
    #----------------------------
    #region proposal network
    n_base_anchors_num=9
    #----------------------------
    #roi header
    roi_size=7
    spatial_scale=1./16
    #----------------------------
    #smooth l1 loss calc
    roi_sigma=1.0
                 
class Fasterrcnn_Config(ConfigBase):
    pass

class Maskrcnn_Config(ConfigBase):
    # global setting

    # according to paper
    # ... with a learning rate of 0.02 which is decreased by 10 at the 120k iteration. We use a weight decay of 0.0001 and momentum of 0.9

    # Learning rate and momentum
    # The Mask RCNN paper uses lr=0.02, but on TensorFlow it causes weights to explode. Likely due to differences in optimizer implementation.
    weight_decay = 0.0001
    lr_decay = 0.1  # 1e-3 -> 1e-4
    learning_rate= 0.02    # or 0.001 according to  Matterport's mask rcnn implementation
    momentum = 0.9
    
    # according to paper
    # ... We train on 8 GPUs (so effective minibatch size is 16) for 160k iterations, with a learning rate of 0.02 which is decreased by 10 at the 120k iteration
    batch_size = 16
    num_epochs = 160
    lr_dec_epochs = 120
    steps_per_epoch = 1000
    # backbone network
    backbone = "resnet101"
    num_classes = 80     # coco dataset has 80 classes
    min_img_size = 800
    max_img_size = 1024

    image_size = (1024,1024)
    # set for instance json file
    dataset_base_path = 'E:\\Datasets\\COCO2017'  # '../datasets/coco2017'  #
    # base anchor box generation
    backbone_stride = [4,8,16,32,64]
    anchor_ratios = [0.5,1,2]
    anchor_scales = [32,64,128,256,512]
    # loss visualize
    vis_env='maskrcnn'
    # generate anchor box for every cell in feature map
    anchor_stride = 1
    # proposal creator for choosing good samples to train roi head network
    pre_train_num=6000
    post_train_num=2000
    pre_test_num=6000
    post_test_num=1000  # according to paper, at test time, the proposal number is 300 for the C4 backbone (as in [36]) and 1000 for FPN (as in [27]).
    skip_small_obj=False  #wether to skip small object
    # proposal target creator for roi head network training
    pos_roi_ratio=0.25  # with a ratio of 1:3 of positive to negatives [12].
    n_roi_sample=512    # N is 64 for the C4 backbone (as in [12, 36]) and 512 for FPN (as in [27]).
    gt_mask_size = (28, 28)
    # roi header
    mask_roi_size = 14
    # eval
    eval_result_path = 'model_storage/eval_result_{}.txt'

running_args = Fasterrcnn_Config()

mask_running_args = Maskrcnn_Config()
