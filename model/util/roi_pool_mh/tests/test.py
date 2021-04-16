import torch as t
import numpy as np
import roi_pool_mh as mh
from torchvision.ops import RoIPool

###############################################################################################################
## it is a version which call c++ implementation, you can get pure py version in roi_pool.py under util floder

class ROI_Pooling_C(t.autograd.Function):
    def __init__(self, roi_size, spatial_scale):
        ROI_Pooling_C.roi_size = roi_size
        ROI_Pooling_C.spatial_scale = spatial_scale
    
    @staticmethod
    def forward(ctx, feat_x, rois):
        n, c, _, _ = feat_x.shape
        num_rois = rois.shape[0]
        assert n == 1 # we only support 1 batch
        # feat to froward, and max position to record for backwarding
        feat = t.zeros(num_rois, c, ROI_Pooling_C.roi_size, ROI_Pooling_C.roi_size, dtype=t.float)
        feat_pos = t.zeros(num_rois, ROI_Pooling_C.roi_size*ROI_Pooling_C.roi_size, c, 2).int()
        # call my roi pool module forward function
        mh.roi_pooling_forward(feat_x.numpy(), rois.numpy(),\
            feat.numpy(), feat_pos.numpy(), \
            ROI_Pooling_C.spatial_scale, ROI_Pooling_C.roi_size)

        # save for backwarding
        ctx.save_for_backward(feat_pos, t.tensor(feat_x.shape))
        return feat
    
    @staticmethod
    def backward(ctx, grad_output):
        '''
        Args:
          grad_output: should be in shape num_rois*num_channel*roi_size*roi_size, default value: 128*512*7*7
        '''
        feat_pos, target_shape = ctx.saved_tensors
        roi_size = ROI_Pooling_C.roi_size
        n, c, h, w = grad_output.shape
        assert (h==roi_size) & (w==roi_size)
        # the grad to prop backward
        grad_input = t.zeros(target_shape.tolist(), dtype=t.float) # tensor can not be size
        # call my roi pool module backward function
        mh.roi_pooling_backward(grad_output.numpy(), feat_pos.numpy(), grad_input.numpy(), roi_size)

        return grad_input, None

if __name__ == '__main__':
    #feat_x = t.rand(1, 4, 37, 50)
    feat_x = t.rand(1, 2, 8, 8, requires_grad=True)
    #rois = t.tensor([[4,4,7,5], [1,3,3,7],[24,13,126,134]], dtype=t.float32)
    rois = t.tensor([[4,4,7,5], [1,3,3,7]], dtype=t.float32)

    #scale=1.0/16
    scale=1.0/2
    roi_size=7
    roi_pooling_lib = RoIPool((roi_size,roi_size),  scale)
    roi_indices = t.zeros(rois.shape[0])
    indices_and_rois = t.cat([roi_indices[:, None], rois], dim=1)
    feat1 = roi_pooling_lib(feat_x, indices_and_rois)

    roi_pooling = ROI_Pooling_C(roi_size, scale)
    feat2 = roi_pooling.apply(feat_x, rois) # 128,512,7,7
    
    print(t.all(feat1==feat2))

    # test backward
    f1 = feat1.sum()
    f1.backward()
    grad1 = feat_x.grad.clone()

    _ = feat_x.grad.zero_()
    f2 = feat2.sum()
    f2.backward()
    grad2 = feat_x.grad.clone()

    print(t.all(grad1==grad2))

