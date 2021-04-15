import torch as t
import numpy as np
import roi_pool_mh as mh
from torchvision.ops import RoIPool

class ROI_Pooling(t.autograd.Function):
    def __init__(self, roi_size, spatial_scale):
        ROI_Pooling.roi_size = roi_size
        ROI_Pooling.spatial_scale = spatial_scale
    
    @staticmethod
    def forward(ctx, feat_x, rois):
        n, c, h, w = feat_x.shape
        num_rois = rois.shape[0]
        assert n == 1 # we only support 1 batch
        
        feat = t.zeros(num_rois, c, ROI_Pooling.roi_size, ROI_Pooling.roi_size, dtype=t.float)
        feat_pos = t.zeros(num_rois, ROI_Pooling.roi_size*ROI_Pooling.roi_size, c, 2).long()
        roi_pooling_forward(feat_x.numpy(), rois.numpy(),\
            feat.numpy(), feat_pos.numpy(), \
            ROI_Pooling.spatial_scale, ROI_Pooling.roi_size)

        # save for backward
        ctx.save_for_backward(feat_pos, t.tensor(feat_x.shape))
        return feat
    
    @staticmethod
    def backward(ctx, grad_output):
        '''
        Args:
          grad_output: should be in shape num_rois*num_channel*roi_size*roi_size, default value: 128*512*7*7
        '''
        feat_pos, target_shape = ctx.saved_tensors
        roi_size = ROI_Pooling.roi_size
        n, c, h, w = grad_output.shape
        assert (h==roi_size) & (w==roi_size)
        grad_input = t.zeros(target_shape.tolist())  # tensor can not be size

        for ch, one_bs_grad in enumerate(grad_output):   # typical value 128*
            for idx in range(roi_size*roi_size): # typical value is 7*7
                pos_y = idx//roi_size
                pos_x = idx%roi_size
                for one_c_ind in range(c):  # typical value is 512
                    grad_input[:, one_c_ind, feat_pos[ch, idx, one_c_ind, 0], feat_pos[ch, idx, one_c_ind, 1]] += \
                        grad_output[ch, one_c_ind, pos_y, pos_x] #here note the y&x position, not x,y, but y,x
        
        return grad_input, None

if __name__ == '__main__':
    #feat_x = t.rand(1, 4, 37, 50)
    feat_x = t.rand(1, 2, 8, 8, requires_grad=True)
    #rois = t.tensor([[4,4,7,5], [1,3,3,7],[24,13,126,134]], dtype=t.float32)
    rois = t.tensor([[4,4,7,5], [1,3,3,7]], dtype=t.float32)

    #scale=1.0/16
    scale=1.0/2
    roi_size=7
    roi = RoIPool((roi_size,roi_size),  scale)
    roi_indices = t.zeros(rois.shape[0])
    indices_and_rois = t.cat([roi_indices[:, None], rois], dim=1)
    feat1 = roi(feat_x, indices_and_rois)

    roi_pooling = ROI_Pooling(roi_size, scale)
    feat2 = roi_pooling.apply(feat_x, rois) # 128,512,7,7
    
    print(t.all(feat1==feat2))
    f = feat.sum()
    f.backward()
