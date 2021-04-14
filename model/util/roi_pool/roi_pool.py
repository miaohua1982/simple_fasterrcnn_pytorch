import torch as t
import numpy as np
from torchvision.ops import RoIPool

class ROI_Pooling(t.autograd.Function):
    def __init__(self, roi_size, spatial_scale):
        ROI_Pooling.roi_size = roi_size
        ROI_Pooling.spatial_scale = spatial_scale
    
    @staticmethod
    def forward(ctx, feat_x, rois):
        n, c, h, w = feat_x.shape
        assert n == 1 # we only support 1 batch
    
        feat = []
        feat_pos = []
        for one_roi in rois:
            scale_roi = t.round(one_roi*ROI_Pooling.spatial_scale+0.000001)
            
            x = t.linspace(scale_roi[0], scale_roi[2]+1, ROI_Pooling.roi_size+1)
            y = t.linspace(scale_roi[1], scale_roi[3]+1, ROI_Pooling.roi_size+1)
            
            gy1, gx1 = t.meshgrid(y[:ROI_Pooling.roi_size], x[:ROI_Pooling.roi_size])
            gy2, gx2 = t.meshgrid(y[1:], x[1:])
            # with shape 49*2
            bins_lefttop = t.cat([gx1.flatten().view(-1,1),gy1.flatten().view(-1,1)], dim=1)
            bins_rightdown = t.cat([gx2.flatten().view(-1,1),gy2.flatten().view(-1,1)], dim=1)
            bins_lefttop = bins_lefttop.floor().int()
            bins_rightdown = bins_rightdown.ceil().int()
            # clip to (0~w) & (0~h)
            bins_lefttop[:,0].clamp_(min=0, max=w)
            bins_lefttop[:,1].clamp_(min=0, max=h)
            bins_rightdown[:,0].clamp_(min=0, max=w)
            bins_rightdown[:,1].clamp_(min=0, max=h)

            feat_val = t.zeros(c, ROI_Pooling.roi_size*ROI_Pooling.roi_size)
            feat_rp = t.zeros(ROI_Pooling.roi_size*ROI_Pooling.roi_size, c, 2).long()

            for idx, (ltxy, rdxy) in enumerate(zip(bins_lefttop, bins_rightdown)):
                fx = feat_x[:,:,ltxy[1]:rdxy[1],ltxy[0]:rdxy[0]].reshape(c, -1)
                fw = rdxy[0]-ltxy[0]
                feat_val[:,idx], feat_args = fx.max(dim=1)
                
                feat_rp[idx,:,0]=feat_args//fw+ltxy[1]   #y
                feat_rp[idx,:,1]=feat_args%fw+ltxy[0]    #x

            feat.append(feat_val.view(c, ROI_Pooling.roi_size, ROI_Pooling.roi_size).unsqueeze(0))    #shape to 4 dims
            feat_pos.append(feat_rp.unsqueeze(0)) #shapt to 4 dims
        
        feat = t.cat(feat, dim=0)
        feat_pos = t.cat(feat_pos, dim=0)
        ctx.save_for_backward(feat_pos)
        ctx.save_for_backward(x.shape)
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
        grad_input = t.zeros(target_shape)
        print(grad_output.shape)
        print(target_shape)
        
        for ch, one_bs_grad in enumerate(grad_output):   # typical value 128*
            for idx in range(roi_size*roi_size): # typical value is 7*7
                pos_y = idx/roi_size
                pos_x = idx%roi_size
                for one_c_ind in range(c):  # typical value is 512
                    grad_input[:, one_c_ind, feat_pos[ch, idx, one_c_ind, 0], feat_pos[ch, idx, one_c_ind, 1]] += \
                        grad_output[ch, one_c_ind, pos_y, pos_x]
        return grad_input, None

if __name__ == '__main__':
    feat_x = t.rand(1, 4, 37, 50)
    rois = t.tensor([[4,4,7,5], [1,3,3,7],[24,13,126,134]], dtype=t.float32)
    #rois = t.tensor([[4,4,7,5], [1,3,3,7]], dtype=t.float32)

    scale=1.0/16
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
