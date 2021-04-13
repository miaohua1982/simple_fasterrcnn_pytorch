import torch as t
import numpy as np

class ROI_Pooling(t.autograd.Function):
    def __init__(self, roi_size, spatial_scal):
        ROI_Pooling.roi_size = roi_size
        ROI_Pooling.spatial_scal = spatial_scal
    
    @staticmethod
    def forward(ctx, x, rois):
        n1, c, h, w = x.shape
        assert n1 == 1 # we only support 1 batch
    
        feat = []
        feat_pos = []
        for one_roi in rois:
            scale_roi = one_roi*ctx.spatial_scal
            
            x = t.linspace(scale_roi[0], scale_roi[2], ctx.roi_size+1)
            y = t.linspace(scale_roi[1], scale_roi[3], ctx.roi_size+1)
            
            gy1, gx1 = t.meshgrid(y[:ctx.roi_size], x[:ctx.roi_size])
            gy2, gx2 = t.meshgrid(y[1:], x[1:])
            # with shape 49*2
            bins_lefttop = t.cat([gx1.flatten().view(-1,1),gy1.flatten().view(-1,1)], dim=1)
            bins_rightdown = t.cat([gx2.flatten().view(-1,1),gy2.flatten().view(-1,1)], dim=1)
            bins_lefttop = bins_lefttop.floor().int()
            bins_rightdown = bins_rightdown.ceil().int()+1
            # clip to (0~w) & (0~h)
            bins_lefttop[0].clip_(min=0, max=w)
            bins_lefttop[1].clip_(min=0, max=h)
            bins_rightdown[0].clip_(min=0, max=w)
            bins_rightdown[1].clip_(min=0, max=h)

            feat_val = t.zeros(c, ctx.roi_size*ctx.roi_size)
            feat_rp = t.zeros(ctx.roi_size*ctx.roi_size, c, 2).long()

            for idx, ltxy, rdxy in enumerate(zip(bins_lefttop, bins_rightdown)):
                fx = x[:,:,ltxy[1]:rdxy[1],ltxy[0]:rdxy[0]].reshape(c, -1)
                fw = rdxy[0]-ltxy[0]
                feat_val[:,idx], feat_args = fx.max(dim=1)
                
                feat_rp[idx,:,0]=feat_args%fw+ltxy[1]
                feat_rp[idx,:,1]=feat_args//fw+ltxy[0]

            feat.append(feat_val)
            feat_pos.append(feat_rp)
        
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
        n, c, h, w = grad_output.shape
        assert (h==ctx.roi_size) & (w==ctx.roi_size)
        grad_input = t.zeros(target_shape)
        
        for ch, one_bs_grad in enumerate(grad_output):   # typical value 128*
            for idx in range(ctx.roi_size*ctx.roi_size): # typical value is 7*7
                pos_x = idx/ctx.roi_size
                pos_y = idx%ctx.roi_size
                for one_c_ind in range(c):  # typical value is 512
                    grad_input[:, one_c_ind, feat_pos[ch, idx, one_c_ind, 0], feat_pos[ch, idx, one_c_ind, 1]] += \
                        grad_output[ch, one_c_ind, pos_x, pos_y]
        return grad_input, None