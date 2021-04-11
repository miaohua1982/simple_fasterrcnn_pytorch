import torch as t

class ROI_Pooling(t.autograd.Function):
    def forward(self, x, rois):
        roi_size = 7
        spatial_scal = 1.0/16

        n, _ = rois.shape
        n1, c, h, w = x.shape
        assert n1 == 1 # we only support 1 batch

    def backward(ctx, )

        self.roi = RoIPool((self.roi_size, self.roi_size), self.spatial_scale)

        # add batch index to roi, get shape 128*5, first column is batch index, next 4 columns are left-top x&y, right-down x&y
        indices_and_rois = t.cat([roi_indices[:, None], rois], dim=1) 
        # roi pooling
        pool = self.roi(x, indices_and_rois) # 
        
        
        
        # after pooling data has shape(128,512,7,7), roi_size = 7
        return feat