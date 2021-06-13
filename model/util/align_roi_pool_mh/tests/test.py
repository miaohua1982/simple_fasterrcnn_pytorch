import torch as t
import numpy as np
from torch.autograd import Function
import align_roi_pool_mh as _backend
from torchvision.ops import RoIPool, RoIAlign

class RoIAlign_C(Function):
    def __init__(self, pool_height, pool_width, spatial_scale, sampling_ratio=-1, extrapolation_value=0):
        RoIAlign_C.pool_height = pool_height
        RoIAlign_C.pool_width = pool_width
        RoIAlign_C.spatial_scale = spatial_scale
        RoIAlign_C.sampling_ratio = sampling_ratio
        RoIAlign_C.extrapolation_value = extrapolation_value

    @staticmethod
    def forward(ctx, image, boxes, box_ind):
        # crops = torch.zeros_like(image) not right
        pool_out = t.zeros([boxes.shape[0], image.shape[1], RoIAlign_C.pool_height, RoIAlign_C.pool_width])

        _backend.align_roi_pool_forward(
                image.detach().numpy(), boxes.detach().numpy(), box_ind.detach().numpy(),
                RoIAlign_C.extrapolation_value, RoIAlign_C.pool_height, RoIAlign_C.pool_width, RoIAlign_C.spatial_scale, RoIAlign_C.sampling_ratio, pool_out.numpy())

        # save for backward
        im_size = image.size()
        ctx.save_for_backward(boxes, box_ind, t.tensor(im_size).int())

        return pool_out
    
    @staticmethod
    def backward(ctx, grad_outputs):
        boxes, box_ind, im_size = ctx.saved_tensors

        grad_outputs = grad_outputs.contiguous()
        grad_image = t.zeros(im_size.numpy().tolist())

        _backend.align_roi_pool_backward(
                grad_outputs.detach().numpy(), boxes.detach().numpy(), box_ind.detach().numpy(), RoIAlign_C.spatial_scale, RoIAlign_C.sampling_ratio, grad_image.numpy())

        return grad_image, None, None


feat_x = t.rand(1, 4, 37, 50, requires_grad=True)
#feat_x = t.rand(1, 2, 8, 8, requires_grad=True)
rois = t.tensor([[4,4,7,5], [1,3,3,7],[24,13,126,134]], dtype=t.float32)
#rois = t.tensor([[4,4,7,5], [1,3,3,7]], dtype=t.float32)

scale=1.0/16
#scale=1.0/2
roi_size=7
roi_align_pooling_lib = RoIAlign((roi_size,roi_size), spatial_scale=scale, sampling_ratio=-1)
roi_indices = t.zeros(rois.shape[0])
indices_and_rois = t.cat([roi_indices[:, None], rois], dim=1)
feat1 = roi_align_pooling_lib(feat_x, indices_and_rois)

# compare 
roi_pool_lib1 = RoIAlign((roi_size,roi_size),  scale, sampling_ratio=2)
feat1_1 = roi_pool_lib1(feat_x, indices_and_rois)

roi_align_pooling = RoIAlign_C(roi_size, roi_size, scale)
feat3 = roi_align_pooling.apply(feat_x, rois, roi_indices)

# can not compare float data directly
feat1_np = np.round(feat1.detach().numpy(), 4)
feat3_np = np.round(feat3.detach().numpy(), 4)
print(np.all(feat1_np == feat3_np))

# test backward
f1 = feat1.sum()
f1.backward()
grad1 = feat_x.grad.clone()

_ = feat_x.grad.zero_()
f3 = feat3.sum()
f3.backward()
grad3 = feat_x.grad.clone()

grad1_np = np.round(grad1.detach().numpy(), 4)
grad3_np = np.round(grad3.detach().numpy(), 4)
print(np.all(grad1_np == grad3_np))


