import torch as t
import numpy as np
from torch.autograd import Function
import align_roi_pool_mh as _backend

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
