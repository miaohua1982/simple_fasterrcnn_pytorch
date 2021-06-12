import torch as t
import numpy as np
from torch.autograd import Function
import align_roi_pool_mh as _backend

class RoIAlign_C(Function):
    @staticmethod
    def forward(ctx, image, boxes, box_ind, pool_height, pool_width, scale, sampling_ratio=-1, extrapolation_value=0):
        # crops = torch.zeros_like(image) not right
        pool_out = t.zeros([boxes.shape[0], image.shape[1], pool_height, pool_width])

        _backend.align_roi_pool_forward(
                image.detach().numpy(), boxes.detach().numpy(), box_ind.detach().numpy(),
                extrapolation_value, pool_height, pool_width, scale, sampling_ratio, pool_out.numpy())

        # save for backward
        im_size = image.size()
        ctx.save_for_backward(boxes, box_ind, t.tensor(im_size).int(), t.tensor(scale).float(), t.tensor(sampling_ratio).int())

        return pool_out
    
    @staticmethod
    def backward(ctx, grad_outputs):
        boxes, box_ind, im_size, scale, sampling_ratio = ctx.saved_tensors

        grad_outputs = grad_outputs.contiguous()
        grad_image = t.zeros(im_size.numpy().tolist())

        _backend.align_roi_pool_backward(grad_outputs.detach().numpy(), boxes.detach().numpy(), box_ind.detach().numpy(), \
                                         scale.item(), sampling_ratio.item(), grad_image.numpy())

        return grad_image, None, None, None, None, None
