import torch
from torch.autograd import Function
import crop_and_size_mh as _backend


class CropAndResizeFunction(Function):

    def __init__(self, crop_height, crop_width, extrapolation_value=0):
        CropAndResizeFunction.crop_height = crop_height
        CropAndResizeFunction.crop_width = crop_width
        CropAndResizeFunction.extrapolation_value = extrapolation_value

    @staticmethod
    def forward(ctx, image, boxes, box_ind):
        # crops = torch.zeros_like(image) not right
        crops = torch.zeros([boxes.shape[0], image.shape[1], CropAndResizeFunction.crop_height, CropAndResizeFunction.crop_width])

        _backend.crop_and_resize_forward(
                image, boxes, box_ind,
                CropAndResizeFunction.extrapolation_value, CropAndResizeFunction.crop_height, CropAndResizeFunction.crop_width, crops)

        # save for backward
        im_size = image.size()
        ctx.save_for_backward(boxes, box_ind, torch.tensor(im_size).long())

        return crops
    
    @staticmethod
    def backward(ctx, grad_outputs):
        boxes, box_ind, im_size = ctx.saved_tensors

        grad_outputs = grad_outputs.contiguous()
        grad_image = torch.zeros_like(grad_outputs).resize_(*im_size)

        _backend.crop_and_resize_backward(
                grad_outputs, boxes, box_ind, grad_image
            )

        return grad_image, None, None
