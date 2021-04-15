import torch as t
from torch.autograd import Variable

class MyReLu(t.autograd.Function):
    def __init__(self, argA, argB):
        MyReLu._argA = argA
        MyReLu._argB = argB
    @staticmethod
    def forward(ctx, input_):
        ctx.save_for_backward(input_)
        output = input_.clamp(min=0)
        print(MyReLu._argA)
        print(MyReLu._argB)
        return output
    @staticmethod
    def backward(ctx, grad_output):
        input_, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input_<0] = 0
        return grad_input