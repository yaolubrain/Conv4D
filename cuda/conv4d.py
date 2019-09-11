import math
import torch
from torch import nn
from torch.autograd import Function
import conv4d_cuda

class CostFunction(Function):

    @staticmethod
    def forward(ctx, feat1, feat2, max_offset_h, max_offset_w):
        ctx.max_offset_h = max_offset_h
        ctx.max_offset_w = max_offset_w
        cost = conv4d_cuda.compute_cost_volume_forward(feat1, feat2, max_offset_h, max_offset_w)
        ctx.save_for_backward(feat1, feat2)
        return cost

    @staticmethod
    def backward(ctx, grad_output):        
        feat1, feat2 = ctx.saved_variables
        grad_feat1, grad_feat2 = conv4d_cuda.compute_cost_volume_backward(grad_output, feat1, feat2, ctx.max_offset_h, ctx.max_offset_w)
        return grad_feat1, grad_feat2, None, None


class Cost(nn.Module):
    def __init__(self, max_offset_h, max_offset_w):
        super().__init__()
        self.max_offset_h = max_offset_h
        self.max_offset_w = max_offset_w
    
    def forward(self, feat1, feat2):
        return CostFunction.apply(feat1, feat2, self.max_offset_h, self.max_offset_w)


class Conv4dFunction(Function):

    @staticmethod
    def forward(ctx, inputs, weight, bias, channels_in, channels_out, ksize, stride, padding, dilation):
        ctx.channels_in = channels_in
        ctx.channels_out = channels_out
        ctx.ksize = ksize
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.bias = False

        U_in = inputs.size(2)
        V_in = inputs.size(3)
        H_in = inputs.size(4)
        W_in = inputs.size(5)

        U_out = (U_in + 2*padding - dilation*(ksize - 1) - 1) / stride + 1;
        V_out = (V_in + 2*padding - dilation*(ksize - 1) - 1) / stride + 1;
        H_out = (H_in + 2*padding - dilation*(ksize - 1) - 1) / stride + 1;
        W_out = (W_in + 2*padding - dilation*(ksize - 1) - 1) / stride + 1;

        assert(U_out > 0)
        assert(V_out > 0)
        assert(H_out > 0)
        assert(W_out > 0)

        outputs = conv4d_cuda.conv4d_forward(inputs, weight, \
                                             channels_in, channels_out, \
                                             ksize, stride, padding, dilation)

        if type(bias) != type(None):
            ctx.bias = True
            outputs += bias

        ctx.save_for_backward(inputs, weight)

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, weight = ctx.saved_variables

        grad_inputs, grad_weight = conv4d_cuda.conv4d_backward(grad_outputs, inputs, weight, ctx.channels_in, ctx.channels_out, ctx.ksize, ctx.stride, ctx.padding, ctx.dilation)

        grad_bias = None

        if ctx.bias:
          grad_bias = grad_outputs.sum((2,3,4), keepdim=True)

        return grad_inputs, grad_weight, grad_bias, None, None, None, None, None, None


class Conv4d(nn.Module):

    def __init__(self, channels_in, channels_out, ksize=1, stride=1, padding=0, dilation=1, bias=True):

        super().__init__()

        self.channels_in = channels_in
        self.channels_out = channels_out
        self.ksize = ksize
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        self.weight = nn.Parameter(torch.randn(channels_out, channels_in*ksize**4))

        if bias:
            self.bias = nn.Parameter(torch.randn(1,channels_out,1,1,1,1))
        else:
            self.bias = None
    
    def forward(self, inputs):
        return Conv4dFunction.apply(inputs, self.weight, self.bias, self.channels_in, self.channels_out, self.ksize, self.stride, self.padding, self.dilation)
