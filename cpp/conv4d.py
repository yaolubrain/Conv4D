import math
import torch
from torch import nn
from torch.autograd import Function
import conv4d_cpp

class CostFunction(Function):

    @staticmethod
    def forward(ctx, feat1, feat2, max_offset_h, max_offset_w):
        ctx.max_offset_h = max_offset_h
        ctx.max_offset_w = max_offset_w
        cost = sgmflow_cpp.compute_cost_volume_forward(feat1, feat2, max_offset_h, max_offset_w)
        ctx.save_for_backward(feat1, feat2)
        return cost

    @staticmethod
    def backward(ctx, grad_output):        
        feat1, feat2 = ctx.saved_variables
        grad_feat1, grad_feat2 = sgmflow_cpp.compute_cost_volume_backward(grad_output, feat1, feat2, ctx.max_offset_h, ctx.max_offset_w)
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
    def forward(ctx, inputs, weight, bias, input_channels, output_channels, ksize, stride, padding, dilation):
        ctx.input_channels = input_channels
        ctx.output_channels = output_channels
        ctx.ksize = ksize
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.bias = False

        outputs = conv4d_cpp.conv4d_forward(inputs, weight, \
                                            input_channels, output_channels, \
                                            ksize, stride, padding, dilation)

        if type(bias) != type(None):
            ctx.bias = True
            outputs += bias

        ctx.save_for_backward(inputs, weight)

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, weight = ctx.saved_variables

        grad_inputs, grad_weight = conv4d_cpp.conv4d_backward(grad_outputs, inputs, weight, ctx.input_channels, ctx.output_channels, ctx.ksize, ctx.stride, ctx.padding, ctx.dilation)

        grad_bias = None

        if ctx.bias:
          grad_bias = grad_outputs.sum((2,3,4), keepdim=True)

        return grad_inputs, grad_weight, grad_bias, None, None, None, None, None


class Conv4d(nn.Module):

    def __init__(self, input_channels, output_channels, ksize=1, stride=1, padding=0, dilation=1, bias=True):

        super().__init__()

        self.input_channels = input_channels
        self.output_channels = output_channels
        self.ksize = ksize
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        self.weight = torch.randn(output_channels, input_channels*ksize**4)

        if bias:
            self.bias = torch.randn(1,output_channels,1,1,1,1)
        else:
            self.bias = None
    
    def forward(self, inputs):
        return Conv4dFunction.apply(inputs, self.weight, self.bias, self.input_channels, self.output_channels, self.ksize, self.stride, self.padding, self.dilation)
