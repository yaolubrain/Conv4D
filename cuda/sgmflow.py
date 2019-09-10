import math
import torch
from torch import nn
from torch.autograd import Function
import sgmflow_cuda

class CostFunction(Function):

    @staticmethod
    def forward(ctx, feat1, feat2, max_offset_h, max_offset_w):
        ctx.max_offset_h = max_offset_h
        ctx.max_offset_w = max_offset_w
        cost = sgmflow_cuda.compute_cost_volume_forward(feat1, feat2, max_offset_h, max_offset_w)
        ctx.save_for_backward(feat1, feat2)
        return cost

    @staticmethod
    def backward(ctx, grad_output):        
        feat1, feat2 = ctx.saved_variables    
        grad_feat1, grad_feat2 = sgmflow_cuda.compute_cost_volume_backward(grad_output, feat1, feat2, ctx.max_offset_h, ctx.max_offset_w)
        return grad_feat1, grad_feat2, None, None


class Cost(nn.Module):
    def __init__(self, max_offset_h, max_offset_w):
        super().__init__()
        self.max_offset_h = max_offset_h
        self.max_offset_w = max_offset_w
    
    def forward(self, feat1, feat2):
        return CostFunction.apply(feat1, feat2, self.max_offset_h, self.max_offset_w)


class PropFunction(Function):

    @staticmethod
    def forward(ctx, cost, edge, max_offset_h, max_offset_w):
        ctx.max_offset_h = max_offset_h
        ctx.max_offset_w = max_offset_w
        aggr, hori_pos_idx, hori_neg_idx, vert_pos_idx, vert_neg_idx = sgmflow_cuda.prop_forward(cost, edge, max_offset_h, max_offset_w)
        ctx.save_for_backward(edge, hori_pos_idx, hori_neg_idx, vert_pos_idx, vert_neg_idx)
        return aggr

    @staticmethod
    def backward(ctx, grad_aggr):        
        edge, hori_pos_idx, hori_neg_idx, vert_pos_idx, vert_neg_idx = ctx.saved_variables
        grad_cost, grad_edge = sgmflow_cuda.prop_backward(grad_aggr, edge, hori_pos_idx, hori_neg_idx, vert_pos_idx, vert_neg_idx, ctx.max_offset_h, ctx.max_offset_w)
        return grad_cost, grad_edge, None, None


class Prop(nn.Module):
    def __init__(self, max_offset_h, max_offset_w):
        super().__init__()
        self.max_offset_h = max_offset_h
        self.max_offset_w = max_offset_w
    
    def forward(self, cost, edge):
        return PropFunction.apply(cost, edge, self.max_offset_h, self.max_offset_w)

