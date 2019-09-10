import argparse
import torch
import random
from torch.autograd import Variable, gradcheck
from sgmflow import *

parser = argparse.ArgumentParser()
parser.add_argument('example', choices=['py', 'cpp', 'cuda'])
parser.add_argument('-b', '--batch-size', type=int, default=3)
parser.add_argument('-f', '--features', type=int, default=17)
parser.add_argument('-s', '--state-size', type=int, default=5)
parser.add_argument('-c', '--cuda', action='store_true')
options = parser.parse_args()

H = 2
W = 2
C = 32
B = 1

f1 = torch.randn(B,C,H,W).cuda().double()
f2 = torch.randn(B,C,H,W).cuda().double()

f1.requires_grad_()
f2.requires_grad_()

max_offset_h = 1
max_offset_w = 1

variables = [f1, f2, max_offset_h, max_offset_w]

if gradcheck(CostFunction.apply, variables, eps=1e-3, atol=1e-5, rtol=1e-3):
    print('Ok')

l = (2*max_offset_h+1) * (2*max_offset_w+1)
C = torch.randn(1,l,2,2).cuda()
e = 0.01*torch.randn(1,1,2,2).cuda()

C.requires_grad_()
e.requires_grad_()

variables = [C, e, max_offset_h, max_offset_w]

if gradcheck(PropFunction.apply, variables, eps=1e-3, atol=1e-5, rtol=1e-3):
    print('Ok')
