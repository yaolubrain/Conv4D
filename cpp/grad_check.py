import argparse
import torch
import random
from torch.autograd import Variable, gradcheck
from conv4d import *

f = torch.randn(2,3,5,5,5,5).double()
f.requires_grad_()

ksize = 3
stride = 2
padding = 1
input_channels = 3
output_channels = 4

w = torch.randn(output_channels, input_channels*ksize**4).double()
b = torch.randn(1,output_channels,1,1,1,1).double()

w = w.requires_grad_()
b = b.requires_grad_()

variables = [f, w, b, input_channels, output_channels, ksize, stride, padding]

if gradcheck(Conv4dFunction.apply, variables, eps=1e-3, atol=1e-5, rtol=1e-3):
    print('Ok')


