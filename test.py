import torch
import numpy as np
import time
from PIL import Image
import conv4d_cpp
import conv4d_cuda

H = 4
W = 5
U = 3
V = 2
C_in = 4
C_out = 3

f1 = torch.randn(1,C_in,U,V,H,W)
W = torch.randn(C_out,C_in*3*3*3*3)

out_cpu = conv4d_cpp.conv4d_forward(f1, W, C_in, C_out, 3, 1, 1, 1)
out_gpu = conv4d_cuda.conv4d_forward(f1.cuda(), W.cuda(), C_in, C_out, 3, 1, 1, 1)

print(out_gpu[0,0,0,0,0,:])
print(out_cpu[0,0,0,0,0,:])

print(torch.sum(torch.abs(out_cpu - out_gpu.cpu())))

#conv4d.bias.data.fill_(1)
#conv2d.bias.data.fill_(1)


