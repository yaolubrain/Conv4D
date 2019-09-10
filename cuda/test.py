import torch
import numpy as np
import time
from PIL import Image
from sgmflow import *

H = 2
W = 2
C = 1

f1 = torch.ones(1,C,H,W).cuda()
f2 = 2*torch.ones(1,C,H,W).cuda()

f1.requires_grad_()
f2.requires_grad_()

M = Cost(1,1)
L = M(f1,f2).pow(2).sum()

L.backward()

print(f1.grad)
