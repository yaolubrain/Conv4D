import torch, time
import numpy as np
from PIL import Image
from conv4d import *

f = torch.ones(1,1,3,3,3,3)
f.requires_grad_()

conv = Conv4d(1,1,1,1,0,False)

g = conv(f)

L = g.pow(2).sum()
L.backward()

print(g)
#print(f.size())
#print(g.size())
#print(conv.bias)

