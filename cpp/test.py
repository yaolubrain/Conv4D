import torch, time
import numpy as np
from PIL import Image
from conv4d import *

f1 = torch.randn(1,1,1,1,5,5)
f2 = f1.view(1,1,5,5)


conv4d = Conv4d(1,1,3,1,1,1,bias=True)
conv2d = torch.nn.Conv2d(1,1,3,1,1,1,bias=True)
conv4d.weight.fill_(1) 
conv2d.weight.data.fill_(1)

conv4d.bias.fill_(1)
conv2d.bias.data.fill_(1)

g1 = conv4d(f1)
g2 = conv2d(f2)

print(g1)
print(g2)
#print(f.size())
#print(g.size())
#print(conv.bias)

