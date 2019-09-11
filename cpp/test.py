import torch, time
import torch.nn as nn
import numpy as np
from PIL import Image
from conv4d import *

f1 = torch.randn(1,3,5,1,1,5)
#f1 = torch.randn(1,1,1,1,5,5)
f2 = f1.view(1,3,5,5)

conv4d = Conv4d(3,1,3,1,1,1,bias=True)
conv2d = torch.nn.Conv2d(3,1,3,1,1,1,bias=True)
conv4d.weight.data.fill_(1) 
conv2d.weight.data.fill_(1)

conv4d.bias.data.fill_(1)
conv2d.bias.data.fill_(1)

g1 = conv4d(f1)
g2 = conv2d(f2)

print(g1.view(g2.size()))
print(g2)


I = torch.randn(1,3,10,10)
J = torch.randn(1,3,10,10)

DH = 5
DW = 4

cost = Cost(DH,DW)

f = nn.Sequential(
        nn.Conv2d(3,32,3,1,1,1,bias=True),
        nn.ReLU(True),
        nn.Conv2d(32,32,3,1,1,1,bias=True))

model = nn.Sequential(
        Conv4d(1,4,3,1,1,1,bias=True),
        nn.ReLU(True),
        Conv4d(4,1,3,1,1,1,bias=True))

param = list(model.parameters()) + list(f.parameters())

opt = torch.optim.Adam(param, lr=1e-2)


for i in range(500):

    opt.zero_grad()

    f1 = f(I)
    f2 = f(J)

    c = cost(f1, f2)
    c = c.view(1, 1, 2*DH+1, 2*DW+1, 10, 10)

    t = model(c)

    L = t.pow(2).mean()
    L.backward()

    opt.step()
    print(L.item())
