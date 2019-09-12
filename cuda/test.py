import torch
import numpy as np
import time
from PIL import Image
from conv4d import *

f1 = torch.randn(1,3,5,1,1,5).cuda()
#f1 = torch.randn(1,1,1,1,5,5)
f2 = f1.view(1,3,5,5)

conv4d = Conv4d(3,1,3,1,1,1,bias=False).cuda()
conv2d = torch.nn.Conv2d(3,1,3,1,1,1,bias=False).cuda()
conv4d.weight.data.fill_(1) 
conv2d.weight.data.fill_(1)

#conv4d.bias.data.fill_(1)
#conv2d.bias.data.fill_(1)

g1 = conv4d(f1)
g2 = conv2d(f2)

print(g1.view(g2.size()))
print(g2)



H = 128
W = 256
DH = 50
DW = 50

I = torch.randn(1,3,H,W).cuda()
J = torch.randn(1,3,H,W).cuda()


cost = Cost(DH,DW)

f = nn.Sequential(
        nn.Conv2d(3,32,3,1,1,1,bias=True),
        nn.ReLU(True),
        nn.Conv2d(32,32,3,1,1,1,bias=True))

model = nn.Sequential(
        Conv4d(1,1,3,1,1,1,bias=True))

f.cuda()
model.cuda()

param = list(model.parameters()) 

opt = torch.optim.Adam(param, lr=1e-3)


for i in range(500):

    opt.zero_grad()

    f1 = f(I)
    f2 = f(J)

    tic = time.time()
    c = cost(f1, f2)
    c = c.view(1, 1, 2*DH+1, 2*DW+1, H, W)
    torch.cuda.synchronize()
    toc = time.time()
#    print(toc - tic)


    tic = time.time()
    t = model(c)
    torch.cuda.synchronize()
    toc = time.time()
    print(toc - tic)

    L = t.pow(2).mean()

    tic = time.time()
    L.backward()
    torch.cuda.synchronize()
    toc = time.time()

#    print(toc - tic)

    opt.step()
    print(L.item())
