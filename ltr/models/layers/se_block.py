# -*- coding: utf-8 -*

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class SELayer(nn.Module):
    def __init__(self, channel, reduction=2):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
        self.gamma = Parameter(torch.zeros(1))

    def forward(self, x):
        b, c, _, _ = x.size()
        # print('input.shape = ',x.shape)
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * (1 + self.gamma * y.expand_as(x))


def params_count(net):
    list1 = []
    for p in net.parameters():
        # print('p-',p.shape)
        list1.append(p)
    # print('len(net.parameters)',len(list1))
    n_parameters = sum(p.numel() for p in net.parameters())
    print('-----Model param: {:.5f}M'.format(n_parameters / 1e6))
    # print('-----Model memory: {:.5f}M'.format(n_parameters/1e6))
    return n_parameters


if __name__ == "__main__":
    model = SELayer(16)
    print(params_count(model))
    with torch.no_grad():
        # x = torch.ones(4*3*16*64*44).reshape(4,3,16,64,44)
        x = torch.ones(1 * 16 * 256 * 256).reshape(1, 16, 256, 256)
        # a = Variable(a.cuda)
        print('x=', x.shape)
        # a,b = net(x)
        # print('a,b=',a.shape,b.shape)
        out_train = model(x)
        print('out_train=', out_train.shape)
