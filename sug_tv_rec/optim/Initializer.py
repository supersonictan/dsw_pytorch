import re

import numpy as np
from torch import nn
from typing import (List,Any,Union,Dict,Callable,Optional,Tuple,Generator,Collection,Iterable,Match,Iterator)


class Initializer(object):
    def __call__(self, model: nn.Module):
        raise NotImplementedError("Initializer must implement this method")



"""
# return self.unk_init(torch.Tensor(self.dim))  这是原来的代码，换成下面的。
return self.unk_init(torch.Tensor(1,self.dim)).squeeze(0)
# self.dim它是一个数，当unk_init = init.xavier_uniform_时候，只传进去一个数他就会报错所以要在前面加上一个1，初始化之后还要用squeeze(0)把多余的维度抽出去，是填充的词向量也是一维的
"""


class XavierNormal(Initializer):
    def __init__(self, gain=1, pattern="."):
        self.gain = gain
        self.pattern = pattern
        super(XavierNormal, self).__init__()

    def __call__(self, submodel: nn.Module):
        for n, p in submodel.named_parameters():
            print("init {} ndimension:{} {}".format(n, p.ndimension(), p.shape))
            if re.search(self.pattern, n) and 'word_embed' not in n:
                if "bias" in n:
                    nn.init.constant_(p, val=0)
                elif p.requires_grad:
                    # print('before ' + str(p))
                    nn.init.xavier_normal_(p, gain=self.gain)
                    # print('after ' + str(p))



class KaimingNormal(Initializer):
    def __init__(self, a=0, mode="fan_in", nonlinearity="leaky_relu", pattern="."):
        self.a = a
        self.mode = mode
        self.nonlinearity = nonlinearity
        self.pattern = pattern
        super(KaimingNormal, self).__init__()

    def __call__(self, submodel: nn.Module):
        for n, p in submodel.named_parameters():
            print("init {} ndimension:{} {}".format(n, p.ndimension(), p.shape))
            if "bias" in n:
                nn.init.constant_(p, val=0)
            elif p.requires_grad:
                if 'norm' in n:
                    # print("before unsqueeze ndimension:{} {}".format(p.ndimension(), p.shape))
                    p = p.unsqueeze(0)
                    # print("after unsqueeze ndimension:{} {}".format(p.ndimension(), p.shape))
                    nn.init.kaiming_normal_(p, a=self.a, mode=self.mode, nonlinearity=self.nonlinearity)
                    p = np.squeeze(p)
                    # print("after squeeze ndimension:{} {}".format(p.ndimension(), p.shape))
                else:
                    nn.init.kaiming_normal_(p, a=self.a, mode=self.mode, nonlinearity=self.nonlinearity)