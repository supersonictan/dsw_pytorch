from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn, Tensor



def dense_layer(inp: int, out: int, p: float = 0.0, bn=False):
    layers = [nn.Linear(inp, out), nn.LeakyReLU(inplace=True)]
    if bn:
        layers.append(nn.BatchNorm1d(out))
    layers.append(nn.Dropout(p))
    return nn.Sequential(*layers)




class DeepHead(nn.Module):
    def __init__(self, hidden_layers, batchnorm=False, dropout=None, prefix_embed=None, sug_embed=None, continuous_cols=None):
        """
        :param hidden_layers:           List[int]
        :param batchnorm:               bool
        :param dropout:                 List[float]
        :param embed_input:             List[Tuple[str, int, int]]
        :param embed_p:                 float
        :param continuous_cols:         List[str]
        """

        super(DeepHead, self).__init__()

        if not dropout:
            dropout = [0.0] * len(hidden_layers)

        # Step2. 根据各个层的 dim, 构建网络结构
        # dense_layer_0 --> nn.Sequential(nn.Linear, ReLU, batchNorm, dropout)
        # dense_layer_1 --> nn.Sequential(nn.Linear, ReLU, batchNorm, dropout)
        # dense_layer_2 --> nn.Sequential(nn.Linear, ReLU, batchNorm, dropout)
        self.dense_sequential = nn.Sequential()
        for i in range(1, len(hidden_layers)):
            self.dense_sequential.add_module("dense_layer_{}".format(i - 1), dense_layer(hidden_layers[i - 1], hidden_layers[i], dropout[i - 1], batchnorm))

        # the output_dim attribute will be used as input_dim when "merging" the models
        self.output_dim = hidden_layers[-1]
        print("deep dense 维度：{}".format(self.output_dim))

    def forward(self, X: Tensor) -> Tensor:
        return self.dense_sequential(X)
