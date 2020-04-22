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




class DeepDense(nn.Module):
    def __init__(self, deep_column_idx, hidden_layers, batchnorm=False, dropout=None, embed_input=None, embed_p=0.0, continuous_cols=None):
        """
        :param deep_column_idx:         Dict[str, int]
        :param hidden_layers:           List[int]
        :param batchnorm:               bool
        :param dropout:                 List[float]
        :param embed_input:             List[Tuple[str, int, int]]
        :param embed_p:                 float
        :param continuous_cols:         List[str]
        """

        super(DeepDense, self).__init__()
        # e.g. [('education', 5, 2), ('workclass', 3, 16)]: List of Tuples with the column name, number of unique values and embedding dimension.
        self.embed_input = embed_input
        # ["age", "hours_per_week"]
        self.continuous_cols = continuous_cols
        # {name1:1, name2"2, name3:3}
        self.deep_column_idx = deep_column_idx

        # Step1. 生成各个层的 dim
        # Embeddings
        if self.embed_input is not None:
            # 这是一个矩阵类, 输入下标0，输出就是embeds矩阵中第0行
            self.embed_layers_dic = nn.ModuleDict({"emb_layer_" + col: nn.Embedding(val, dim) for col, val, dim in self.embed_input})
            self.embed_dropout = nn.Dropout(embed_p)
            emb_inp_dim = np.sum([embed[2] for embed in self.embed_input])  # 所有 embed 的长度
        else:
            emb_inp_dim = 0

        # Continuous
        if self.continuous_cols is not None:
            cont_inp_dim = len(self.continuous_cols)
        else:
            cont_inp_dim = 0

        # Dense Layers
        input_dim = emb_inp_dim + cont_inp_dim

        # 每一层 维度 [50, 64, 32]
        hidden_layers = [input_dim] + hidden_layers
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
        # [(education, 11, 32), ...]
        if self.embed_input is not None:
            # X所有数据 对应的 embedding
            #    embed特征_1  embed特征_2   embed特征_3
            # x=[0,0,0,0,    0,0,0,0,      0,0,0,0]
            x = [
                self.embed_layers_dic["emb_layer_" + col](X[:, self.deep_column_idx[col]].long())
                for col, _, _ in self.embed_input
            ]
            x = torch.cat(x, 1)
            x = self.embed_dropout(x)
        if self.continuous_cols is not None:
            cont_idx = [self.deep_column_idx[col] for col in self.continuous_cols]
            x_cont = X[:, cont_idx].float()
            x = torch.cat([x, x_cont], 1) if self.embed_input is not None else x_cont
        return self.dense_sequential(x)
