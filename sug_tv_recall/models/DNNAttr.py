import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
import math
import time

embedding_dim = 32
need_pretrain_embedding = False
pad_size = 16
dropout = 0.5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_head = 6
hidden_dim = 512
pad_size = 20
num_classes = 64
num_encoder = 6


class DNNAttr(nn.Module):
    def __init__(self):
        super(DNNAttr, self).__init__()
        self.output_dim = num_classes

        self.embedding = nn.Embedding(180000, embedding_dim, padding_idx=0)

        self.tanh1 = nn.Tanh()

        self.fc = nn.Linear(embedding_dim, self.output_dim)


    def forward(self, X, X_query_embed):

        """
        text [seten_len,batch_size]>>>[seten_len,batch_Size,emb_dim]
        """
        # [512, len, 32]
        embedded = self.embedding(X)

        # 将 bacth_size 的维度放在前面 [batch_size,seten_len,embed_size] 也可以用transpose
        # embedded = embedded.permute(1, 0, 2)

        # 沿着seq_len维度 句子进行平均此话  得到句子的平局词向量 [batch_size,1,embed_size]>>>[batch_Size,embed_Size]
        # https://blog.csdn.net/qq_29678299/article/details/103102397
        # size=[512, 32]
        pooled = F.avg_pool2d(embedded, (embedded.shape[1], 1)).squeeze()
        # print("pooled.shape:" + str(pooled.shape))

        M = self.tanh1(pooled)
        # print("M.shape:" + str(M.shape))

        # X_query_embed:[512, 1, 32] --> [512, 32]
        X_query_embed = X_query_embed.squeeze()
        # print("X_query_embed.shape:" + str(X_query_embed.shape))

        # [512, 32]
        alpha = F.softmax(torch.mul(M, X_query_embed), dim=1)
        out = pooled * alpha

        # 将平局之后的句子向量放到线性层 进行FC[batch_Size,embed_Size]>>>[batch_Size,output_Size]
        return self.fc(out)
