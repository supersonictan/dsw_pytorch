#! /usr/bin/python
# -*- coding: utf-8 -*-
import codecs
import os
import time
import torch
import torch.nn as nn
import numpy as np
from sklearn import metrics
import pickle as pkl


WORD_EMBEDDING_DIM = 300
HIDDEN_DIM = 64
HIDDEN_LAYERS = 2
LABEL_NUM = 1
DROPOUT = 0.5


# 创建模型
class TextLSTM(nn.Module):
    def __init__(self, pre_train_embed_path, need_pretrain_embedding=True):
        super(TextLSTM, self).__init__()
        self.output_dim = HIDDEN_DIM * HIDDEN_LAYERS

        # 1. embedding
        PRE_TRAIN_WORD_EMBEDDING = torch.tensor(np.load(pre_train_embed_path)["embeddings"].astype('float32'))

        if need_pretrain_embedding:
            self.lstm_word_embedding = nn.Embedding.from_pretrained(PRE_TRAIN_WORD_EMBEDDING, freeze=False)
        else:
            self.lstm_word_embedding = nn.Embedding(1105553, WORD_EMBEDDING_DIM)

        # 2. LSTM 以 word_embeddings 作为输入, 输出维度为 hidden_dim 的隐藏状态值
        """
        以下关于shape的注释只针对单向
        output: [batch_size, time_step, hidden_size]
        h_n: [num_layers, batch_size, hidden_size] # 虽然LSTM的batch_first为True,但是h_n/c_n的第一维还是num_layers
        c_n: 同h_n
        """
        self.lstm = nn.LSTM(WORD_EMBEDDING_DIM, HIDDEN_DIM, HIDDEN_LAYERS, bidirectional=True, batch_first=True, dropout=DROPOUT)


    def forward(self, x):
        # x format: torch.LongTensor类型的 (x, seq_len)
        x = x

        '''
            获取 batch word_embedding: 
            输入为两个维度(batch的大小，每个batch的单词个数)，
            输出则在两个维度上加上词向量的大小(N, W, embedding_dim)
        '''
        batch_embeds = self.lstm_word_embedding(x)  # [batch_size, seq_len, embeding]=[128, 32, 300]

        '''
        input(batch, seq_len, input_size)
        output(batch, seq_len, hidden_size * num_directions)
        '''
        out, _ = self.lstm(batch_embeds)

        out = out[:, -1, :]

        #out = self.lstm_fc(out[:, -1, :])  # 句子最后时刻的 hidden state
        return out

