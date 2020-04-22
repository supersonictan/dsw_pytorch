import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
import math

embedding_dim = 300
need_pretrain_embedding = False
pad_size = 16
dropout = 0.5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_head = 6
hidden_dim = 512
pad_size = 20
num_classes = 64
num_encoder = 6



# https://blog.csdn.net/shenfuli/article/details/105349720
class TransformerEncoder(nn.Module):
    def __init__(self, pre_train_embed_path):
        super(TransformerEncoder, self).__init__()
        self.is_training = True
        hidden_size = 300

        self.output_dim = num_classes
        print("TransformerEncoder 维度：{}".format(self.output_dim))

        PRE_TRAIN_WORD_EMBEDDING = torch.tensor(np.load(pre_train_embed_path)["embeddings"].astype('float32'))
        if need_pretrain_embedding:
            self.embedding = nn.Embedding.from_pretrained(PRE_TRAIN_WORD_EMBEDDING, freeze=False)
        else:
            self.embedding = nn.Embedding(1105553, embedding_dim)
            # TODO: init embedding

        self.position = PositionalEncoding()

        encoder_layer = nn.TransformerEncoderLayer(d_model=300, nhead=5, dim_feedforward=2048)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=5)

        # self.rnn = nn.LSTM(embedding_dim, hidden_size, 2, bidirectional=True, batch_first=True, dropout=dropout)

        self.fc1 = nn.Linear(embedding_dim, num_classes)


    def forward(self, X):
        embed = self.embedding(X)
        # print('embed shape:{}'.format(embed.shape))

        embed = self.position(embed)
        # print('position shape:{}'.format(embed.shape))

        embed = self.transformer_encoder(embed)
        # print('transformer_encoder shape:{}'.format(embed.shape))

        out = torch.sum(embed, 1)  # [batch, embedding_size]
        out = self.fc1(out)

        # print('output shape:{}'.format(out.shape))
        return out

        # out = self.fc1(embed)
        # return out
        # out, _ = self.rnn(embed)
        # out = out[:, -1, :]

        # return out



class PositionalEncoding(nn.Module):
    def __init__(self, d_model = 300, dropout = 0.2, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + nn.Parameter(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)