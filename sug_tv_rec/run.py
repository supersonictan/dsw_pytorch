import sys

from models.DeepDense import DeepDense
from models.TextLSTM import TextLSTM
from models.TransformerEncoder import TransformerEncoder
from models.Wide import Wide

from models.WideDeep import WideDeep
from optim.Initializer import KaimingNormal, XavierNormal
from optim.radam import RAdam
from preprocessing.Preprocessor import WidePreprocessor, DeepPreprocessor, DeepTextPreprocessor, MultiDeepTextPreprocessor

import numpy as np
import pandas as pd
import torch
from odps import ODPS
from odps.df import DataFrame
import common_io,time

UNK, PAD = '<UNK>', '<PAD>'  # 未知字，padding符号
prefix_pad_size = 6
prefix_dic_path = '/home/admin/workspace/project/odps/bin/prefix_dic'
sug_dic_path = '/home/admin/workspace/project/odps/bin/sug_query_dic'
summary_path = '/home/admin/workspace/project/dsw_pytorch/sug_tv_rec/log'



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
odps = ODPS('LTAI8OjacOR3fwkT', 'BQWM6tSBjBnjNEXJx3S0q3gLUHJEHK', 'ytsoku', endpoint='http://service-corp.odps.aliyun-inc.com/api')

def get_sequence_idx(df: DataFrame, col_name: str, padding_size=15, vocab=None):
    tokenizer = lambda x: [y for y in x.split(' ')]
    def trans_text2id(content):  # 'a b c'
        # content = content[0]
        token_list = tokenizer(content)
        seq_len = len(token_list)

        if seq_len < padding_size:
            token_list.extend([PAD] * (padding_size - seq_len))
        else:
            token_list = token_list[: padding_size]
            seq_len = padding_size

        # word to id
        word2id_list = [vocab.get(w, vocab.get(UNK)) for w in token_list]
        # print('\n')
        # print(token_list)
        # print(word2id_list)

        return word2id_list

    df_list = df[col_name].copy().astype(np.str).values.tolist()  # ['a b c', 'd c e', 'e f g']
    word2id_list = [trans_text2id(x) for x in df_list]
    ret = np.array(word2id_list)
    return ret



def item2id(df: DataFrame, col_name: str, padding_size=15, vocab:dict=None):
    df = df[col_name].copy().astype(np.str)
    # print(df.head(100))
    df_list = df.map(lambda x: vocab.get(x, vocab.get(UNK)))
    # print(df_list)
    return df_list.values[:, np.newaxis]


if __name__ == '__main__':
    t_load_start = time.clock()
    names = ['uid', 'prefix', 'sug', 'label', 'ott_uv_norm', 'onehot_uv', 'show_expo', 'ugc3_expo', 'is_offical_name', 'is_bigword', 'category', 'meizi_series', 'is_recent_online', 'category_prefer', 'onehot_category_prefer', 'onehot_category', 'onehot_gender', 'family_pred_gender', 'family_pred_age_level', 'seq']
    df = pd.read_csv("/home/admin/workspace/project/odps/bin/sug_tv_traindata.csv", sep='\t', names=names)
    t_load_end = time.clock()
    print('Load train data Cost: %s Seconds' % (t_load_end - t_load_start))

    """
    Wide 特征: 
        onehot  onehot_gender, onehot_category_prefer, onehot_category, onehot_uv, ott_uv_norm, is_offical_name, is_bigword, meizi_series, is_recent_online
        croass  ("category", "family_pred_gender")
    Deep 特征:
        category
    """
    # ----------------------- wide 特征 -----------------------
    wide_cols = ["onehot_gender", "onehot_category_prefer", "onehot_category", "onehot_uv", "is_offical_name", "is_bigword", "meizi_series", "is_recent_online"]
    crossed_cols = [("category", "family_pred_gender")]

    t_load_start = time.clock()
    X_wide = np.hstack((df['onehot_gender'].str.split(' ', expand=True).astype(np.float).values, df['onehot_category_prefer'].str.split(' ', expand=True).astype(np.float).values))
    X_wide = np.hstack((X_wide, df['onehot_category'].str.split(' ', expand=True).astype(np.float).values))
    X_wide = np.hstack((X_wide, df['onehot_uv'].str.split(' ', expand=True).astype(np.float).values))
    X_wide = np.hstack((X_wide, df['is_offical_name'].astype(np.float).values[:, np.newaxis]))
    X_wide = np.hstack((X_wide, df['is_bigword'].astype(np.float).values[:, np.newaxis]))
    X_wide = np.hstack((X_wide, df['meizi_series'].astype(np.float).values[:, np.newaxis]))
    X_wide = np.hstack((X_wide, df['is_recent_online'].astype(np.float).values[:, np.newaxis]))

    t_load_end = time.clock()
    print('Wide fit_transform Cost: %s Seconds' % (t_load_end - t_load_start))
    # print(X_wide)

    # ----------------------- deep 列 -----------------------
    cat_embed_cols = [("family_pred_gender", 8), ("family_pred_age_level", 12), ("category", 8)]
    continuous_cols = ["ott_uv_norm", "category_prefer"]
    t_load_start = time.clock()
    prepare_deep = DeepPreprocessor(embed_cols_list=cat_embed_cols, continuous_cols=continuous_cols)
    X_deep = prepare_deep.fit_transform(df)
    t_load_end = time.clock()
    print('\nDeep fit_transform Cost: %s Seconds' % (t_load_end - t_load_start))
    # print(X_deep)

    # ----------------------- prefix 列 ---------------------
    t_load_start = time.clock()
    df_prefix = pd.read_csv(prefix_dic_path, sep='\t', names=['prefix', 'id'])
    vocab_prefix = dict(zip(list(df_prefix.prefix), list(df_prefix.id.astype(np.int64))))
    print("\nFinished Load prefix dic size:{}".format(len(vocab_prefix)))

    X_prefix = item2id(df, col_name='prefix', padding_size=1, vocab=vocab_prefix)
    t_load_end = time.clock()
    print('Prefix Data Cost: %s Seconds' % (t_load_end - t_load_start))
    print("X_prefix shape:" + str(X_prefix.shape))


    # ---------------------- query 列 ----------------------
    t_load_start = time.clock()
    df_sug = pd.read_csv(sug_dic_path, sep='\t', names=['sug_query', 'id']).astype(np.str)
    vocab_sug = dict(zip(df_sug.sug_query, df_sug.id.astype(np.int64)))
    print("\nFinished Load sug_query dic size:{}".format(len(vocab_sug)))

    X_sug = item2id(df, col_name='sug', padding_size=1, vocab=vocab_sug)
    t_load_end = time.clock()
    print('Sug Data Cost: %s Seconds' % (t_load_end - t_load_start))
    print("X_sug shape:" + str(X_sug.shape))

    # ----------------------- user sequence ---------------------
    t_load_start = time.clock()
    X_seq = get_sequence_idx(df, col_name='seq', padding_size=15, vocab=vocab_sug)
    t_load_end = time.clock()
    print('\nSug Data Cost: %s Seconds' % (t_load_end - t_load_start))
    # print(X_seq)

    target = "label"
    target = df[target].values


    # Build model
    wide = Wide(wide_dim=X_wide.shape[1], output_dim=1)
    deepdense = DeepDense(hidden_layers=[64, 32], dropout=[0.2, 0.2], deep_column_idx=prepare_deep.deep_column_idx, embed_input=prepare_deep.emb_col_val_dim_tuple, continuous_cols=continuous_cols)
    transformer = TransformerEncoder()

    wide_deep_model = WideDeep(wide=wide, deepdense=deepdense, deeptext=transformer)

    # 1.设定 optimizer ==> 2.Init 子 model 各层种参数 ==> 3.StepLR
    wide_opt = torch.optim.Adam(wide_deep_model.wide.parameters())
    deep_opt = RAdam(wide_deep_model.deepdense.parameters())
    text_opt = torch.optim.Adam(wide_deep_model.deeptext.parameters())

    wide_sch = torch.optim.lr_scheduler.StepLR(wide_opt, step_size=3)
    deep_sch = torch.optim.lr_scheduler.StepLR(deep_opt, step_size=5)
    text_sch = torch.optim.lr_scheduler.StepLR(text_opt, step_size=5)

    optimizers = {"wide": wide_opt, "deepdense": deep_opt, 'deeptext': text_opt}
    schedulers = {"wide": wide_sch, "deepdense": deep_sch, 'deeptext': text_sch}
    initializers = {"wide": KaimingNormal, "deepdense": XavierNormal, 'deeptext': KaimingNormal}
    

    wide_deep_model.compile(method='binary', optimizers_dic=optimizers, lr_schedulers_dic=schedulers, initializers_dic=initializers)

    wide_deep_model.fit(X_wide=X_wide, X_deep=X_deep, X_text=X_seq, target=target, n_epochs=4, batch_size=512, val_split=0.2, summary_path=summary_path)


