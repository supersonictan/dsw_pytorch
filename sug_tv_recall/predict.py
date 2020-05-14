import sys

from models.DeepDense import DeepDense
from models.TextLSTM import TextLSTM
from models.TransformerEncoder import TransformerEncoder
from models.Wide import Wide

from models.WideDeep import WideDeep
from optim.Initializer import KaimingNormal, XavierNormal
from optim.radam import RAdam
from pandas import DataFrame
from preprocessing.Preprocessor import WidePreprocessor, DeepPreprocessor, DeepTextPreprocessor, MultiDeepTextPreprocessor

import numpy as np
import pandas as pd
import torch, os
from torch.utils.data import DataLoader
import time
import torch
from preprocessing.WideDeepDataset import WideDeepDataset




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



UNK, PAD = '<UNK>', '<PAD>'  # 未知字，padding符号


prefix_dic_path = '/home/admin/workspace/project/odps/bin/prefix_dic'
sug_dic_path = '/home/admin/workspace/project/odps/bin/sug_query_dic'
summary_path = '/home/admin/workspace/project/dsw_pytorch/sug_tv_rec/log'
train_data_path = "/home/admin/workspace/project/odps/bin/sug_tv_traindata.csv"
eval_data_path = "/home/admin/workspace/project/odps/bin/sug_tv_evaldata.csv"

def load_evaldata():
    t_load_start = time.time()
    names = 'uid','prefix','sug','label','ott_uv_norm','show_expo','ugc3_expo','is_offical_name','category','category_prefer','family_pred_gender','family_pred_age_level','seq','f0','f1','f2','f3','f4','f5','f6','f7','f8','f9','f10','f11','f12','f13','f14','f15','f16'
    df = pd.read_csv(eval_data_path, sep='\t', names=names)
    t_load_end = time.time()
    print('Load train data Cost: %s Seconds' % (t_load_end - t_load_start))

    """
    Wide 特征: 
        onehot  onehot_gender, onehot_category_prefer, onehot_category, onehot_uv, ott_uv_norm, is_offical_name, is_bigword, meizi_series, is_recent_online
        croass  ("category", "family_pred_gender")
    Deep 特征:
        category
    """
    # ----------------------- wide 特征 -----------------------
    wide_cols = ["onehot_gender", "onehot_category_prefer", "onehot_category", "onehot_uv", "is_offical_name",
                 "is_bigword", "meizi_series", "is_recent_online"]
    crossed_cols = [("category", "family_pred_gender")]

    t_load_start = time.time()
    X_wide = df['f0'].astype(np.float).values[:, np.newaxis]
    X_wide = np.hstack((X_wide, df['f1'].astype(np.float).values[:, np.newaxis]))
    X_wide = np.hstack((X_wide, df['f2'].astype(np.float).values[:, np.newaxis]))
    X_wide = np.hstack((X_wide, df['f3'].astype(np.float).values[:, np.newaxis]))
    X_wide = np.hstack((X_wide, df['f4'].astype(np.float).values[:, np.newaxis]))
    X_wide = np.hstack((X_wide, df['f5'].astype(np.float).values[:, np.newaxis]))
    X_wide = np.hstack((X_wide, df['f6'].astype(np.float).values[:, np.newaxis]))
    X_wide = np.hstack((X_wide, df['f7'].astype(np.float).values[:, np.newaxis]))
    X_wide = np.hstack((X_wide, df['f8'].astype(np.float).values[:, np.newaxis]))
    X_wide = np.hstack((X_wide, df['f9'].astype(np.float).values[:, np.newaxis]))
    X_wide = np.hstack((X_wide, df['f10'].astype(np.float).values[:, np.newaxis]))
    X_wide = np.hstack((X_wide, df['f11'].astype(np.float).values[:, np.newaxis]))
    X_wide = np.hstack((X_wide, df['f12'].astype(np.float).values[:, np.newaxis]))
    X_wide = np.hstack((X_wide, df['f13'].astype(np.float).values[:, np.newaxis]))
    X_wide = np.hstack((X_wide, df['f14'].astype(np.float).values[:, np.newaxis]))
    X_wide = np.hstack((X_wide, df['f15'].astype(np.float).values[:, np.newaxis]))
    X_wide = np.hstack((X_wide, df['f16'].astype(np.float).values[:, np.newaxis]))

    t_load_end = time.time()
    print('Wide fit_transform Cost: %s Seconds' % (t_load_end - t_load_start))
    # print(X_wide)

    # ----------------------- deep 列 -----------------------
    cat_embed_cols = [("family_pred_gender", 8), ("family_pred_age_level", 12), ("category", 8)]
    continuous_cols = ["ott_uv_norm", "category_prefer"]
    t_load_start = time.time()
    prepare_deep = DeepPreprocessor(embed_cols_list=cat_embed_cols, continuous_cols=continuous_cols)
    X_deep = prepare_deep.fit_transform(df)
    t_load_end = time.time()
    print('\nDeep fit_transform Cost: %s Seconds' % (t_load_end - t_load_start))
    # print(X_deep)

    # ----------------------- prefix 列 ---------------------
    t_load_start = time.time()
    df_prefix = pd.read_csv(prefix_dic_path, sep='\t', names=['prefix', 'id'])
    vocab_prefix = dict(zip(list(df_prefix.prefix), list(df_prefix.id.astype(np.int64))))
    print("\nFinished Load prefix dic size:{}".format(len(vocab_prefix)))

    X_prefix = item2id(df, col_name='prefix', padding_size=1, vocab=vocab_prefix)
    t_load_end = time.time()
    print('Prefix Data Cost: %s Seconds' % (t_load_end - t_load_start))
    print("X_prefix shape:" + str(X_prefix.shape))

    # ---------------------- query 列 ----------------------
    t_load_start = time.time()
    df_sug = pd.read_csv(sug_dic_path, sep='\t', names=['sug_query', 'id']).astype(np.str)
    vocab_sug = dict(zip(df_sug.sug_query, df_sug.id.astype(np.int64)))
    print("\nFinished Load sug_query dic size:{}".format(len(vocab_sug)))

    X_sug = item2id(df, col_name='sug', padding_size=1, vocab=vocab_sug)
    t_load_end = time.time()
    print('Sug Data Cost: %s Seconds' % (t_load_end - t_load_start))
    print("X_sug shape:" + str(X_sug.shape))

    # ----------------------- user sequence ---------------------
    t_load_start = time.time()
    X_seq = get_sequence_idx(df, col_name='seq', padding_size=15, vocab=vocab_sug)
    t_load_end = time.time()
    print('\nSug Data Cost: %s Seconds' % (t_load_end - t_load_start))
    # print(X_seq)

    target = "label"
    target = df[target].values

    return X_wide, X_deep, prepare_deep, X_seq, X_prefix, X_sug, target


def load_model_and_test(save_path):
    # init wide model
    wide = Wide(wide_dim=17, output_dim=1)

    # init deep_dense model
    deep_column_idx = dict()
    deep_column_idx['family_pred_gender'] = 0
    deep_column_idx['family_pred_age_level'] = 1
    deep_column_idx['category'] = 2
    deep_column_idx['ott_uv_norm'] = 3
    deep_column_idx['category_prefer'] = 4
    emb_col_val_dim_tuple = []
    emb_col_val_dim_tuple.append(('family_pred_gender', 4, 8))
    emb_col_val_dim_tuple.append(('family_pred_age_level', 1024, 12))
    emb_col_val_dim_tuple.append(('category', 28, 8))
    deepdense = DeepDense(hidden_layers=[64, 32], dropout=[0.2, 0.2], deep_column_idx=deep_column_idx, embed_input=emb_col_val_dim_tuple, continuous_cols=['ott_uv_norm', 'category_prefer'])

    # init transformer model
    transformer = TransformerEncoder()

    wide_deep_model = WideDeep(wide=wide, deepdense=deepdense, deeptext=transformer, head_layers=[256,64])

    # test data
    X_eval_wide, X_eval_deep, prepare_eval_deep, X_eval_seq, X_eval_prefix, X_eval_sug, eval_target = load_evaldata()
    print("HAHAHAH~~~~~~~~~~~~~~~~")
    print(X_eval_wide[0, :])
    print(X_eval_deep[0, :])
    print(X_eval_seq[0, :])
    print(X_eval_prefix[0, :])
    print(X_eval_sug[0, :])
    X_val = {"X_wide": X_eval_wide, "X_deep": X_eval_deep, "X_text": X_eval_seq, "X_prefix": X_eval_prefix, "X_sug":X_eval_sug, "target": eval_target}
    eval_set = WideDeepDataset(**X_val)
    eval_loader = DataLoader(dataset=eval_set, batch_size=512, num_workers=os.cpu_count(), shuffle=False)

    # load model
    st = time.time()
    wide_deep_model.load_state_dict(torch.load(save_path))
    wide_deep_model.method = 'binary'
    wide_deep_model.class_weight = None
    wide_deep_model.cuda()
    ed = time.time()
    print("Load model Cost:{0:>3} seconds".format((ed - st)))

    wide_deep_model.eval()
    loss_valid, auc_valid = wide_deep_model._test_validation_set(eval_loader)
    print("Load model Val Loss: {0:>5.3},  Val AUC: {1:>5.3}".format(loss_valid, auc_valid))


if __name__ == "__main__":
    load_model_and_test('/home/admin/workspace/project/dsw_pytorch/sug_tv_rec/log/05-03_13.35/sug_saved_model.pt')
