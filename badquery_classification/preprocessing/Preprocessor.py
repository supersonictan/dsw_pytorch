# coding: UTF-8
#! /usr/bin/python
# -*- coding: utf-8 -*-
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd
import warnings
import pickle as pkl
import time

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler







class BasePreprocessor(object):
    def fit(self, df: pd.DataFrame):
        raise NotImplementedError("Preprocessor must implement this method")

    def transform(self, df: pd.DataFrame):
        raise NotImplementedError("Preprocessor must implement this method")

    def fit_transform(self, df: pd.DataFrame):
        raise NotImplementedError("Preprocessor must implement this method")




# input: cols = ['education', 'relationship', 'workclass']
# output: {'education': {'Bachelors': 0, 'HS-grad': 1, '11th': 2, 'Masters': 3, '9th': 4}, 'workclass': {'State-gov': 0, 'Self-emp-not-inc': 1, 'Private': 2}}
def label_encoder(df_inp:pd.DataFrame, cols:Optional[List[str]]=None, val_to_idx: Optional[Dict[str, Dict[str, int]]] = None):
    df = df_inp.copy()

    if cols is None:
        # 根据数据类型选择特征
        cols = list(df.select_dtypes(include=["object"]).columns)

    if not val_to_idx:
        val_types = dict()
        for c in cols:
            val_types[c] = df[c].unique()
        val_to_idx = dict()
        for k, v in val_types.items():
            # {'education': {'Bachelors': 0, 'HS-grad': 1, '11th': 2, 'Masters': 3, '9th': 4}, 'workclass': {'State-gov': 0, 'Self-emp-not-inc': 1, 'Private': 2}}
            val_to_idx[k] = {o: i for i, o in enumerate(val_types[k])}

    for k, v in val_to_idx.items():
        df[k] = df[k].apply(lambda x: v[x])

    return df, val_to_idx



def gen_vocab_dic(df_inp:pd.DataFrame, text_col):

    df_new = df_inp.copy()[text_col].str.split(" ", expand=True).stack()
    val_types = dict()

    for c in text_col:
        val_types[c] = df_new[c].unique()

    # tmp_list = df_inp.copy()[text_col].values.tolist()
    # print(type(tmp_list))
    # word_uniq = []
    # for i in range(0, len(tmp_list)):
    #     word_uniq = word_uniq + tmp_list[i].split(" ")
    #
    # vocab_dic = {k: v for v, k in enumerate(word_uniq)}

    print(val_types)





UNK, PAD = '<UNK>', '<PAD>'  # 未知字，padding符号
class MultiDeepTextPreprocessor(BasePreprocessor):
    def __init__(self, text_cols_list:str = None, pad_size:int=6, term_dic_path=None):
        super(MultiDeepTextPreprocessor, self).__init__()
        self.text_cols_list = text_cols_list
        self.pad_size = pad_size
        self.term_dic_path = term_dic_path

    def fit(self, df: pd.DataFrame) -> BasePreprocessor:
        df = pd.read_csv(self.term_dic_path, names=['term', 'id'])
        self.vocab = dict(zip(list(df.term), list(df.id)))
        print("Finished Load term dic size:{}".format(len(self.vocab)))

        return self


    def transform(self, df: pd.DataFrame) -> np.ndarray:
        # tokenizer = lambda x: [y for y in x]  # char-level
        tokenizer = lambda x: [y for y in x.split(' ')]
        def trans_text2id(content):
            content = content[0]
            token_list = tokenizer(content)
            # print(token_list)
            seq_len = len(token_list)

            if len(token_list) < self.pad_size:
                token_list.extend([PAD] * (self.pad_size - len(token_list)))
            else:
                token_list = token_list[:self.pad_size]
                seq_len = self.pad_size

            # word to id
            word2id_list = [self.vocab.get(w, self.vocab.get(UNK)) for w in token_list]

            return word2id_list

        df_list = df.copy()[self.text_cols_list].values.tolist()
        df_list_new = [trans_text2id(x) for x in df_list]

        return np.array(df_list_new)

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.fit(df).transform(df)











UNK, PAD = '<UNK>', '<PAD>'  # 未知字，padding符号
class DeepTextPreprocessor(BasePreprocessor):
    def __init__(self, text_cols_list:str = None, pad_size:int=16, vocab_path=None, char_level=True):
        super(DeepTextPreprocessor, self).__init__()
        self.text_cols_list = text_cols_list
        self.pad_size = pad_size
        self.vocab_path = vocab_path
        self.char_level = char_level

    def fit(self, df: pd.DataFrame) -> BasePreprocessor:
        df_text = df.copy()[self.text_cols_list]
        # 加载词表
        self.vocab = pkl.load(open(self.vocab_path, 'rb'))
        print("Finish Load Vocab ！")
        return self


    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if not self.char_level:
            tokenizer = lambda x: [y for y in x.split(' ')]
        else:
            tokenizer = lambda x: [y for y in x]  # char-level
        def trans_text2id(content):
            content = content[0]
            token_list = tokenizer(content)
            # print(token_list)
            seq_len = len(token_list)

            if len(token_list) < self.pad_size:
                token_list.extend([PAD] * (self.pad_size - len(token_list)))
            else:
                token_list = token_list[:self.pad_size]
                seq_len = self.pad_size

            # word to id
            word2id_list = [self.vocab.get(w, self.vocab.get(UNK)) for w in token_list]

            return word2id_list

        st = time.clock()
        df_list = df.copy()[self.text_cols_list].values.tolist()
        ed = time.clock()
        print("TextCol tolist() cost:{} seconds", ed - st)

        st = time.clock()
        df_list_new = [trans_text2id(x) for x in df_list]
        ed = time.clock()
        print("TextCol trans_text2id() cost:{} seconds", ed - st)

        # print(df_list_new)
        # df_text = df.copy()[self.text_cols_list]
        # # print(df_text.values)
        # df_text[self.text_cols_list] = df_text[self.text_cols_list].apply(trans_text2id)
        # print(np.array(df_text.values))
        # # print(df_text.values)

        return np.array(df_list_new)

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.fit(df).transform(df)






"""
    1. 对 col 中不连续的特征 ==> 转化成对应编号
        eg: 需要embed_col=[("education", 10), ("relationship", 8)] ==> 返回将 colname=education 中的value 转化成 0/1/2/3/4, 用于获取 embedding
    2. 对连续数值特征标准化
"""
class DeepPreprocessor(BasePreprocessor):
    # embed_cols_list = [("education", 10), ("relationship", 8), ("workclass", 10), ("occupation", 10), ("native_country", 10)]
    # continuous_cols = ["age", "hours_per_week"]
    def __init__(self, embed_cols_list:List[Tuple[str, int]]=None, continuous_cols:List[str] = None, default_embed_dim:int=8, already_standard:Optional[List[str]] = None):
        super(DeepPreprocessor, self).__init__()
        self.embed_cols_list = embed_cols_list
        self.continuous_cols_list = continuous_cols
        self.already_standard = already_standard
        self.default_embed_dim = default_embed_dim

        # embed_dim_dic = {"education":10, "relationship":8}
        self.embed_dim_dic = dict(self.embed_cols_list)

        # 需要归一化的 col
        if self.already_standard is not None:
            self.standardize_cols = [c for c in self.continuous_cols_list if c not in self.already_standard]
        else:
            self.standardize_cols = self.continuous_cols_list

        assert (self.embed_cols_list is not None) or (self.continuous_cols_list is not None), "'embed_cols_list' and 'continuous_cols_list' are 'None'. Please, define at least one of the two."

    def fit(self, df: pd.DataFrame) -> BasePreprocessor:
        if self.embed_cols_list is not None:
            # 返回指定 embedding_name 的 DataFrame, embed_cols_list = [("education", 10), ("relationship", 8), ("workclass", 10), ("occupation", 10), ("native_country", 10)]
            embed_colname = [emb[0] for emb in self.embed_cols_list]
            df_emb = df.copy()[embed_colname]

            # {'education': {'Bachelors': 0, 'HS-grad': 1, '11th': 2, 'Masters': 3, '9th': 4}, 'workclass': {'State-gov': 0, 'Self-emp-not-inc': 1, 'Private': 2}}
            _, self.encoding_dict = label_encoder(df_emb, cols=df_emb.columns.tolist())

            self.emb_col_val_dim_tuple: List = []

            # K: education
            # V: {'Bachelors': 0, 'HS-grad': 1, '11th': 2, 'Masters': 3, '9th': 4}
            for k, v in self.encoding_dict.items():
                self.emb_col_val_dim_tuple.append(
                    (k, len(v), self.embed_dim_dic[k]))  # [('education', 5, 2), ('workclass', 3, 16)]

        if self.continuous_cols_list is not None:
            df_cont = df.copy()[self.continuous_cols_list]
            df_std = df_cont[self.standardize_cols]
            self.sklearn_scaler = StandardScaler().fit(df_std.values)

        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        #
        if self.embed_cols_list is not None:
            embed_colname = [emb[0] for emb in self.embed_cols_list]
            df_emb = df.copy()[embed_colname]
            df_emb, _ = label_encoder(df_emb, cols=df_emb.columns.tolist(), val_to_idx=self.encoding_dict)

        # 标准化处理
        if self.continuous_cols_list is not None:
            df_cont = df.copy()[self.continuous_cols_list]
            self.sklearn_scaler.mean_
            df_std = df_cont[self.standardize_cols]
            df_cont[self.standardize_cols] = self.sklearn_scaler.transform(df_std.values)

        try:
            df_deep = pd.concat([df_emb, df_cont], axis=1)
        except:
            try:
                df_deep = df_emb.copy()
            except:
                df_deep = df_cont.copy()

        # {'education':0, 'relationship':1}
        self.deep_column_idx = {k: v for v, k in enumerate(df_deep.columns)}
        return df_deep.values

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.fit(df).transform(df)





"""
    input: DataFrame类型数据, wide_cols_name, cross_col_name
    ouput: np.ndarray 类型的 one-hot数据
    
    1. 生成交叉特征列B: concat(col_a, '_', col_b)
    2. 对 [wide_col + B] 列执行 sklearn.OneHotEncoder
        enc.fit([[0, 0, 3], [1, 1, 0], [0, 2, 1], [1, 0, 2]]) ==> transform([[0, 1, 3]]) ==> [[ 1.  0.  0.  1.  0.  0.  0.  0.  1.]]
"""
class WidePreprocessor(BasePreprocessor):
    # wide_cols = ['age_buckets', 'education', 'relationship','workclass','occupation',... 'native_country','gender']
    # crossed_cols = [('education', 'occupation'), ('native_country', 'occupation')]
    def __init__(self, wide_cols: List[str], crossed_cols=None, already_dummies: Optional[List[str]] = None, sparse=False):
        super(WidePreprocessor, self).__init__()
        self.wide_name_list = wide_cols
        self.crossed_name_list = crossed_cols
        self.already_dummies = already_dummies
        self.sklearn_one_hot_enc = OneHotEncoder(sparse=sparse)

    def fit(self, df: pd.DataFrame) -> BasePreprocessor:
        df_wide = df.copy()[self.wide_name_list]

        if self.crossed_name_list is not None:
            # 在 df_wide 结构中增加新的交叉列明 + 对应数据(对应数据concat(a, '-', b))
            df_wide, crossed_colnames_list = self._cross_cols(df_wide)
            self.wide_crossed_cols_list = self.wide_name_list + crossed_colnames_list
        else:
            self.wide_crossed_cols_list = self.wide_name_list

        if self.already_dummies:
            dummy_cols = [c for c in self.wide_crossed_cols_list if c not in self.already_dummies]
            self.sklearn_one_hot_enc.fit(df_wide[dummy_cols])
        else:
            self.sklearn_one_hot_enc.fit(df_wide[self.wide_crossed_cols_list])
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        self.sklearn_one_hot_enc.categories_

        df_wide = df.copy()[self.wide_name_list]
        if self.crossed_name_list is not None:
            df_wide, _ = self._cross_cols(df_wide)
        if self.already_dummies:
            X_oh_1 = df_wide[self.already_dummies].values
            dummy_cols = [c for c in self.wide_crossed_cols_list if c not in self.already_dummies]
            X_oh_2 = self.sklearn_one_hot_enc.transform(df_wide[dummy_cols])
            return np.hstack((X_oh_1, X_oh_2))
        else:
            return self.sklearn_one_hot_enc.transform(df_wide[self.wide_crossed_cols_list])

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.fit(df).transform(df)

    def _cross_cols(self, df: pd.DataFrame):
        crossed_colnames_list = []
        # crossed_name_list = [('education', 'occupation'), ('native_country', 'occupation')]
        for two_cross_cols_list in self.crossed_name_list:
            two_cross_cols_list = list(two_cross_cols_list)  # ['education', 'occupation']
            for c in two_cross_cols_list:
                df[c] = df[c].astype("str")

            new_colname = "_".join(two_cross_cols_list)
            df[new_colname] = df[two_cross_cols_list].apply(lambda x: "-".join(x), axis=1)
            crossed_colnames_list.append(new_colname)
        return df, crossed_colnames_list

