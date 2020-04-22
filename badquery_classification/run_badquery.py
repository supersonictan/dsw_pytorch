import sys

from models.DeepDense import DeepDense
from models.TextLSTM import TextLSTM
from models.TransformerEncoder import TransformerEncoder
from models.Wide import Wide

from models.WideDeep import WideDeep
from optim.Initializer import KaimingNormal, XavierNormal
from optim.radam import RAdam
from preprocessing.Preprocessor import WidePreprocessor, DeepPreprocessor, DeepTextPreprocessor, \
    MultiDeepTextPreprocessor

import numpy as np
import pandas as pd
import torch
from odps import ODPS
from odps.df import DataFrame

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

embedding_path = '/home/admin/workspace/project/embedding_SougouNews.npz'
summary_path = './log/badquery'
vocab_path = '/home/admin/workspace/project/vocab.pkl'
term_dic_path = '/home/admin/workspace/project/term_dic.csv'

with_text = True
two_text = True
odps = ODPS('LTAI8OjacOR3fwkT', 'BQWM6tSBjBnjNEXJx3S0q3gLUHJEHK', 'ytsoku', endpoint='http://service-corp.odps.aliyun-inc.com/api')

if __name__ == '__main__':
    bExist = odps.exist_table('ads_soku_badquery_feature_lite')
    if not bExist:
        print("Table:{} exists:{}".format('ads_soku_badquery_feature_lite', bExist))

    odps_table = odps.get_table('ads_soku_badquery_feature_lite')
    odps_df = DataFrame(odps_table.get_partition("ds='20200422'")) 

    print('before to_pandas')
    df = odps_df.to_pandas()
    print('after to_pandas')

    df["term_num_bucket"] = pd.cut(df.term_num, bins=[0, 1, 3, 4, 6, 15], labels=np.arange(5))
    df["bounce_rate_bucket"] = pd.cut(df.bounce_rate, bins=[-1, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.1], labels=np.arange(9))
    print(df.head())

    # wide 列名
    wide_cols = ["term_num_bucket", "bounce_rate_bucket", "is_ne"]
    crossed_cols = [("term_num_bucket", "is_ne"), ("bounce_rate_bucket", "term_num_bucket")]
    # deep 列名
    cat_embed_cols = [("main_category", 8)]
    continuous_cols = ["uv", "bounce_rate", "term_num"]

    # text 列明
    text_cols = 'query'

    # text 列名
    text_cols2 = 'text_feature'
    target = "label"
    target = df[target].values

    """
    Wide 输入
        [[0. 0. 0. 0. 1. 0. 0. 1. 0. 0. 0. 0. 1. 0. 1. 0. 0. 0. 0. 0. 1. 0. 0. 1. 0. 1. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]
         [0. 0. 0. 0. 0. 1. 0. 0. 0. 1. 0. 1. 0. 0. 0. 1. 0. 0. 1. 0. 0. 0. 0. 0. 1. 0. 1. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 1. 0. 0.]]
    """
    prepare_wide = WidePreprocessor(wide_cols=wide_cols, crossed_cols=crossed_cols)
    X_wide = prepare_wide.fit_transform(df)

    """
    Deep 输入
    [[ 2. 0. 2. 2. 1. -0.346393 0.31569232]
     [ 2. 1. 1. 1. 1. 0.9675134 -2.0520001]]
    """
    prepare_deep = DeepPreprocessor(embed_cols_list=cat_embed_cols, continuous_cols=continuous_cols)
    X_deep = prepare_deep.fit_transform(df)

    X_text = None
    X_text2 = None
    # text1 输入
    if with_text:
        prepare_text = DeepTextPreprocessor(text_cols_list=text_cols, pad_size=16, vocab_path=vocab_path)
        X_text = prepare_text.fit_transform(df)
    """
    [[  66  440 4761 4761 4761 4761 4761 4761 4761 4761 4761 4761 4761 4761 4761 4761]
     [   5  440 4761 4761 4761 4761 4761 4761 4761 4761 4761 4761 4761 4761 4761 4761]]
    """

    # text2 输入（term粒度）
    if with_text and two_text:
        prepare_text2 = MultiDeepTextPreprocessor(text_cols_list=text_cols2, pad_size=20, term_dic_path=term_dic_path)
        X_text2 = prepare_text2.fit_transform(df)


    # Build model
    wide = Wide(wide_dim=X_wide.shape[1], output_dim=1)
    deepdense = DeepDense(hidden_layers=[64, 32], dropout=[0.2, 0.2], deep_column_idx=prepare_deep.deep_column_idx, embed_input=prepare_deep.emb_col_val_dim_tuple, continuous_cols=continuous_cols)

    if with_text:
        lstm = TextLSTM(embedding_path)
    if two_text:
        # transformer = TextLSTM(embedding_path, need_pretrain_embedding=False)
        transformer = TransformerEncoder(embedding_path)

    if with_text and two_text:
        wide_deep_model = WideDeep(wide=wide, deepdense=deepdense, deeptext=lstm, deeptext2=transformer)
    elif with_text:
        wide_deep_model = WideDeep(wide=wide, deepdense=deepdense, deeptext=lstm)
    else:
        wide_deep_model = WideDeep(wide=wide, deepdense=deepdense)

    # 1.设定 optimizer ==> 2.Init 子 model 各层种参数 ==> 3.StepLR
    wide_opt = torch.optim.Adam(wide_deep_model.wide.parameters())
    deep_opt = RAdam(wide_deep_model.deepdense.parameters())
    if with_text:
        text_opt = torch.optim.Adam(wide_deep_model.deeptext.parameters())
    if two_text:
        text_opt2 = torch.optim.Adam(wide_deep_model.deeptext2.parameters())

    wide_sch = torch.optim.lr_scheduler.StepLR(wide_opt, step_size=3)
    deep_sch = torch.optim.lr_scheduler.StepLR(deep_opt, step_size=5)
    if with_text:
        text_sch = torch.optim.lr_scheduler.StepLR(text_opt, step_size=5)
    if two_text:
        text_sch2 = torch.optim.lr_scheduler.StepLR(text_opt2, step_size=5)

    if with_text and two_text:
        optimizers = {"wide": wide_opt, "deepdense": deep_opt, 'deeptext': text_opt, 'deeptext2': text_opt2}
        schedulers = {"wide": wide_sch, "deepdense": deep_sch, 'deeptext': text_sch, 'deeptext2': text_sch2}
        initializers = {"wide": KaimingNormal, "deepdense": XavierNormal, 'deeptext': KaimingNormal, 'deeptext2': KaimingNormal}
    elif with_text:
        optimizers = {"wide": wide_opt, "deepdense": deep_opt, 'deeptext': text_opt}
        schedulers = {"wide": wide_sch, "deepdense": deep_sch, 'deeptext': text_sch}
        initializers = {"wide": KaimingNormal, "deepdense": XavierNormal, 'deeptext': KaimingNormal}
    else:
        optimizers = {"wide": wide_opt, "deepdense": deep_opt}
        schedulers = {"wide": wide_sch, "deepdense": deep_sch}
        initializers = {"wide": KaimingNormal, "deepdense": XavierNormal}

    wide_deep_model.compile(method='binary', optimizers_dic=optimizers, lr_schedulers_dic=schedulers, initializers_dic=initializers)

    wide_deep_model.fit(X_wide=X_wide, X_deep=X_deep, X_text=X_text, X_text2=X_text2, target=target, n_epochs=10, batch_size=256,val_split=0.2, summary_path=summary_path)




