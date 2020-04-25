import warnings
from sklearn import metrics

import numpy as np
import os

import time
from numpy import mean
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import (List, Union, Dict, Optional, Tuple)

from torch.utils.tensorboard import SummaryWriter
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader

from models.DeepDense import dense_layer
from util.WideDeepUtil import _train_val_split

use_cuda = torch.cuda.is_available()
n_cpus = os.cpu_count()
#n_cpus = 1



class WideDeep(nn.Module):
    def __init__(self, wide, deepdense, output_dim=1, deeptext=None, deeptext2=None, deepimage=None, deephead=None, head_layers=None, head_dropout=None, head_batchnorm=None):
        """
        :param wide:            nn.Module
        :param deepdense:       nn.Module
        :param output_dim:      int
        :param deeptext:        nn.Module
        :param deepimage:       nn.Module
        :param deephead:        nn.Module
        :param head_layers:     List[int]
        :param head_dropout:    List
        :param head_batchnorm:  bool
        """

        super(WideDeep, self).__init__()
        self.wide = wide
        self.deepdense = deepdense
        self.deeptext = deeptext
        self.deeptext2 = deeptext2
        self.deepimage = deepimage
        self.deephead = deephead

        if self.deephead is None:
            if head_layers is not None:
                input_dim: int = self.deepdense.output_dim + self.deeptext.output_dim + self.deepimage.output_dim
                head_layers = [input_dim] + head_layers
                if not head_dropout:
                    head_dropout = [0.0] * (len(head_layers) - 1)
                self.deephead = nn.Sequential()
                for i in range(1, len(head_layers)):
                    self.deephead.add_module(
                        "head_layer_{}".format(i - 1),
                        dense_layer(
                            head_layers[i - 1],
                            head_layers[i],
                            head_dropout[i - 1],
                            head_batchnorm,
                        ),
                    )
                self.deephead.add_module(
                    "head_out", nn.Linear(head_layers[-1], output_dim)
                )
            else:
                self.deepdense = nn.Sequential(self.deepdense, nn.Linear(self.deepdense.output_dim, output_dim))
                if self.deeptext is not None:
                    self.deeptext = nn.Sequential(self.deeptext, nn.Linear(self.deeptext.output_dim, output_dim))
                if self.deeptext2 is not None:
                    self.deeptext2 = nn.Sequential(self.deeptext2, nn.Linear(self.deeptext2.output_dim, output_dim))
                if self.deepimage is not None:
                    self.deepimage = nn.Sequential(self.deepimage, nn.Linear(self.deepimage.output_dim, output_dim))

        # print("\nNetwork Structure:")
        # print(self)
        # print("\n")


    def forward(self, X: Dict[str, Tensor]) -> Tensor:
        # Wide output: direct connection to the output neuron(s)
        out = self.wide(X["wide"])
        # print('wide shape:{}'.format(out.shape))

        # Deep output: either connected directly to the output neuron(s) or passed through a head first
        if self.deephead:
            deepside = self.deepdense(X["deepdense"])
            if self.deeptext is not None:
                deepside = torch.cat([deepside, self.deeptext(X["deeptext"])], axis=1)  # type: ignore
            if self.deepimage is not None:
                deepside = torch.cat([deepside, self.deepimage(X["deepimage"])], axis=1)  # type: ignore
            deepside_out = self.deephead(deepside)
            return out.add_(deepside_out)
        else:
            out.add_(self.deepdense(X["deepdense"]))
            if self.deeptext is not None:
                # print('deepdense shape:{}'.format(out.shape))
                out_text = self.deeptext(X["deeptext"])

                # print('text shape:{}'.format(out_text.shape))

                out.add_(out_text)

            if self.deeptext2 is not None:
                out_text = self.deeptext2(X["deeptext2"])
                out.add_(out_text)

            if self.deepimage is not None:
                out.add_(self.deepimage(X["deepimage"]))
            return out

    # 设置训练时用的参数
    def compile(self, method, optimizers_dic=None, lr_schedulers_dic=None, initializers_dic=None, with_focal_loss=False, alpha=0.25, gamma=2, verbose=1, seed=1, class_weight: Optional[Union[float, List[float], Tuple[float]]] = None):
        """
        :param method:                          str
        :param optimizers_dic:                  Dict[str, Optimizer]
        :param lr_schedulers_dic:               Dict[str, _LRScheduler]
        :param initializers_dic:                Dict[str, Initializer]
        :param with_focal_loss:                 bool
        :param alpha:
        :param gamma:
        :param verbose:
        :param seed:
        :return:
        """

        self.verbose = verbose
        self.seed = seed
        self.early_stop = False
        self.method = method
        self.with_focal_loss = with_focal_loss
        if self.with_focal_loss:
            self.alpha, self.gamma = alpha, gamma

        if isinstance(class_weight, float):
            self.class_weight = torch.tensor([1.0 - class_weight, class_weight])
        elif isinstance(class_weight, (tuple, list)):
            self.class_weight = torch.tensor(class_weight)
        else:
            self.class_weight = None

        if initializers_dic is not None:
            # Init 各子模型参数 e.g. {"wide": KaimingNormal(), "deepdense": XavierNormal()}
            # 实例化
            instantiated_initializers = {}
            for model_name, initializer in initializers_dic.items():
                print("model:{} ParamInitializer:{}".format(model_name, str(initializer)))
                instantiated_initializers[model_name] = initializer()

            # 遍历 WideDeep 的子模型. wide==>nn.Module deepdense==>nn.Module etc.
            for name, child in self.named_children():
                print("\nstart init {} param.....".format(name))
                instantiated_initializers[name](child)

        if optimizers_dic is not None:
            # Valid optimizer e.g. self.optimizer_dic = {"wide": torch.optim.Adam, "deepdense": RAdam}
            print("\n")
            for model_name, opt in optimizers_dic.items():
                print("model:{0:>1}     optimizer:{1}".format(model_name, opt.__class__.__name__))

            opt_names = list(optimizers_dic.keys())
            mod_names = [n for n, c in self.named_children()]

            for mn in mod_names:
                assert mn in opt_names, "No optimizer found for {}".format(mn)
                self.optimizer_dic = optimizers_dic

        if lr_schedulers_dic is not None:
            # self.lr_schedulers_dic = {"wide": torch.optim.lr_scheduler.StepLR, "deepdense": torch.optim.lr_scheduler.StepLR}
            # https://blog.csdn.net/Strive_For_Future/article/details/83213971
            self.lr_schedulers_dic = lr_schedulers_dic
            sc_name_list = [sc.__class__.__name__.lower() for model_name, sc in self.lr_schedulers_dic.items()]
            self.cyclic = any(["cycl" in sn for sn in sc_name_list])

        if use_cuda:
            print("use cuda!")
            self.cuda()

    # TODO: Metrics, EarlyStopping, ModelCheckpoint, EachModel-LR
    # TODO: DeepDense参数改为 batchnorm 会报错
    def fit(self, X_wide=None, X_deep=None, X_text=None, X_text2=None, X_img=None, X_train=None, X_val=None, val_split=None, target=None, n_epochs=1, validation_freq=1, batch_size=256, summary_path='log/',
        warm_up=False, warm_epochs=4, warm_max_lr=0.01, warm_deeptext_gradual=False,warm_deeptext_max_lr=0.01,warm_deeptext_layers=None,warm_deepimage_gradual=False,warm_deepimage_max_lr=0.01,warm_deepimage_layers=None,warm_routine="howard"
    ):
        """
        :param X_wide:                              np.ndarray
        :param X_deep:                              np.ndarray
        :param X_text:                              np.ndarray
        :param X_img:                               np.ndarray
        :param X_train:                             Dict[str, np.ndarray]
        :param X_val:                               Dict[str, np.ndarray]
        :param val_split:                           float
        :param target:                              np.ndarray
        :param n_epochs:                            int
        :param validation_freq:                     int
        :param batch_size:
        :param warm_up:                             bool
        :param warm_epochs:                         int
        :param warm_max_lr:                         float
        :param warm_deeptext_gradual:               bool
        :param warm_deeptext_max_lr:                float
        :param warm_deeptext_layers:                List[nn.Module]
        :param warm_deepimage_gradual:              bool
        :param warm_deepimage_max_lr:               float
        :param warm_deepimage_layers:               List[nn.Module]
        :return:
        """

        if X_train is None and (X_wide is None or X_deep is None or target is None):
            raise ValueError("Training data is missing")

        self.batch_size = batch_size
        # 返回 WideDeepDataset 枚举类型
        train_set, eval_set = _train_val_split(self, X_wide, X_deep, X_text, X_text2, X_img, X_train, X_val, val_split, target)
        train_loader = DataLoader(dataset=train_set, batch_size=batch_size, num_workers=n_cpus)
        eval_loader = DataLoader(dataset=eval_set, batch_size=batch_size, num_workers=n_cpus, shuffle=False)

        if warm_up:
            # warm up...
            self._warm_up(
                train_loader,
                warm_epochs,
                warm_max_lr,
                warm_deeptext_gradual,
                warm_deeptext_layers,
                warm_deeptext_max_lr,
                warm_deepimage_gradual,
                warm_deepimage_layers,
                warm_deepimage_max_lr,
                warm_routine,
            )

        dev_best_loss = float('inf')
        batch_num = 0  # 记录训练了第几个batch
        writer = SummaryWriter(log_dir=summary_path + '/' + time.strftime('%m-%d_%H.%M', time.localtime()))
        for epoch in range(n_epochs):
            # train step...
            epoch_logs: Dict[str, float] = {}
            # self.callback_container.on_epoch_begin(epoch, logs=epoch_logs)
            self.train_running_loss = 0.0
            print('Epoch [{}/{}]'.format(epoch + 1, n_epochs))

            for batch_idx, (data, target) in enumerate(train_loader):
                st = time.clock()
                batch_num += 1
                self.train()
                X = {k: v.cuda() for k, v in data.items()} if use_cuda else data
                y = target.float() if self.method != "multiclass" else target
                y = y.cuda() if use_cuda else y

                # 梯度置零
                for _, op in self.optimizer_dic.items():
                    op.zero_grad()

                # 预测值
                y_pred = self._activation_fn(self.forward(X))

                # cal loss and accurate
                train_loss = self._cal_loss_and_backprop(y_pred, y)

                if batch_num % 100 == 0:
                    train_acc = self._cal_binary_accuracy(y_pred, y)
                    loss_valid, acc_valid = self._test_validation_set(eval_loader)

                    writer.add_scalar("loss/train", train_loss.item(), batch_num)
                    writer.add_scalar("loss/eval", loss_valid.item(), batch_num)
                    writer.add_scalar("acc/train", train_acc, batch_num)
                    writer.add_scalar("acc/eval", acc_valid, batch_num)
                    ed = time.clock()

                    msg = 'Iter: {0:>6},  Train Loss: {1:>5.3},  Train Acc: {2:>6.3%},  Val Loss: {3:>5.3},  Val Acc: {4:>6.3%} Cost:{5:>3} seconds'
                    # msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%}'
                    print(msg.format(batch_num, train_loss.item(), train_acc, loss_valid, acc_valid, ed-st))

                if self.lr_schedulers_dic:
                    self._lr_scheduler_step(step_location="on_batch_end")

            if self.lr_schedulers_dic:
                self._lr_scheduler_step(step_location="on_epoch_end")
            # if self.early_stop:
            #     self.callback_container.on_train_end(epoch_logs)
            #     break
            # self.callback_container.on_train_end(epoch_logs)

        writer.close()
        self.train()


    def _cal_binary_accuracy(self, y_pred: Tensor, y_true: Tensor) -> np.ndarray:
        if self.method == "binary":
            y_pred_round = y_pred.round()
            correct_count = y_pred_round.eq(y_true.view(-1, 1)).float().sum().item()
            total_count = len(y_pred)
            accuracy = float(correct_count) / float(total_count)
            return np.round(accuracy, 4)
        else:
            y_pred_round = y_pred.round()
            correct_count = y_pred_round.eq(y_true.view(-1, 1)).float().sum().item()
            total_count = len(y_pred)
            accuracy = float(correct_count) / float(total_count)
            return np.round(accuracy, 4)

    def _cal_loss_and_backprop(self, y_pred, y_true, train_mode=True):
        loss = 0.0
        if self.method == "binary":
            loss = F.binary_cross_entropy(y_pred, y_true.view(-1, 1), weight=self.class_weight)
        if self.method == "regression":
            loss = F.mse_loss(y_pred, y_true.view(-1, 1))
        if self.method == "multiclass":
            loss = F.cross_entropy(y_pred, y_true, weight=self.class_weight)

        if train_mode:
            # 求梯度
            loss.backward()

            # optimizer更新参数空间
            for _, op in self.optimizer_dic.items():
                op.step()

        return loss

    def _activation_fn(self, inp: Tensor) -> Tensor:
        if self.method == "binary":
            return torch.sigmoid(inp)
        else:
            # F.cross_entropy will apply logSoftmax to the preds in the case of 'multiclass'
            return inp

    # update learning_rate
    def _lr_scheduler_step(self, step_location: str):
        # CyclicLR 每个batch, 其他每个 epoch
        # if self.cyclic:
        if step_location == "on_batch_end":
            for model_name, scheduler in self.lr_schedulers_dic.items():
                if "cycl" in scheduler.__class__.__name__.lower():
                    scheduler.step()
        elif step_location == "on_epoch_end":
            for model_name, scheduler in self.lr_schedulers_dic.items():
                if "cycl" not in scheduler.__class__.__name__.lower():
                    scheduler.step()

    # test loss/accurate of validation data
    def _test_validation_set(self, eval_loader):
        self.eval()
        loss_total = 0
        correct_count_total = 0.0
        predict_all = np.array([], dtype=int)
        labels_all = np.array([], dtype=int)

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(eval_loader):
                X = {k: v.cuda() for k, v in data.items()} if use_cuda else data
                y = target.float() if self.method != "multiclass" else target
                y = y.cuda() if use_cuda else y

                # 预测值
                y_pred = self._activation_fn(self.forward(X))

                loss = self._cal_loss_and_backprop(y_pred, y, train_mode=False)
                loss_total += loss

                y_pred_round = y_pred.round()
                predic = y_pred_round.data.cpu().numpy()

                labels = y.data.cpu().numpy()
                labels_all = np.append(labels_all, labels)
                predict_all = np.append(predict_all, predic)
                # correct_count_total += y_pred_round.eq(y.view(-1, 1)).float().sum().item()

        loss_valid = loss_total / len(eval_loader)
        acc = metrics.accuracy_score(labels_all, predict_all)
        return loss_valid, acc



    def predict(self,X_wide: np.ndarray,X_deep: np.ndarray,X_text: Optional[np.ndarray] = None,X_img: Optional[np.ndarray] = None,X_test: Optional[Dict[str, np.ndarray]] = None) -> np.ndarray:
        r"""
        fit method that must run after calling 'compile'

        Parameters
        ----------
        X_wide: np.ndarray, Optional. Default=None
            One hot encoded wide input.
        X_deep: np.ndarray, Optional. Default=None
            Input for the deepdense model
        X_text: np.ndarray, Optional. Default=None
            Input for the deeptext model
        X_img : np.ndarray, Optional. Default=None
            Input for the deepimage model
        X_test: Dict, Optional. Default=None
            Testing dataset for the different model branches.  Keys are
            'X_wide', 'X_deep', 'X_text', 'X_img' and 'target' the values are
            the corresponding matrices e.g X_train = {'X_wide': X_wide,
            'X_wide': X_wide, 'X_text': X_text, 'X_img': X_img}

        **WideDeep assumes that X_wide, X_deep and target ALWAYS exist, while
        X_text and X_img are optional

        Returns
        -------
        preds: np.array with the predicted target for the test dataset.
        """
        preds_l = self._predict(X_wide, X_deep, X_text, X_img, X_test)
        if self.method == "regression":
            return np.vstack(preds_l).squeeze(1)
        if self.method == "binary":
            preds = np.vstack(preds_l).squeeze(1)
            return (preds > 0.5).astype("int")
        if self.method == "multiclass":
            preds = np.vstack(preds_l)
            return np.argmax(preds, 1)

    def predict_proba(self,X_wide: np.ndarray,X_deep: np.ndarray,X_text: Optional[np.ndarray] = None,X_img: Optional[np.ndarray] = None,X_test: Optional[Dict[str, np.ndarray]] = None) -> np.ndarray:
        r"""
        Returns
        -------
        preds: np.ndarray
            Predicted probabilities of target for the test dataset for  binary
            and multiclass methods
        """
        preds_l = self._predict(X_wide, X_deep, X_text, X_img, X_test)
        if self.method == "binary":
            preds = np.vstack(preds_l).squeeze(1)
            probs = np.zeros([preds.shape[0], 2])
            probs[:, 0] = 1 - preds
            probs[:, 1] = preds
            return probs
        if self.method == "multiclass":
            return np.vstack(preds_l)

    def get_embeddings(
        self, col_name: str, cat_encoding_dict: Dict[str, Dict[str, int]]
    ) -> Dict[str, np.ndarray]:
        r"""
        Get the learned embeddings for the categorical features passed through deepdense.

        Parameters
        ----------
        col_name: str,
            Column name of the feature we want to get the embeddings for
        cat_encoding_dict: Dict
            Categorical encodings. The function is designed to take the
            'encoding_dict' attribute from the DeepPreprocessor class. Any
            Dict with the same structure can be used

        Returns
        -------
        cat_embed_dict: Dict
            Categorical levels of the col_name feature and the corresponding
            embeddings

        Example:
        -------
        Assuming we have already train the model:

        >>> model.get_embeddings(col_name='education', cat_encoding_dict=deep_preprocessor.encoding_dict)
        {'11th': array([-0.42739448, -0.22282735,  0.36969638,  0.4445322 ,  0.2562272 ,
        0.11572784, -0.01648579,  0.09027119,  0.0457597 , -0.28337458], dtype=float32),
         'HS-grad': array([-0.10600474, -0.48775527,  0.3444158 ,  0.13818645, -0.16547225,
        0.27409762, -0.05006042, -0.0668492 , -0.11047247,  0.3280354 ], dtype=float32),
        ...
        }

        where:

        >>> deep_preprocessor.encoding_dict['education']
        {'11th': 0, 'HS-grad': 1, 'Assoc-acdm': 2, 'Some-college': 3, '10th': 4, 'Prof-school': 5,
        '7th-8th': 6, 'Bachelors': 7, 'Masters': 8, 'Doctorate': 9, '5th-6th': 10, 'Assoc-voc': 11,
        '9th': 12, '12th': 13, '1st-4th': 14, 'Preschool': 15}
        """
        for n, p in self.named_parameters():
            if "embed_layers" in n and col_name in n:
                embed_mtx = p.cpu().data.numpy()
        encoding_dict = cat_encoding_dict[col_name]
        inv_encoding_dict = {v: k for k, v in encoding_dict.items()}
        cat_embed_dict = {}
        for idx, value in inv_encoding_dict.items():
            cat_embed_dict[value] = embed_mtx[idx]
        return cat_embed_dict

    def _loss_fn(self, y_pred: Tensor, y_true: Tensor) -> Tensor:  # type: ignore
        if self.with_focal_loss:
            return FocalLoss(self.alpha, self.gamma)(y_pred, y_true)
        if self.method == "regression":
            return F.mse_loss(y_pred, y_true.view(-1, 1))
        if self.method == "binary":
            return F.binary_cross_entropy(
                y_pred, y_true.view(-1, 1), weight=self.class_weight
            )
        if self.method == "multiclass":
            return F.cross_entropy(y_pred, y_true, weight=self.class_weight)


    def _warm_up(self, loader: DataLoader, n_epochs: int, max_lr: float, deeptext_gradual: bool, deeptext_layers: List[nn.Module], deeptext_max_lr: float, deepimage_gradual: bool, deepimage_layers: List[nn.Module], deepimage_max_lr: float, routine: str = "felbo",
    ):
        r"""
        Simple wrappup to individually warm up model components
        """
        if self.deephead is not None:
            raise ValueError(
                "Currently warming up is only supported without a fully connected 'DeepHead'"
            )
        # This is not the most elegant solution, but is a soluton "in-between"
        # a non elegant one and re-factoring the whole code
        warmer = WarmUp(
            self._activation_fn, self._loss_fn, self.metric, self.method, self.verbose
        )
        warmer.warm_all(self.wide, "wide", loader, n_epochs, max_lr)
        warmer.warm_all(self.deepdense, "deepdense", loader, n_epochs, max_lr)
        if self.deeptext:
            if deeptext_gradual:
                warmer.warm_gradual(
                    self.deeptext,
                    "deeptext",
                    loader,
                    deeptext_max_lr,
                    deeptext_layers,
                    routine,
                )
            else:
                warmer.warm_all(self.deeptext, "deeptext", loader, n_epochs, max_lr)
        if self.deepimage:
            if deepimage_gradual:
                warmer.warm_gradual(
                    self.deepimage,
                    "deepimage",
                    loader,
                    deepimage_max_lr,
                    deepimage_layers,
                    routine,
                )
            else:
                warmer.warm_all(self.deepimage, "deepimage", loader, n_epochs, max_lr)


    def _training_step(self, data: Dict[str, Tensor], target: Tensor, batch_idx: int):
        self.train()
        X = {k: v.cuda() for k, v in data.items()} if use_cuda else data
        y = target.float() if self.method != "multiclass" else target
        y = y.cuda() if use_cuda else y

        self.optimizer_dic.zero_grad()
        y_pred = self._activation_fn(self.forward(X))
        # a = torch.where(y_pred >= 0.5, 0, y_pred)
        # train_acc = metrics.accuracy_score(a, y)

        loss = self._loss_fn(y_pred, y)
        loss.backward()
        self.optimizer_dic.step()

        self.train_running_loss += loss.item()
        avg_loss = self.train_running_loss / (batch_idx + 1)

        if self.metric is not None:
            acc = self.metric(y_pred, y)
            return acc, avg_loss,loss
        else:
            return None, avg_loss, loss


    def _validation_step(self, data: Dict[str, Tensor], target: Tensor, batch_idx: int):

        self.eval()
        with torch.no_grad():

            X = {k: v.cuda() for k, v in data.items()} if use_cuda else data
            y = target.float() if self.method != "multiclass" else target
            y = y.cuda() if use_cuda else y

            y_pred = self._activation_fn(self.forward(X))
            loss = self._loss_fn(y_pred, y)
            self.valid_running_loss += loss.item()
            avg_loss = self.valid_running_loss / (batch_idx + 1)

        if self.metric is not None:
            acc = self.metric(y_pred, y)
            return acc, avg_loss
        else:
            return None, avg_loss

    def _predict(
        self,
        X_wide: np.ndarray,
        X_deep: np.ndarray,
        X_text: Optional[np.ndarray] = None,
        X_img: Optional[np.ndarray] = None,
        X_test: Optional[Dict[str, np.ndarray]] = None,
    ) -> List:
        r"""
        Hidden method to avoid code repetition in predict and predict_proba.
        For parameter information, please, see the .predict() method
        documentation
        """
        if X_test is not None:
            test_set = WideDeepDataset(**X_test)
        else:
            load_dict = {"X_wide": X_wide, "X_deep": X_deep}
            if X_text is not None:
                load_dict.update({"X_text": X_text})
            if X_img is not None:
                load_dict.update({"X_img": X_img})
            test_set = WideDeepDataset(**load_dict)

        test_loader = DataLoader(
            dataset=test_set,
            batch_size=self.batch_size,
            num_workers=n_cpus,
            shuffle=False,
        )
        test_steps = (len(test_loader.dataset) // test_loader.batch_size) + 1

        self.eval()
        preds_l = []
        with torch.no_grad():
            with trange(test_steps, disable=self.verbose != 1) as t:
                for i, data in zip(t, test_loader):
                    t.set_description("predict")
                    X = {k: v.cuda() for k, v in data.items()} if use_cuda else data
                    preds = self._activation_fn(self.forward(X))
                    if self.method == "multiclass":
                        preds = F.softmax(preds, dim=1)
                    preds = preds.cpu().data.numpy()
                    preds_l.append(preds)
        self.train()
        return preds_l
