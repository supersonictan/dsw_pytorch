import numpy as np
from sklearn.model_selection import train_test_split
from torch import Tensor

from preprocessing.WideDeepDataset import WideDeepDataset




def _train_val_split(self, X_wide=None, X_deep=None, X_text=None, X_text2=None, X_img=None, X_train=None, X_val=None, val_split=None, target=None):
    """
    :param self                 nn.Module
    :param X_wide:              np.ndarray
    :param X_deep:              np.ndarray
    :param X_text:              np.ndarray
    :param X_img:               np.ndarray
    :param X_train:             Dict[str, np.ndarray]
    :param X_val:               Dict[str, np.ndarray]
    :param val_split:           float
    :param target:              np.ndarray
    :return:
    """
    #  Without validation
    if X_val is None and val_split is None:
        print("X_val is None and val_split is None...")

        if X_train is not None:
            X_wide, X_deep, target = (X_train["X_wide"], X_train["X_deep"], X_train["target"])

            if "X_text" in X_train.keys():
                print('"X_text" in X_train.keys()')
                X_text = X_train["X_text"]
            if "X_img" in X_train.keys():
                X_img = X_train["X_img"]

        X_train = {"X_wide": X_wide, "X_deep": X_deep, "target": target}

        try:
            X_train.update({"X_text": X_text})
        except:
            pass
        try:
            X_train.update({"X_img": X_img})
        except:
            pass

        train_set = WideDeepDataset(**X_train)
        eval_set = None

    #  With validation
    else:
        if X_val is not None:
            # if a validation dictionary is passed, then if not train
            # dictionary is passed we build it with the input arrays
            # (either the dictionary or the arrays must be passed)
            if X_train is None:
                X_train = {"X_wide": X_wide, "X_deep": X_deep, "target": target}
                if X_text is not None:
                    X_train.update({"X_text": X_text})
                if X_img is not None:
                    X_train.update({"X_img": X_img})
        else:
            # if a train dictionary is passed, check if text and image
            # datasets are present. The train/val split using val_split
            if X_train is not None:
                print("GaGaGa...")
                X_wide, X_deep, target = (X_train["X_wide"], X_train["X_deep"], X_train["target"])
                if "X_text" in X_train.keys():
                    X_text = X_train["X_text"]
                if "X_img" in X_train.keys():
                    X_img = X_train["X_img"]

            (X_tr_wide, X_val_wide, X_tr_deep, X_val_deep, y_tr, y_val) = train_test_split(X_wide, X_deep, target, test_size=val_split, random_state=self.seed,
                stratify=target if self.method != "regression" else None,
            )
            X_train = {"X_wide": X_tr_wide, "X_deep": X_tr_deep, "target": y_tr}
            X_val = {"X_wide": X_val_wide, "X_deep": X_val_deep, "target": y_val}
            try:
                X_tr_text, X_val_text = train_test_split(X_text, test_size=val_split, random_state=self.seed, stratify=target if self.method != "regression" else None)
                X_train.update({"X_text": X_tr_text}), X_val.update({"X_text": X_val_text})
            except:
                pass

            try:
                X_tr_text2, X_val_text2 = train_test_split(X_text2, test_size=val_split, random_state=self.seed, stratify=target if self.method != "regression" else None)
                X_train.update({"X_text2": X_tr_text2}), X_val.update({"X_text2": X_val_text2})
            except:
                pass

        train_set = WideDeepDataset(**X_train)
        eval_set = WideDeepDataset(**X_val)
    return train_set, eval_set