import numpy as np
import torch

from sklearn.utils import Bunch
from torch.utils.data import Dataset



class WideDeepDataset(Dataset):
    def __init__(self, X_wide, X_deep, target=None, X_text=None, X_text2=None, X_img=None):
        """
        :param X_wide:                      np.ndarray
        :param X_deep:                      np.ndarray
        :param target:                      np.ndarray
        :param X_text:                      np.ndarray
        :param X_img:                       np.ndarray
        """
        self.X_wide = X_wide
        self.X_deep = X_deep
        self.X_text = X_text
        self.X_text2 = X_text2
        self.X_img = X_img
        self.Y = target

    def __getitem__(self, idx: int):
        # 获取指定 idx 行. 获取列方法: X_wide[: dx]
        X = Bunch(wide=self.X_wide[idx])
        X.deepdense = self.X_deep[idx]

        if self.X_text is not None:
            X.deeptext = self.X_text[idx]

        if self.X_text2 is not None:
            X.deeptext2 = self.X_text2[idx]

        if self.Y is not None:
            y = self.Y[idx]
            return X, y
        else:
            return X

    def __len__(self):
        return len(self.X_deep)
