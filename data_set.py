# -*- coding: utf-8 -*-
"""
读入libsvm类型的数据，拆分为特征和目标值
"""
from sklearn.datasets import load_svmlight_file
from collections import Counter
import numpy as np


class DataSet:
    def __init__(self, filename):
        # read data
        self.features, self.target = load_svmlight_file(filename)
        # self.features = np.array(self.features.data).reshape([-1, 24])
        # tmp = self.features[:, 17]
        # self.features[:, 17] = self.features[:, 16]
        # self.features[:, 16] = tmp
        print(Counter(self.target))


if __name__ == "__main__":
    dataset = DataSet('out_result/feature_importance_dt_0826.csv')
