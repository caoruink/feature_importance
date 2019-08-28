# feature_importance
计算有监督学习数据的特征重要性，基于树类模型
## 文件描述
- data_set.py  读入libsvm类型的数据，拆分为特征和目标值
- prediction_models.py 分类模型。根据data_set读入的数据建立树类模型（决策树，GBDT,RF），后得到feature_importance，并写出文件。注意，做了十折交叉验证，所以输出了十行，每一列对应特征。需要制定**输入文件名**（数据文件）和**输出文件名**（feature_importace文件）
- regressor.py 回归模型。其他同上。

后可在excel中计算平均值，对特征排序。
