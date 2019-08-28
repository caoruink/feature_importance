# -*- coding: utf-8 -*-
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeRegressor
from data_set import DataSet

from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, median_absolute_error, \
    r2_score


class Predictors(DataSet):
    def __init__(self, filename):
        super().__init__(filename)

        # Cross-validated classification reports which are used to calculate the average
        self.train_model()

    def train_model(self):
        """
        Train with a model and make a five-fold cross-validation to draw a ROC graph based on the predicted results.
        :return:
        """
        kf = KFold(n_splits=5, shuffle=True)
        for train_index, val_index in kf.split(self.features):
            x_train, x_val = self.features[train_index], self.features[val_index]
            y_train, y_val = self.target[train_index], self.target[val_index]
            # self.test_specific_model(self.gbdt(), x_train, x_val, y_train, y_val, "gbdt")
            # self.test_specific_model(self.rf(), x_train, x_val, y_train, y_val, "rf")
            self.test_specific_model(self.dt(), x_train, x_val, y_train, y_val, "DT")
        return

    def test_specific_model(self, clf, x_train, x_test, y_train, y_test, clf_name=None):
        """
        train a model
        :param clf: model
        :param x_train:
        :param x_test:
        :param y_train:
        :param y_test:
        :return:
        """
        # train
        print("train the model".center(72, '*'))
        clf.fit(x_train, y_train)
        # predict
        y_predict = clf.predict(x_test)
        clf.fit(x_train, y_train)
        print(clf.feature_importances_)
        for each in clf.feature_importances_:
            file.write(str(each) + ',')
        file.write('\n')

        print('模型的平均绝对误差为：', mean_absolute_error(y_predict, y_test))
        print('模型的均方误差为：', mean_squared_error(y_predict, y_test))
        print('模型的中值绝对误差为：', median_absolute_error(y_predict, y_test))
        print('模型的可解释方差值为：', explained_variance_score(y_predict, y_test))
        print('模型的R^2值为：', r2_score(y_predict, y_test))
        return

    def gbdt(self):
        from sklearn.ensemble import GradientBoostingRegressor
        return GradientBoostingRegressor(learning_rate=0.01, n_estimators=1000, )

    def rf(self):
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(n_estimators=1000, n_jobs=8, max_depth=7, min_samples_leaf=20)

    def dt(self):
        return DecisionTreeRegressor()


if __name__ == '__main__':
    file = open("out_result/feature_importance_dt_xuefei_13.csv", 'w')
    predictor = Predictors('in_data/libsvm_data_0826.txt')
    file.close()
