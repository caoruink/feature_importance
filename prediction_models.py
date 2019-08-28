# -*- coding: utf-8 -*-
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import classification_report
from data_set import DataSet


class Predictors(DataSet):
    def __init__(self, filename):
        super().__init__(filename)
        # Cross-validated classification
        self.train_model()

    def train_model(self):
        """
        Train with a model and make a five-fold cross-validation to draw a ROC graph based on the predicted results.
        :return:
        """
        kf = KFold(n_splits=10, shuffle=True)
        for train_index, val_index in kf.split(self.features):
            x_train, x_val = self.features[train_index], self.features[val_index]
            y_train, y_val = self.target[train_index], self.target[val_index]
            # self.test_specific_model(self.gbdt(), x_train, x_val, y_train, y_val, "gbdt")
            # self.test_specific_model(self.rf(), x_train, x_val, y_train, y_val, "rf")
            # self.test_specific_model(self.adaboost(), x_train, x_val, y_train, y_val, "ada")
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

        # print the classification report
        print("predict with test data and the classification report".center(72, '*'))
        # print(classification_report(y_test, y_predict, target_names=['无', '有']))
        print(classification_report(y_test, y_predict))
        return

    @staticmethod
    def adaboost():
        clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=5, class_weight='balanced'),
                                 n_estimators=1600, learning_rate=0.01)
        return clf

    def gbdt(self):
        from sklearn.ensemble import GradientBoostingClassifier
        return GradientBoostingClassifier(learning_rate=0.01, n_estimators=1000, )

    def rf(self):
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=1000, n_jobs=8, class_weight='balanced',
                                      max_depth=7, min_samples_leaf=20,)

    def dt(self):
        return DecisionTreeClassifier()


if __name__ == '__main__':
    file = open("out_result/feature_importance_dt_0826.csv", 'w')
    predictor = Predictors('in_data/libsvm_data_0826.txt')
    file.close()
