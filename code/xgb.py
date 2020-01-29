# coding: utf-8
import numpy as np
np.random.seed(42)

from hyperopt import hp
import pickle
import xgboost as xgb
from xgboost import XGBClassifier, XGBRegressor

from base import Classifier, Regressor


print(xgb.__version__)


class XGBoostClassifier(Classifier):

    def __init__(self, data, metric):

        super().__init__(data, metric)

        self.name = type(self).__name__

        self.hyperspace = {
            'max_depth': hp.quniform('max_depth', 3, 12, 1),
            'n_estimators': hp.quniform('n_estimators', 1, 1000, 10),
            'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1)
        }


    def _instantiate_classifier(self, space):

        return XGBClassifier(
            objective='binary:logistic',
            learning_rate=0.1,
            n_estimators=int(space['n_estimators']),
            max_depth=int(space['max_depth']),
            min_child_weight=3, gamma=0.2, subsample=0.6,
            colsample_bytree=space['colsample_bytree'],
            scale_pos_weight=1,
            seed=42)



class XGBoostRegressor(Regressor):

    def __init__(self, data, output_dir):

        super().__init__(data, output_dir)

        self.name = type(self).__name__

        self.hyperspace = {
            'max_depth': hp.quniform("max_depth", 3, 12, 1),
            'n_estimators': hp.quniform('n_estimators', 1, 1000, 10),
            'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1)
        }


    @staticmethod
    def _load_model(pklfile):

        with open(pklfile, "rb") as pkl:

            model = pickle.load(pkl)

        return model


    @staticmethod
    def _save_model(model, space, filename):

        pklfile = '{}.pkl'.format(filename)

        with open(pklfile, "wb") as pkl:

            pickle.dump(model, pkl)

        return pklfile


    def _fit_regressor(self, space, x_train, y_train):

        model = XGBRegressor(
            objective ='reg:squarederror',
            colsample_bytree = space['colsample_bytree'],
            max_depth = int(space['max_depth']),
            n_estimators = int(space['n_estimators']),
            learning_rate = 0.1, alpha=10,
            random_state=42
        )

        model.fit(x_train, y_train)

        return model
