# coding: utf-8
import numpy as np
np.random.seed(42)

from base import Classifier
from hyperopt import hp
from sklearn.svm import SVC


class SVMClassifier(Classifier):

    def __init__(self, data, metric):

        super().__init__(data, metric)

        self.name = type(self).__name__

        self.hyperspace = {
            'C': hp.lognormal('C', -1, 1),
            'gamma': hp.loguniform('gamma', -3, 1),
            'kernel': hp.choice('kernel', [
                {'ktype': 'rbf'},
                {'ktype': 'poly', 'degree': hp.quniform('degree', 1, 5, 1)},
            ]),
        }


    def _instantiate_classifier(self, space):

        if space['kernel'] == 0:
            # RBF
            return SVC(
                kernel='rbf',
                C=space['C'],
                gamma=space['gamma'],
                max_iter=1e6,
                random_state=42)

        elif space['kernel'] == 1:
            # Poly
            return SVC(
                kernel='poly',
                degree=int(space['degree']),
                C=space['C'],
                gamma=space['gamma'],
                max_iter=1e6,
                random_state=42)

        elif space['kernel']['ktype'] == 'rbf':
            # RBF
            return SVC(
                kernel=space['kernel']['ktype'],
                C=space['C'],
                gamma=space['gamma'],
                max_iter=1e6,
                random_state=42)

        elif space['kernel']['ktype'] == 'poly':
            # Poly
            return SVC(
                kernel=space['kernel']['ktype'],
                degree=int(space['kernel']['degree']),
                C=space['C'],
                gamma=space['gamma'],
                max_iter=1e6,
                random_state=42)
