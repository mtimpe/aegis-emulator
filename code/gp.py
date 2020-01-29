# coding: utf-8
from base import Regressor
from hyperopt import hp
import pickle
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, DotProduct
from sklearn.gaussian_process.kernels import ExpSineSquared, Matern
from sklearn.gaussian_process.kernels import RationalQuadratic


class GPRegressor(Regressor):

    def __init__(self, data, output_dir):

        super().__init__(data, output_dir)

        self.name = type(self).__name__

        self.hyperspace = {
            'matern': hp.choice('matern', [True, False]),
            'rationalQuad': hp.choice('rationalQuad', [True, False]),
            'constant': hp.uniform ('constant', 0.0, 5.0),
            'rbf': hp.uniform ('rbf', 0.0, 5.0),
            'alpha': hp.uniform ('alpha', 0.0, 0.01),
            'restarts': self.uniform_int('restarts', 0, 5)
        }


    def _load_model(self, pklfile):

        with open(pklfile, "rb") as pkl:

            space = pickle.load(pkl)

        model = self._fit_regressor(space,self.data.scaled_x,self.data.scaled_y)

        return model


    @staticmethod
    def _save_model(model, space, filename):

        pklfile = '{}.pkl'.format(filename)

        with open(pklfile, "wb") as pkl:

            pickle.dump(space, pkl)

        return pklfile


    @staticmethod
    def uniform_int(name, lower, upper):
        return hp.quniform(name, lower, upper, q=1)


    def _fit_regressor(self, space, x_train, y_train):

        kernel = (ConstantKernel(space['constant']) *
                  RBF(length_scale=space['rbf']))

        if space['matern']: kernel += Matern()

        if space['rationalQuad']: kernel += RationalQuadratic()


        model = GaussianProcessRegressor(
                kernel = kernel,
                alpha = space['alpha'],
                n_restarts_optimizer = int(space['restarts'])
        )

        model.fit(x_train, y_train)

        return model
