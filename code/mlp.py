# coding: utf-8
import numpy as np
np.random.seed(42)

import tensorflow as tf
tf.compat.v1.set_random_seed(42)

#print(tf.__version__)

from base import Classifier, Regressor
from hyperopt import hp
from keras import backend as K
from os import environ
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras import layers, Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.optimizers import Adam
from tensorflow.python.util import deprecation


# Surpress deprecation warnings
deprecation._PRINT_DEPRECATION_WARNINGS = False

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class MLPClassifier(Classifier):

    def __init__(self, data, output_dir):

        super().__init__(data, output_dir)

        self.name = type(self).__name__
        self.n_input = data.scaled_x.shape[1]

        n_units_min = 1
        n_units_max = 24

        self.hyperspace = {

            'layers': hp.choice('layers', [
                {
                    'n_layers': 1,
                    'n_units_layer': [
                        self.uniform_int('n_units_layer_11', n_units_min, n_units_max),
                    ],
                },
                {
                    'n_layers': 2,
                    'n_units_layer': [
                        self.uniform_int('n_units_layer_21', n_units_min, n_units_max),
                        self.uniform_int('n_units_layer_22', n_units_min, n_units_max),
                    ],
                },
                {
                    'n_layers': 3,
                    'n_units_layer': [
                        self.uniform_int('n_units_layer_31', n_units_min, n_units_max),
                        self.uniform_int('n_units_layer_32', n_units_min, n_units_max),
                        self.uniform_int('n_units_layer_33', n_units_min, n_units_max),
                    ],
                }
            ])
        }


    @staticmethod
    def uniform_int(name, lower, upper):
        return hp.quniform(name, lower, upper, q=1)


    def _create_classifier(self, space, x_train, y_train):
        """Build multi-layer perceptron with current hyperparameter set.

        Args:
            space (dict):  Dictionary of hyperspace parameters and ranges.
            x_train (arr): Training features.
            y_train (vec): Training labels.

        Returns:
            obj: Trained classifier.
        """

        K.clear_session()


        # Number of neurons in each layer
        layer_sizes = [int(n) for n in space['layers']['n_units_layer']]


        model = Sequential()

        for idx, n_hidden in enumerate(layer_sizes):

            if idx == 0:

                # Input layer
                model.add(layers.Dense(n_hidden, activation=tf.nn.relu,
                        kernel_initializer=glorot_uniform(seed=42),
                        input_shape=[self.n_input]))

            else:
                # Hidden layers
                model.add(layers.Dense(n_hidden, activation=tf.nn.relu,
                        kernel_initializer=glorot_uniform(seed=42)))


        # Output layer
        model.add(layers.Dense(1))


        model.compile(
            loss='mse',
            optimizer=Adam(),
            metrics=['mae', 'mse']
        )


        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=10
        )


        model.fit(x_train, y_train,
            epochs=1000,
            shuffle=False,
            validation_split = 0.2,
            verbose=0,
            callbacks=[early_stop]
        )

        return model


class MLPRegressor(Regressor):
    """Multi-layer perceptron (MLP) regressor.

    Args:
        Regressor (cls): Regressor base class.

    Returns:
        obj: Trained regressor.

    """

    def __init__(self, data, output_dir):

        super().__init__(data, output_dir)

        self.name = type(self).__name__


        try:

            self.n_input = data.scaled_x.shape[1]

        except:

            self.n_input = 12


        # Min/max number of neurons per hidden layer
        n_units_min = 1
        n_units_max = 24


        self.hyperspace = {

            'layers': hp.choice('layers', [
                {
                    'n_layers': 1,
                    'n_units_layer': [
                        self.uniform_int('n_units_layer_11',
                        n_units_min, n_units_max),
                    ],
                },
                {
                    'n_layers': 2,
                    'n_units_layer': [
                        self.uniform_int('n_units_layer_21',
                        n_units_min, n_units_max),
                        self.uniform_int('n_units_layer_22',
                        n_units_min, n_units_max),
                    ],
                },
                {
                    'n_layers': 3,
                    'n_units_layer': [
                        self.uniform_int('n_units_layer_31',
                        n_units_min, n_units_max),
                        self.uniform_int('n_units_layer_32',
                        n_units_min, n_units_max),
                        self.uniform_int('n_units_layer_33',
                        n_units_min, n_units_max),
                    ],
                }
            ])
        }


    @staticmethod
    def uniform_int(name, lower, upper):
        return hp.quniform(name, lower, upper, q=1)


    def _fit_regressor(self, space, x_train, y_train):
        """Build multi-layer perceptron with current hyperparameter set."""

        K.clear_session()


        # Number of neurons in each layer
        layer_sizes = [int(n) for n in space['layers']['n_units_layer']]


        model = Sequential()

        for idx, n_hidden in enumerate(layer_sizes):

            if idx == 0:

                # Input layer
                model.add(layers.Dense(n_hidden, activation=tf.nn.relu,
                        kernel_initializer=glorot_uniform(seed=42),
                        input_shape=[self.n_input]))

            else:
                # Hidden layers
                model.add(layers.Dense(n_hidden, activation=tf.nn.relu,
                        kernel_initializer=glorot_uniform(seed=42)))


        # Output layer
        model.add(layers.Dense(1))


        model.compile(
            loss='mse',
            optimizer=Adam(),
            metrics=['mae', 'mse']
        )


        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=10
        )


        model.fit(x_train, y_train,
            epochs=1000,
            shuffle=False,
            validation_split = 0.2,
            verbose=0,
            callbacks=[early_stop]
        )

        return model


    @staticmethod
    def _load_model(h5_file):

        return load_model(h5_file)


    @staticmethod
    def _save_model(model, space, filename):

        h5_file = '{}.h5'.format(filename)

        model.save(h5_file)

        return h5_file


    def _create_best(self):
        """Build multi-layer perceptron with best hyperparameter set."""

        space = self.best_space


        K.clear_session()


        layer_sizes = []
        layer_sizes_keys = []

        for key in space.keys():
            if str(int(space['layers'])+1) in key:
                layer_sizes_keys.append(key)

        layer_sizes_keys.sort()

        for key in layer_sizes_keys:
            layer_sizes.append(int(space[key]))


        model = Sequential()

        for idx, n_hidden in enumerate(layer_sizes):

            if idx == 0:

                # Input layer
                model.add(layers.Dense(n_hidden, activation=tf.nn.relu,
                        kernel_initializer=glorot_uniform(seed=42),
                        input_shape=[self.n_input]))

            else:

                # Hidden layers
                model.add(layers.Dense(n_hidden, activation=tf.nn.relu,
                        kernel_initializer=glorot_uniform(seed=42)))


        # Output layer
        model.add(layers.Dense(1))


        model.compile(
            loss='mse',
            optimizer=Adam(),
            metrics=['mae', 'mse']
        )


        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=10
        )

        
        model.fit(self.data.scaled_x, self.data.y,
            epochs=1000,
            shuffle=False,
            validation_split = 0.2,
            verbose=0,
            callbacks=[early_stop]
        )


        return model
