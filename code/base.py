# coding: utf-8
import numpy as np
np.random.seed(42)

from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
import pandas as pd
import pickle
from shutil import move
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, average_precision_score, f1_score
from time import time


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)


__author__ = "Miles Timpe, Maria Han Veiga, and Mischa Knabenhans"
__maintainer__ = "Miles Timpe"
__credits__ = ["Miles Timpe", "Maria Han Veiga", "Mischa Knabenhans"]
__email__ = "mtimpe@physik.uzh.ch"
__copyright__ = "Copyright 2020, ICS"
__license__ = "GNU General Public License v3.0"
__version__ = "1.0.0"
__status__ = "Production"


class Classifier:
    """Base class for classifier models."""

    def __init__(self, data, metric):
        self.data = data
        self.metric = metric


    @staticmethod
    def _classifier_metrics(y_test, y_pred):
        """Calculate and return basic metrics for a set of predictions."""

        accuracy  = accuracy_score(y_test, y_pred)
        f1_val    = f1_score(y_test, y_pred)
        precision = average_precision_score(y_test, y_pred)

        return({'accuracy': accuracy, 'f1': f1_val, 'precision': precision})


    def _hpo_info(self, hpo_evals):

        # Classifier hyperparameter optimization
        print('\n\nSearching for optimal classifier architecture')
        #print('\n\tCross-validating using 80/20 splits')
        print('\n\tTraining set   (N={})'.format(
            int(len(self.data.x.mtotal) * 0.8)))
        print('\tValidation set (N={})'.format(
            int(len(self.data.x.mtotal) * 0.2)))
        print('\n\tSearch pattern: Tree-structured Parzen Estimator')
        print('\tk-folds:        {}'.format(self.data.k_folds))
        print('\tTrain/validate  80/20')
        print('\tHPO iterations: {}'.format(hpo_evals))
        print('\tLoss function:  F1 score\n')


    def hpo(self, hpo_evals):
        """Find optimal classifier architecture with hyperopt."""

        self._hpo_info(hpo_evals)

        hpo_t0 = time()

        trials = Trials()

        self.best_space = fmin(fn=self._objective, space=self.hyperspace,
                               algo=tpe.suggest, max_evals=hpo_evals,
                               trials=trials)

        self.hpo_time = time() - hpo_t0

        self.best_trial = self._get_best_trial(trials)

        self._hpo_results()


    def _hpo_results(self):

        f1   = self.best_trial['result']['f1_score']
        acc  = self.best_trial['result']['accuracy']
        prec = self.best_trial['result']['precision']

        print('\n\tHPO required {:.1f} seconds'.format(self.hpo_time))

        print('\n\tAccuracy:  {:.4f}'.format(acc))
        print('\tPrecision: {:.4f}'.format(prec))
        print('\tF1 score:  {:.4f}'.format(f1))


    def _objective(self, space):
        """Objective function for hyperopt's fmin() function.

        Args:
            space (dict): Dictionary of hyperparameters and ranges.

        Returns:
            dict: hyperopt performance metrics.

        """

        acc_cv = []
        prec_cv = []
        f1_cv = []


        for k in range(self.data.k_folds):

            x_train, y_train, x_test, y_test = self.data.get_kth_fold(k=k)


            y_train = self.data._encode_one_hot(y_train)
            y_test  = self.data._encode_one_hot(y_test)


            # Train model and predict
            model = self._instantiate_classifier(space)

            model.fit(x_train, y_train)

            y_pred = model.predict(x_test)

            del model


            # Evaluate model
            metrics = self._classifier_metrics(y_test, y_pred)

            acc_cv.append(metrics['accuracy'])
            prec_cv.append(metrics['precision'])
            f1_cv.append(metrics['f1'])


        return({
            'loss':-np.mean(f1_cv),
            'accuracy': np.mean(acc_cv),
            'precision': np.mean(prec_cv),
            'f1_score': np.mean(f1_cv),
            'status': STATUS_OK
            })


    def _get_best_trial(self, trials):
        """Return the best trial from a hyperopt Trials object."""

        ok_list = [t for t in trials if STATUS_OK == t['result']['status']]

        losses = [float(t['result']['loss']) for t in ok_list]

        # Loss is the negative r2-score, so take minimum to get best trial.
        index_of_min_loss = np.argmin(losses)

        return ok_list[index_of_min_loss]


    def evaluate(self, x_test, y_test, pklfile):
        """Evaluate classifier performance on independent test set."""

        # Train best classifier architecture on full dataset
        x_train = self.data.scaled_x
        y_train = self.data._encode_one_hot(self.data.y)


        top_model = None
        top_score = None
        top_metrics = None
        top_preds = None
        top_training_preds = None

        if self.name == 'mlp':
            n_ensemble = 10
        else:
            n_ensemble = 1

        for n in range(n_ensemble):

            t0 = time()

            model = self._instantiate_classifier(self.best_space)

            model.fit(x_train, y_train)

            # Predictions on training set
            y_training_pred = model.predict(x_train)

            # Predictions on test set
            y_pred = model.predict(x_test)

            # One-hot encode labels
            y_test = self.data._encode_one_hot(y_test)

            # Evaluate classifier metrics
            metrics = self._classifier_metrics(y_test, y_pred)

            metrics['training_time'] = time() - t0

            # Confusion matrix
            TN, FP, FN, TP = confusion_matrix(y_test, y_pred).ravel()

            metrics['tn'] = TN
            metrics['fp'] = FP
            metrics['fn'] = FN
            metrics['tp'] = TP

            print('\t{:02d}\tF1 score:  {:.4f}'.format(n+1, metrics['f1']))

            # Keep track of best model and score
            if not top_model:
                top_score = metrics['f1']
                top_metrics = metrics
                top_model = model
                top_preds = y_pred
                top_training_preds = y_training_pred
            elif top_score < metrics['f1']:
                top_score = metrics['f1']
                top_metrics = metrics
                top_model = model
                top_preds = y_pred
                top_training_preds = y_training_pred

            del model

        print('\n\tTP: {}\n\tFP: {}\n\tTN: {}\n\tFN: {}'.format(
            metrics['tp'], metrics['fp'], metrics['tn'], metrics['fn']))

        print('\n\tBest Accuracy:  {:.4f}'.format(top_metrics['accuracy']))
        print('\tBest Precision: {:.4f}'.format(top_metrics['precision']))
        print('\tBest F1 score:  {:.4f}'.format(top_metrics['f1']))

        # Save model
        print('\nSerializing best model at {}'.format(pklfile))
        pickle.dump(top_model, open(pklfile, "wb"))

        # Save predicted training zero mask
        if self.data.tss < 10000:
            lhs_file = '12D_LHS10K_TSS_{:05d}'.format()
        else:
            lhs_file = '12D_LHS10K'

        self._save_mask(lhs_file, top_training_preds)

        # Save predicted test zero mask
        self._save_mask('12D_LHS500', top_preds)


        del top_model

        return(metrics, y_test, y_pred)


    def _save_mask(self, lhs, y_pred):

        masked_file = '../data/csv/{}_masked.csv'.format(lhs)

        df = pd.read_csv("../data/csv/{}.csv".format(lhs))

        if self.data.target == 'lr_mass':
            col_name = 'lr_exists'
        elif self.data.target == 'slr_mass':
            col_name = 'slr_exists'

        df[col_name] = y_pred

        df.to_csv(masked_file, index=False)



class Regressor:
    """Base class for regressor models."""

    def __init__(self, data, output_dir):

        self.data = data
        self.output_dir = output_dir
        self.x_scaler = None
        self.y_scaler = None
        self.best_model = None
        self.best_score = None


    def hpo(self, hpo_evals):

        with open('{}/hpo.csv'.format(self.output_dir), 'w') as f:
            f.write('f1,f2,f3,f4,f5,mean\n')

        self._hpo_info(hpo_evals)

        hpo_t0 = time()

        trials = Trials()

        self.hpo_iter = 0

        self.best_space = fmin(fn=self._objective, space=self.hyperspace,
                               algo=tpe.suggest, max_evals=hpo_evals,
                               trials=trials)

        self.hpo_time = time() - hpo_t0

        print('\n\tRegressor HPO took {:.1f} seconds'.format(self.hpo_time))

        best_trial = self._get_best_trial(trials)


        r2   = -best_trial['result']['loss']
        mbe  = best_trial['result']['mbe']
        mae  = best_trial['result']['mae']
        rmse = best_trial['result']['rmse']

        return({'r2':r2, 'mbe':mbe, 'mae':mae, 'rmse':rmse, 'nan':0,
                'vss':int(len(self.data.x.mtotal)*0.2)})


    def _hpo_info(self, hpo_evals):

        print('\n\tSearch pattern: Tree-structured Parzen Estimator')
        print('\tHPO iterations: {}'.format(hpo_evals))
        print('\tLoss function:  r2-score\n')


    @staticmethod
    def _regression_metrics(y_test, y_pred):

        nanless_test = []
        nanless_pred = []

        for yt, yp in zip(y_test, y_pred):

            if not np.isnan(yt):
                nanless_test.append(yt)
                nanless_pred.append(yp)

        y_test = np.array(nanless_test)
        y_pred = np.array(nanless_pred)

        nan_count = int(len(y_test) - len(nanless_test))

        r2 = r2_score(y_test, y_pred)
        mbe = np.mean(y_pred - y_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        return({'r2':r2, 'mbe':mbe, 'mae':mae, 'rmse':rmse, 'nan':nan_count,
                'y_true':y_test, 'y_pred':y_pred, 'vss':len(y_test)})


    def _objective(self, space):

        self.hpo_iter += 1

        # Keep track of scores across all folds
        r2_cv, mbe_cv, mae_cv, rmse_cv = [], [], [], []

        cv_score = None
        cv_model = None


        for k in range(self.data.k_folds):

            x_train, y_train, x_test, y_test = self.data.get_kth_fold(k=k)

            model = self._fit_regressor(space, x_train, y_train)

            model_file = '{}/hpo/hpo_{:03d}_{}'.format(
                self.output_dir, self.hpo_iter, k+1)

            model_path = self._save_model(model, space, model_file)

            y_pred = model.predict(x_test)

            y_pred = self._squeeze_predictions(y_pred)

            del model


            metrics = self._regression_metrics(y_test, y_pred)


            if not cv_model:
                cv_score = metrics['r2']
                cv_model = model_path

            elif cv_score < metrics['r2']:
                cv_score = metrics['r2']
                cv_model = model_path


            r2_cv.append(metrics['r2'])
            mbe_cv.append(metrics['mbe'])
            mae_cv.append(metrics['mae'])
            rmse_cv.append(metrics['rmse'])


        mean_r2   = np.mean(r2_cv)
        mean_mbe  = np.mean(mbe_cv)
        mean_mae  = np.mean(mae_cv)
        mean_rmse = np.mean(rmse_cv)


        ext = cv_model.split('.')[-1]

        best_arch = '{}/best/best.{}'.format(self.output_dir, ext)


        r2str = ','.join([str(x) for x in r2_cv])

        with open('{}/hpo.csv'.format(self.output_dir), 'a') as f:
            f.write('{},{}\n'.format(r2str, mean_r2))


        if not self.best_score:

            self.best_score = mean_r2
            self.best_model = best_arch

            move(cv_model, best_arch)

        elif self.best_score < mean_r2:

            self.best_score = mean_r2
            self.best_model = best_arch

            move(cv_model, best_arch)


        return{'loss':-mean_r2, 'model': cv_model,
               'r2': mean_r2, 'mbe': mean_mbe, 'mae': mean_mae,
               'rmse': mean_rmse, 'status': STATUS_OK}


    @staticmethod
    def _get_best_trial(trials):

        ok_list = [t for t in trials if STATUS_OK == t['result']['status']]

        losses = [float(t['result']['loss']) for t in ok_list]

        # Loss is the negative r2-score, so take minimum to get best trial.
        index_of_min_loss = np.argmin(losses)

        return ok_list[index_of_min_loss]


    def evaluate_best(self, data):

        model = self._load_model(self.best_model)

        y_pred = model.predict(data.scaled_x)

        y_pred = self._squeeze_predictions(y_pred)

        metrics = self._regression_metrics(data.scaled_y, y_pred)

        y_pred = y_pred.reshape(1, -1)

        y_unscaled = data.y_scaler.inverse_transform(y_pred)

        y_unscaled = y_unscaled[0]

        y_true = data.y
        y_pred = y_unscaled

        return(metrics, y_true, y_pred)


    @staticmethod
    def _squeeze_predictions(y_pred):

        try:

            if y_pred.shape[1] == 1:

                return(np.squeeze(y_pred))

        except:

            return(y_pred)
