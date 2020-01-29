# coding: utf-8
import numpy as np
np.random.seed(42)

import pandas as pd
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler


__author__ = "Miles Timpe, Maria Han Veiga, and Mischa Knabenhans"
__maintainer__ = "Miles Timpe"
__credits__ = ["Miles Timpe", "Maria Han Veiga", "Mischa Knabenhans"]
__email__ = "mtimpe@physik.uzh.ch"
__copyright__ = "Copyright 2020, ICS"
__license__ = "GNU General Public License v3.0"
__version__ = "1.0.0"
__status__ = "Production"


class Dataset:

    def __init__(self, lhs, tss, feature_names, target, k_folds=5,
                 scaling_method='standard', filter_nan=True, use_null=True,
                 cascade=False, external_x_scaler=None, external_y_scaler=None):

        self.lhs = lhs
        self.tss = tss
        self.features = feature_names
        self.target = target
        self.collision_ids = None
        self.k_folds = k_folds
        self.scaling_method = scaling_method
        self.x_scaler = None
        self.y_scaler = None
        self.filter_nan = filter_nan
        self.use_null = use_null
        self.cascade = cascade
        self.external_x_scaler = external_x_scaler
        self.external_y_scaler = external_y_scaler

        self._message()

        self.x, self.scaled_x, self.y, self.scaled_y = self._load_dataset()

        self.kf = KFold(n_splits=self.k_folds, random_state=42, shuffle=True)


    @staticmethod
    def _encode_one_hot(val):
        """One-hot encode value. 1 if zero, 0 if non-zero."""

        encoded = []

        for y in val:
            if y > 0:
                encoded.append(0)
            else:
                encoded.append(1)

        return encoded


    def get_kth_fold(self, k=0):
        """Extract the k-th fold from the training set."""

        folds = list(self.kf.split(self.scaled_x))

        train_index, test_index = folds[k]

        return(self.scaled_x.iloc[train_index], self.scaled_y.iloc[train_index],
               self.scaled_x.iloc[test_index], self.scaled_y.iloc[test_index])


    def _message(self):

        print('\n\tDataset:        {}'.format(self.lhs))
        print('\tTSS:            {}'.format(self.tss))
        print('\tTarget:         {}'.format(self.target))
        print('\tScaler:         {}'.format(self.scaling_method))
        print('\tFilter NaNs:    {}'.format(self.filter_nan))
        print('\tUse null:       {}'.format(self.use_null))
        print('\tCascade:        {}'.format(self.cascade))


    def summary(self):

        print('\nlhs:      {}'.format(self.lhs))
        print('tss:      {}'.format(self.tss))
        print('\nfeatures:')

        for n, feat in enumerate(self.features):
            print('{0:>5}.  {1:<15}'.format(n+1, feat))

        print('\ntarget:   {}'.format(self.target))
        print('k-folds:  {}'.format(self.k_folds))
        print('scaler:   {}\n'.format(self.scaling_method))


    def _load_dataset(self):

        df = self._load_csv()

        Y = df.pop(self.target)
        X = df

        scaler = {
            'none': None,
            'minmax': MinMaxScaler(),
            'standard': StandardScaler(),
            'robust': RobustScaler()
        }

        if self.external_x_scaler:

            x_scaler = self.external_x_scaler

            scaled_X = x_scaler.transform(X)
            scaled_X = pd.DataFrame(scaled_X, columns=X.columns)

            self.x_scaler = x_scaler

        elif scaler[self.scaling_method]:

            x_scaler = StandardScaler()

            scaled_X = x_scaler.fit_transform(X)
            scaled_X = pd.DataFrame(scaled_X, columns=X.columns)

            self.x_scaler = x_scaler

        else:

            scaled_X = X


        if self.external_y_scaler:

            y_scaler = self.external_y_scaler

            scaled_Y = y_scaler.transform(Y.values.reshape(-1, 1))
            scaled_Y = pd.Series(data=np.squeeze(scaled_Y), name=Y.name)

            self.y_scaler = y_scaler

        elif scaler[self.scaling_method]:

            y_scaler = StandardScaler()

            scaled_Y = y_scaler.fit_transform(Y.values.reshape(-1, 1))
            scaled_Y = pd.Series(data=np.squeeze(scaled_Y), name=Y.name)

            self.y_scaler = y_scaler

        else:

            scaled_Y = Y


        self.features = X.columns

        return X, scaled_X, Y, scaled_Y


    def _load_csv(self):

        features = self.features

        features.append(self.target)
        features.append('collision_id')
        features.append('total_angular_momentum')


        if self.cascade and self.target == 'slr_mass':
            features.append('lr_mass')

        self.features = features


        tss_max = {
            '12D_LHS200': 200,
            '12D_LHS500': 500,
            '12D_LHS10K': 10000,
        }


        if self.tss < tss_max[self.lhs]:
            csv = '../datasets/{}_TSS_{:05d}.csv'.format(self.lhs, self.tss)
        else:
            csv = '../datasets/{}.csv'.format(self.lhs)


        try:

            raw_dataset = pd.read_csv(csv, usecols=self.features, sep=",")

        except FileNotFoundError:

            print('\n\tSubset does not exist yet! Creating new subset.')

            raw_dataset = self._create_subset()


        print('\n\tRaw dataset size: {}'.format(len(raw_dataset.mtotal)))


        raw_dataset[self.target] = pd.to_numeric(raw_dataset[self.target],
                errors='coerce')


        if self.filter_nan:

            raw_dataset = raw_dataset[raw_dataset[self.target].notna()]

            print('\tDataset w/o NaNs: {}'.format(
                  len(raw_dataset.mtotal)))


        if not self.use_null:

            raw_dataset = raw_dataset[raw_dataset[self.target] > 0]

            print('\tDataset w/o zero: {}'.format(len(raw_dataset.mtotal)))


        # Remove known crashed sim in test set
        raw_dataset = raw_dataset[raw_dataset['collision_id'] != "7bY6a5O4"]


        # Extract collision IDs, masks
        self.collision_ids = raw_dataset.pop('collision_id')
        self.J_tot         = raw_dataset.pop('total_angular_momentum')


        # Reset columns names
        self.features = raw_dataset.columns


        return raw_dataset.copy()


    def _create_subset(self):

        code_dir = Path(__file__).parent.absolute()

        csv_dir = "{}/datasets".format(Path(code_dir).parents[0])

        csv_file = '{}/{}.csv'.format(csv_dir, self.lhs)

        df = pd.read_csv(csv_file, sep=",")

        subset = df.sample(self.tss, random_state=42)

        new_file = '{}/{}_TSS_{:05d}.csv'.format(csv_dir, self.lhs, self.tss)

        subset.to_csv(new_file, sep=',', index=False)

        return subset[self.features]
