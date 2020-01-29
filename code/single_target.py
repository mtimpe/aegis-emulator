#!/usr/bin/env python3
# coding: utf-8
import numpy as np
np.random.seed(42)

import argparse
from dataset import Dataset
import joblib
from os import makedirs
from os.path import exists, isfile
import pandas as pd
from pathlib import Path
import pickle
from plots import correlations
from sys import exit


__author__ = "Miles Timpe, Maria Han Veiga, and Mischa Knabenhans"
__maintainer__ = "Miles Timpe"
__credits__ = ["Miles Timpe", "Maria Han Veiga", "Mischa Knabenhans"]
__email__ = "mtimpe@physik.uzh.ch"
__copyright__ = "Copyright 2020, ICS"
__license__ = "GNU General Public License v3.0"
__version__ = "1.0.0"
__status__ = "Production"


def make_dir_struct(root, method, target, tss):
    """Setup the output directory.

    Args:
        root (str):   Root of output directory structure.
        method (str): Regression method (e.g., MLP).
        target (str): Targeted post-impact property.
        tss (int):    Training set size.

    Returns:
        str: Path to metric file.

    """

    if root:

        output_dir = "{}/{}/{}/{}".format(root, method, target, tss)

    else:

        code_dir = Path(__file__).parent.absolute()

        models_dir = "{}/models/regressors".format(Path(code_dir).parents[0])

        output_dir = "{}/{}/{}/{}".format(models_dir, target, method, tss)

    hpo_dir = "{}/hpo".format(output_dir)

    best_dir = "{}/best".format(output_dir)


    for folder in [hpo_dir, best_dir]:
        if not exists(folder):
            makedirs(folder)


    metrics_file = "{}/metrics.csv".format(output_dir)

    return(output_dir, metrics_file)



def train(method, target, tss, hpo_evals, kfolds, scaler, use_null, root):
    """Train single-target [method] regressor for [target].

    Args:
        method:    Regression method: GP, MLP, or XGB.
        target:    Post-impact property to regress.
        tss:       Training set size.
        hpo_evals: Number of optimization steps.
        kfolds:    Number of folds in cross-validation.
        scaler:    Standardization method.
        use_null:  Train with zero values?


    Returns:
        r2_score.  Coefficient of determination.
    """


    # Pre-impact features to use in training and prediction
    raw_features = [
        'mtotal','gamma', 'b_inf','v_inf',
        'targ_core', 'targ_omega_norm', 'targ_theta', 'targ_phi',
        'proj_core', 'proj_omega_norm', 'proj_theta', 'proj_phi'
    ]

    # Validation set size
    vss = 500


    welcome(use_null)

    output_folder, metrics_file = make_dir_struct(root, method, target, tss)


    print('\nLoading training dataset (LHS10K)')

    ds = Dataset("12D_LHS10K", tss, raw_features, target,
                 scaling_method=scaler, use_null=use_null)


    # Save data scalers
    joblib.dump(ds.x_scaler, "{}/x_scaler.save".format(output_folder))
    joblib.dump(ds.y_scaler, "{}/y_scaler.save".format(output_folder))


    # Initialize untrained emulator
    if method == 'gp':
        from gp import GPRegressor
        emu = GPRegressor(ds, output_folder)
    elif method == 'mlp':
        from mlp import MLPRegressor
        emu = MLPRegressor(ds, output_folder)
    elif method == 'xgb':
        from xgb import XGBoostRegressor
        emu = XGBoostRegressor(ds, output_folder)
    else:
        exit('Method not available!')



    if (tss > 2000) and (method == "gp"):

        # Check if model exists for GP with TSS=2000
        gp_pkl = "{}/{}/gp/2000/best/best.pkl".format(output_root, target)

        if isfile(gp_pkl):

            emu.best_model = gp_pkl

    else:

        # Regressor hyperparameter optimization
        print('\n\nSearching for optimal regressor architecture')
        print('\n\tUsing 5-fold cross-validation with an 80/20 split\n')
        print('\tTraining set   (N={})'.format(int(len(ds.collision_ids)*0.8)))
        print('\tValidation set (N={})'.format(int(len(ds.collision_ids)*0.2)))


        hpo_metrics = emu.hpo(hpo_evals)


        print('\n\tBest mean regressor performance during HPO:')
        report_metrics(hpo_metrics)

        append_metrics('hpo', hpo_metrics, metrics_file)



    # Load test data
    print('\n\n\nLoading test data')

    dt = Dataset("12D_LHS500", vss, raw_features, target,
                 scaling_method=scaler, use_null=False,
                 external_x_scaler=ds.x_scaler,
                 external_y_scaler=ds.y_scaler)


    print('\n\nEvaluating assuming perfect nan classification:')

    print('\n\tx: {}'.format(len(dt.scaled_x)))
    print('\ty: {}'.format(len(dt.scaled_y)))

    _, y_true_nonan, y_pred_nonan = emu.evaluate_best(dt)

    metrics_nonan = emu._regression_metrics(y_true_nonan, y_pred_nonan)

    report_metrics(metrics_nonan)
    append_metrics('nonan', metrics_nonan, metrics_file)


    results_file = '{}/results_nonan.csv'.format(output_folder)

    save_results(dt.x, y_true_nonan, y_pred_nonan, results_file,
                 dt.collision_ids, target, method, tss, scaler, hpo_evals,
                 use_null)


    figure_name = '{}/correlation_nonan.png'.format(output_folder)

    correlations(dt.x, y_true_nonan, y_pred_nonan, figure_name, metrics_nonan)


    ############################################################################

    print('\n\nEvaluating model assuming perfect null classification:')

    ids_pzc, x_input, m_tot, j_tot, y_true_nonan_nonull, y_pred_nonan_nonull = perfect_null_classifier(
            dt.collision_ids, dt.x.values, dt.x.mtotal, dt.J_tot,
            y_true_nonan, y_pred_nonan, target)

    x_input = pd.DataFrame(x_input, columns=dt.x.columns)

    dx_pzc = pd.DataFrame(x_input, columns=dt.x.columns)

    dx_pzc['collision_id'] = ids_pzc

    # Evaluate regressor with pre-classification
    metrics_nonan_nonull = emu._regression_metrics(y_true_nonan_nonull,
                                                   y_pred_nonan_nonull)

    print('\n\tx: {}'.format(len(x_input)))
    print('\ty: {}'.format(len(y_true_nonan_nonull)))

    report_metrics(metrics_nonan_nonull)
    append_metrics('nonan_nonull', metrics_nonan_nonull, metrics_file)


    results_file = '{}/results_pnc_pzc.csv'.format(output_folder)

    save_results(dx_pzc, y_true_nonan_nonull, y_pred_nonan_nonull, results_file,
                 ids_pzc, target, method, tss, scaler, hpo_evals, use_null)


    figure_name = '{}/correlation_nonan_nonull.png'.format(output_folder)

    correlations(x_input, y_true_nonan_nonull, y_pred_nonan_nonull, figure_name,
                 metrics_nonan_nonull)


    ############################################################################

    print('\n\nEvaluating model with physics enforcement:')

    y_pred_phys = enforce_physics(y_true_nonan_nonull, y_pred_nonan_nonull,
                                  m_tot, j_tot, target)

    print('\n\tx:      {}'.format(len(m_tot)))
    print('\ty_true: {}'.format(len(y_true_nonan_nonull)))
    print('\ty_pred: {}'.format(len(y_pred_nonan_nonull)))
    print('\ty_phys: {}'.format(len(y_pred_phys)))

    # Evaluate regressor with physics enforcement
    metrics_phys = emu._regression_metrics(y_true_nonan_nonull, y_pred_phys)

    report_metrics(metrics_phys)
    append_metrics('physics', metrics_phys, metrics_file)


    results_file = '{}/results_nonan_nonull_phys.csv'.format(output_folder)

    save_results(dt.x, y_true_nonan, y_pred_nonan, results_file,
                 dt.collision_ids, target, method, tss, scaler, hpo_evals,
                 use_null)


    figure_name = '{}/correlation_nonan_nonull_phys.png'.format(output_folder)

    correlations(x_input, y_true_nonan, y_pred_phys, figure_name, metrics_phys)



def append_metrics(stage, metrics, filepath):

    if not isfile(filepath):
        with open(filepath, 'w') as f:
            f.write('stage,vss,r2,rmse,mae,mbe\n')

    with open(filepath, 'a') as f:
        f.write('{},{},{},{},{},{}\n'.format(stage, metrics['vss'],
                metrics['r2'], metrics['rmse'], metrics['mae'], metrics['mbe']))


def enforce_physics(y_true, y_pred, m_tot, j_tot, target):

    if target in ['lr_mass', 'slr_mass', 'debris_mass']:

        y_max = m_tot

    elif target in ['lr_mass_norm', 'slr_mass_norm', 'debris_mass_norm',
                    'lr_core', 'slr_core', 'debris_iron', 'lr_mixing_ratio',
                    'slr_mixing_ratio', 'debris_mixing_ratio',
                    'lr_angular_momentum_norm', 'slr_angular_momentum_norm',
                    'debris_angular_momentum_norm', 'lr_condensed', 'lr_liquid',
                    'lr_vapor', 'lr_intermediate', 'slr_condensed',
                    'slr_liquid', 'slr_vapor', 'slr_intermediate',
                    'debris_condensed', 'debris_liquid', 'debris_vapor',
                    'debris_intermediate']:

        y_max = [1] * len(y_pred)

    elif target in ['lr_theta', 'slr_theta']:

        y_max = [180] * len(y_pred)

    elif target in ['lr_angular_momentum', 'slr_angular_momentum',
                    'debris_angular_momentum']:

        y_max = j_tot

    else:

        y_max = [1e+99] * len(y_pred)


    y_new = []


    n_enf = 0

    for yp, yt, ym in zip(y_pred, y_true, y_max):

        if yp < 0:
            yp = 0
            n_enf += 1

        if yp > ym:
            yp = ym
            n_enf += 1

        y_new.append(yp)

    print("\n\tEnforcements: {}".format(n_enf))

    return y_new


def perfect_null_classifier(collision_ids, x_vec, m_tot, j_tot, y_test, y_pred,
                            target):

    # Post-hoc zero classification
    new_ids  = []
    new_x    = []
    new_m    = []
    new_j    = []
    new_true = []
    new_pred = []

    z = 0

    for cid, x, m, j, yt, yp in zip(collision_ids, x_vec, m_tot, j_tot,
                                    y_test, y_pred):

        if yt == 0:
            yp = 0
            z += 1

        new_ids.append(cid)
        new_x.append(x)
        new_m.append(m)
        new_j.append(j)
        new_true.append(yt)
        new_pred.append(yp)

    print('\n\tZeros: {}'.format(z))

    return new_ids, new_x, new_m, new_j, new_true, new_pred


def trained_nan_classifier(collision_ids, x_vec, y_test, y_pred, lr_zero,
                           slr_zero, target):

    # Post-hoc zero classification
    new_ids  = []
    new_x    = []
    new_true = []
    new_pred = []


    for cid, x, yt, yp, lr, slr in zip(collision_ids, x_vec, y_test, y_pred,
                                       lr_zero, slr_zero):

        if ('lr_' in target) and ('slr_' not in target):
            if lr == 1:
                continue

        if 'slr_' in target:
            if (lr == 1) or (slr == 1):
                continue

        new_ids.append(cid)
        new_x.append(x)
        new_true.append(yt)
        new_pred.append(yp)

    return new_ids, new_x, new_true, new_pred



def welcome(use_null):
    """Print short welcome message.

    Args:
        use_null (bool): Train with or without zero values.

    """

    print('\n\nWelcome to A E G I S, version1')
    print('Advanced Emulation for Giant Impact Simulations')
    print('Miles Timpe, Maria Han Veiga, & Mischa Knabenhans')
    print('27 November, 2020, Zurich, Switzerland\n')

    if not use_null:
        print('\nTraining regressor without zero values.')


def report_metrics(metrics):

    print('\n\tNaNs:     {}'.format(metrics['nan']))
    print('\tr2-score: {:.4f}'.format(metrics['r2']))
    print('\tMBE:      {:.2e}'.format(metrics['mbe']))
    print('\tMAE:      {:.2e}'.format(metrics['mae']))
    print('\tRMSE:     {:.2e}'.format(metrics['rmse']))



def save_results(data, y_true, y_pred, filename, collision_ids, target, method,
                 tss, scaler, hpo_evals, use_null):

    df = data.copy()

    df['collision_id'] = collision_ids
    df['y_true'] = y_true
    df['y_pred'] = y_pred

    with open(filename, 'w') as f:

        f.write('{}\n'.format('#'*40))
        f.write('# Target:      {}\n'.format(target))
        f.write('# Method:      {}\n'.format(method))
        f.write('# TSS:         {}\n'.format(tss))
        f.write('# Scaler:      {}\n'.format(scaler))
        f.write('#\n# HPO algo:    {}\n'.format('TPE'))
        f.write('# HPO steps:   {}\n'.format(hpo_evals))
        f.write('#\n# Training zeros:   {}\n'.format(use_null))
        f.write('#\n# NaN  classifier:  {}\n'.format('Perfect'))
        f.write('# Zero classifier:  {}\n'.format('None'))
        f.write('# Physics:          {}\n#\n'.format('Off'))
        f.write('{}\n'.format('#'*40))


    with open(filename, 'a') as f:

        df.to_csv(f, index=False)


if __name__ == "__main__":

    tss_max = {
        '12D_LHS200': 200,
        '12D_LHS500': 500,
        '12D_LHS10K': 10000,
    }


    parser = argparse.ArgumentParser()

    parser.add_argument("method",
            type=str, help="Emulation method (e.g., MLP).")

    parser.add_argument("target",
            type=str, help="Parameter to emulate (e.g., lr_mass).")

    parser.add_argument("-tss", "--training_set_size",
            type=int, default=10000,
            help="Training set size (TSS).")

    parser.add_argument("-hs", "--hpo_steps",
            type=int, default=100,
            help="Number of HPO steps.")

    parser.add_argument("-k", "--k_folds",
            type=int, default=5,
            help="Number of cross-validation folds during HPO.")

    parser.add_argument("-s", "--scaling_method",
            type=str, default='standard',
            help="Scaling method to use on input data (e.g., StandardScaler).")

    parser.add_argument('--use_null',
            dest='use_null', action='store_true',
            help='Use null values when training regressor?')
    parser.set_defaults(use_null=True)

    parser.add_argument('-out', '--output_root',
            type=str, default=None,
            help='Root of output folder.')

    args = parser.parse_args()


    if args.training_set_size > tss_max["12D_LHS10K"]:
        args.training_set_size = tss_max["12D_LHS10K"]


    if args.method not in ['gp', 'xgb', 'mlp']:
        exit('Selected emulation method not available!')


    if args.scaling_method not in ['none', 'minmax', 'robust', 'standard']:
        exit('Selected scaling method not available!')


    # Train regressor
    train(args.method, args.target, args.training_set_size,
          args.hpo_steps, args.k_folds, args.scaling_method, args.use_null,
          args.output_root)
