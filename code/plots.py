# coding: utf-8
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def correlations(df, y_true, y_pred, figname, metrics):


    color_by = df["gamma"].values


    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    test_colors = np.array(color_by)

    new_labels = []
    new_preds = []
    new_colors = []

    for tl, tp, tc in zip(y_true, y_pred, test_colors):
        if not np.isnan(tl):
            new_labels.append(tl)
            new_preds.append(tp)
            new_colors.append(tc)


    y_true = np.array(new_labels)
    y_pred = np.array(new_preds)
    test_colors = np.array(new_colors)


    vmin = min(min(y_true), min(y_pred))
    vmax = max(max(y_true), max(y_pred))

    error = y_pred - y_true

    xmax = max(min(test_colors), max(test_colors))

    fig, axes = plt.subplots(1, 2, figsize=(8,4), facecolor='white')

    ax = axes[0]

    ax.scatter(y_true, y_pred,
               c='blue', edgecolor='none', #cmap=plt.cm.magma,
               vmin=-xmax, vmax=xmax, alpha=0.5)


    ax.set_xlabel(r'Simulated Values')
    ax.set_ylabel(r'Predicted Values')
    ax.axis('equal')
    ax.axis('square')
    ax.set_xlim([vmin,vmax])
    ax.set_ylim([vmin,vmax])
    ax.plot([vmin, vmax], [vmin, vmax], color='darkorange')

    vspan = abs(vmax - vmin)

    vtop  = vmax - 0.1 * vspan
    vleft = vmin + 0.1 * vspan
    vline = vspan * 0.05

    ax.text(vleft, vtop-3*vline, '{:<4} = {:.4e}'.format('r2', metrics['r2']))
    ax.text(vleft, vtop-4*vline, '{:<4} = {:.2e}'.format('rmse', metrics['rmse']))
    ax.text(vleft, vtop-5*vline, '{:<4} = {:.2e}'.format('mae', metrics['mae']))
    ax.text(vleft, vtop-6*vline, '{:<4} = {:.2e}'.format('mbe', metrics['mbe']))


    ax = axes[1]
    sns.distplot(error, ax=ax)
    ax.set_xlabel("Prediction Error")
    ax.set_ylabel("Count")

    plt.tight_layout()

    plt.savefig(figname, format='png', dpi=300)

    plt.close()
