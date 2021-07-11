#!/usr/bin/env python

import glob
import h5py
import itertools
import json
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import torch
import torchist
import tqdm

from scipy.ndimage import gaussian_filter
from scipy.stats import kstest
from sklearn.metrics import roc_curve, auc
from typing import Dict, List, Tuple, Union

import amsi


Array = np.ndarray
Scalar = Union[bool, int, float]
Vector = Union[List[Scalar], Array]


plt.rcParams.update({
    'axes.axisbelow': True,
    'axes.linewidth': 0.8,
    'figure.autolayout': True,
    'figure.figsize': (3.2, 2.4),
    'font.size': 12.,
    'legend.fontsize': 'x-small',
    'lines.linewidth': 0.8,
    'lines.markersize': 6.,
    'savefig.bbox': 'tight',
    'savefig.transparent': True,
    'xtick.labelsize': 'x-small',
    'xtick.major.width': 0.8,
    'ytick.labelsize': 'x-small',
    'ytick.major.width': 0.8,
})

if mpl.checkdep_usetex(True):
    plt.rcParams.update({
        'font.family': ['serif'],
        'font.serif': ['Computer Modern'],
        'text.usetex': True,
    })

#############
# Auxiliary #
#############

def translate(mask: str) -> str:
    try:
        return ', '.join(
            l for l, m in zip(simulator.labels, mask)
            if m == '1'
        )
    except:
        return mask

def match(patterns: Union[str, List[str]]) -> List[str]:
    if type(patterns) is str:
        patterns = [patterns]

    return [f for p in patterns for f in glob.glob(p, recursive=True)]


#########
# Plots #
#########

def loss_plot(dfs: List[pd.DataFrame]) -> mpl.figure.Figure:
    fig = plt.figure()

    for df in dfs:
        lines = plt.plot(df['epoch'], df['v_mean'])
        color = lines[-1].get_color()

        plt.plot(df['epoch'], df['mean'], color=color, linestyle='--', alpha=0.5)

    plt.grid()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    handles = [
        mpl.lines.Line2D([], [], color='k', label='validation'),
        mpl.lines.Line2D([], [], color='k', linestyle='--', label='training'),
    ]

    plt.legend(handles=handles)

    return fig


def error_plot(df: pd.DataFrame, legend: List[str], quantity: str = None) -> mpl.figure.Figure:
    df = df.groupby('mask', sort=False)
    mu, sigma = df.mean(), df.std()

    labels = [translate(mask) for mask in mu.index]
    mu, sigma = mu.to_numpy().T, sigma.to_numpy().T

    width, _ = plt.rcParams['figure.figsize']

    if mu.shape[0] > 2:
        fig = plt.figure(figsize=(width, width * len(labels) / 12))
        space = np.linspace(-.5, .5, mu.shape[0] + 2)[1:-1]
    else:
        fig = plt.figure(figsize=(width, width * len(labels) / 15))
        space = np.linspace(-.66, .66, mu.shape[0] + 2)[1:-1]

    for i, shift in enumerate(space):
        transform = mpl.transforms.Affine2D().translate(0., shift) + plt.gca().transData
        plt.errorbar(
            mu[i], labels,
            xerr=sigma[i],
            marker='o', markersize=2.5,
            linestyle='none', linewidth=1.,
            transform=transform,
        )

    plt.grid()
    plt.xlabel(quantity)
    plt.ylabel(None)
    plt.ylim(bottom=-1, top=len(labels))
    plt.legend(legend)

    return fig


def coverage_plot(df: pd.DataFrame) -> mpl.figure.Figure:
    width, _ = plt.rcParams['figure.figsize']
    fig = plt.figure(figsize=(width, width))

    for mask in df['mask'].unique():
        p = df['percentile'][df['mask'] == mask]
        p = np.hstack([0., np.sort(p), 1.])

        cdf = np.linspace(0., 1., len(p))

        plt.step(p, cdf, label=translate(mask))

    plt.plot([0, 1], [0, 1], color='k', ls='--')

    plt.grid()
    plt.xlabel(r'$p$')
    plt.ylabel(r'CDF$(p)$')
    plt.legend(loc='upper left')

    return fig


def consistency_plot(files: List[str], oom: int = -2) -> mpl.figure.Figure:
    dfs = []

    for f in files:
        df = pd.read_csv(f, dtype={0: str})
        df = df.set_index(df.columns[0])

        dfs.append(df.values)

    dfs = np.stack(dfs)
    labels = [translate(mask) for mask in df.columns]

    mu = dfs[0]
    mask = ~np.all(np.isnan(dfs), axis=0)
    mu[mask] = np.nanmean(dfs[:, mask], axis=0)

    cmap = plt.get_cmap('Reds').copy()
    cmap.set_bad((.9, .9, .9))

    rows, cols = mu.shape
    width, _ = plt.rcParams['figure.figsize']

    fig = plt.figure(figsize=(width * rows / 15,) * 2)

    ax = sns.heatmap(
        mu * (10 ** -oom),
        vmin=0., vmax=5.,
        cmap=cmap,
        annot=True, fmt='.1f', annot_kws={'fontsize': 6.},
        cbar=False,
        square=True,
        xticklabels=labels, yticklabels=labels,
    )

    at = mpl.offsetbox.AnchoredText(
        f'$\\times 10^{{{oom}}}$',
        loc='upper left',
        prop={'fontsize': plt.rcParams['legend.fontsize']},
        frameon=False,
    )
    ax.add_artist(at)

    for spine in ax.spines.values():
        spine.set_visible(True)

    return fig


def roc_plot(data: List[np.ndarray]) -> mpl.figure.Figure:
    width, _ = plt.rcParams['figure.figsize']
    fig = plt.figure(figsize=(width, width))

    for x in data:
        fp, tp, _ = roc_curve(x[:, 0], x[:, 1], sample_weight=x[:, 2])
        area = auc(fp, tp)

        plt.step(fp, tp, label='{:.3f}'.format(area))

    plt.plot([0, 1], [0, 1], color='k', linestyle='--')

    plt.axis('square')
    plt.grid()
    plt.xlabel('False Positive rate')
    plt.ylabel('True Positive rate')
    plt.legend(loc='upper left')

    return fig


##########
# Corner #
##########

def search_quantiles(x: Array, quantiles: Vector) -> Vector:
    x = np.sort(x, axis=None)[::-1]
    cdf = np.cumsum(x)
    idx = np.searchsorted(cdf, np.asarray(quantiles) * cdf[-1])

    return x[idx]


class AlphaLinearColormap(mpl.colors.Colormap):
    def __new__(self, color: str, alpha: float = 1.):
        rgb = mpl.colors.to_rgb(color)

        return mpl.colors.LinearSegmentedColormap.from_list(
            color + '_v',
            (rgb + (0.,), rgb + (alpha,))
        )


class NonLinearColormap(mpl.colors.Colormap):
    def __init__(self, cmap: mpl.colors.Colormap, levels: Vector):
        super().__init__(name=cmap.name + '_nl')

        self.cmap = cmap

        levels = np.asarray(levels)

        self.dom = (levels - levels.min()) / (levels.max() - levels.min())
        self.img = np.linspace(0., 1., len(levels))

    def __getattr__(self, attr: str):
        return getattr(self.cmap, attr)

    def __call__(self, x: np.ndarray, alpha: float = 1.0, **kwargs) -> np.ndarray:
        y = np.interp(x, self.dom, self.img)
        return self.cmap(y, alpha)


def corner(
    data: List[Dict[Tuple[int, int], Array]],
    dims: List[int],
    low: Vector,
    high: Vector,
    quantiles: Vector = [.6827, .9545, .9973],
    labels: List[str] = None,
    legend: List[str] = None,
    hide_legend: bool = False,
    star: Vector = None,
    size: float = 6.4,
) -> mpl.figure.Figure:
    D = len(dims)

    fig, axs = plt.subplots(
        D, D,
        squeeze=False,
        figsize=(size, size),
        sharex='col',
        gridspec_kw={'wspace': 0., 'hspace': 0.},
    )

    quantiles = np.sort(np.asarray(quantiles))
    quantiles = np.append(quantiles[::-1], 0.)

    colors = [c for _, c in zip(data, mpl.colors.TABLEAU_COLORS)]

    for i in range(D):
        for j in range(D):
            ax = axs[i, j]

            # Only lower triangle
            if j > i:
                ax.axis('off')
                continue

            # Data
            a, b = dims[i], dims[j]

            for color, hists in zip(colors, data):
                if (a, b) in hists:
                    hist = hists[(a, b)]
                else:
                    continue

                x = np.linspace(low[b], high[b], hist.shape[-1] + 1)
                y = np.linspace(low[a], high[a], hist.shape[0] + 1)
                x = x[1:] - (x[1] - x[0]) / 2
                y = y[1:] - (y[1] - y[0]) / 2

                ## Draw
                if i == j:
                    hist = hist / (x[1] - x[0])
                    ax.step(x, hist, where='mid', color=color)

                    _, ymax = ax.get_ylim()
                    ymax = max(ymax, hist.max() * 1.0625)

                    ax.set_xlim(left=low[a], right=high[a])
                    ax.set_ylim(bottom=0., top=ymax)
                else:
                    levels = search_quantiles(hist, quantiles)
                    levels = np.unique(levels)

                    cf = ax.contourf(
                        x, y, hist,
                        levels=levels,
                        cmap=NonLinearColormap(AlphaLinearColormap(color, 0.5), levels),
                    )
                    ax.contour(cf, colors=color)

                    if j > 0:
                        ax.sharey(axs[i, j - 1])
                    else:
                        ax.set_ylim(bottom=low[a], top=high[a])

            # Ticks
            if i == D - 1:
                ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(3, prune='both'))
                plt.setp(
                    ax.get_xticklabels(),
                    rotation=45.,
                    horizontalalignment='right',
                    rotation_mode='anchor',
                )
            else:
                ax.xaxis.set_ticks_position('none')

            if i == j:
                ax.set_yticks([])
            elif j == 0:
                ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(3, prune='both'))
            else:
                ax.yaxis.set_ticks_position('none')

            # Labels
            if labels is not None:
                if i == D - 1:
                    ax.set_xlabel(labels[b])

                if j == 0 and i != j:
                    ax.set_ylabel(labels[a])

            ax.label_outer()

            # theta*
            if star is not None:
                if i != j:
                    ax.plot(
                        star[b], star[a],
                        color='k',
                        marker='*',
                    )
                else:
                    ax.axvline(
                        star[a],
                        color='k',
                        linestyle='--',
                    )

    # Legend
    if star is not None:
        handles = [mpl.lines.Line2D([], [], color='k', marker='*', linestyle='None', label=r'$\theta^*$')]
    else:
        handles = []

    cmap = AlphaLinearColormap('black', 0.5)

    handles += [
        mpl.patches.Patch(color=cmap(1 - (p + q) / 2), label=r'{:.1f}\,\%'.format(p * 100))
        for p, q in zip(quantiles[:-1], quantiles[1:])
    ]

    if legend is not None:
        handles += [
            mpl.lines.Line2D([], [], color=color, label=label)
            for color, label in zip(colors, legend)
            if label is not None
        ]

    if not hide_legend:
        anc = (size - 0.1) / size
        fig.legend(handles=handles, loc='upper right', bbox_to_anchor=(anc, anc), frameon=False)

    return fig


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Graphical results')

    parser.add_argument('type', choices=['loss', 'error', 'coverage', 'consistency', 'roc', 'corner'])
    parser.add_argument('input', nargs='+', help='input file(s) (CSV or JSON)')
    parser.add_argument('-o', '--output', default='products/plots/out.pdf', help='output file (PDF)')

    parser.add_argument('-simulator', default=None, choices=['SLCP', 'MLCP', 'GW', 'HH'])

    args = parser.parse_args()

    # Output
    if os.path.dirname(args.output):
        os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Simulator
    SIMULATORS = {
        'SLCP': amsi.SLCP,
        'MLCP': amsi.MLCP,
        'GW': amsi.GW,
        'HH': amsi.HH,
    }

    if args.simulator is None:
        file = args.input[0].upper()

        for key in SIMULATORS:
            if key in file:
                args.simulator = key
                break

    simulator = SIMULATORS[args.simulator]()

    # Loss plots
    if args.type == 'loss':
        dfs = [
            pd.read_csv(f)
            for f in args.input
        ]

        loss_plot(dfs)

        plt.savefig(args.output)
        plt.close()

    # Error plots
    if args.type == 'error':
        keys = []
        dfs = []

        with open(args.input[0]) as f:
            for key, files in json.load(f).items():
                dfss = [
                    pd.read_csv(f, dtype={'mask': str})
                    for f in match(files)
                ]

                keys.append(key)
                dfs.append(pd.concat(dfss, ignore_index=True))

        for metric in ['probability', 'entropy', 'distance']:
            if metric == 'probability':
                err = dfs[0][['mask', 'total_probability']]

                for i, df in enumerate(dfs[1:]):
                    err = err.join(df['total_probability'], rsuffix=f'_{i}')

                error_plot(err, keys, 'Probability')
            elif metric == 'entropy':
                if 'entropy_truth' in dfs[0].columns:
                    err = dfs[0][['mask', 'entropy_truth', 'entropy']]
                    legend = ['GT (MCMC)'] + keys
                else:
                    err = dfs[0][['mask', 'entropy']]
                    legend = keys

                for i, df in enumerate(dfs[1:]):
                    err = err.join(df['entropy'], rsuffix=f'_{i}')

                error_plot(err, legend, 'Entropy')
            elif metric == 'distance':
                if 'wd_truth' in dfs[0].columns:
                     err = dfs[0][['mask', 'wd_truth']]
                else:
                    continue

                for i, df in enumerate(dfs[1:]):
                    err = err.join(df['wd_truth'], rsuffix=f'_{i}')

                error_plot(err, keys, r'$W_d$ to GT')

            plt.savefig(args.output.replace('.pdf', f'_{metric}.pdf'))
            plt.close()

    # Coverage plots
    if args.type == 'coverage':
        dfs = [
            pd.read_csv(f, usecols=['mask', 'percentile'], dtype={'mask': str})
            for f in args.input
        ]
        df = pd.concat(dfs, ignore_index=True)

        coverage_plot(df)

        plt.savefig(args.output)
        plt.close()

    # Consistency plots
    if args.type == 'consistency':
        consistency_plot(args.input)

        plt.savefig(args.output)
        plt.close()

    # ROC plots
    if args.type == 'roc':
        data = {}

        for filename in args.input:
            with h5py.File(filename) as f:
                for mask, val in f.items():
                    val = val[...]

                    if mask in data:
                        data[mask].append(val)
                    else:
                        data[mask] = [val]

        for mask, seq in data.items():
            roc_plot(seq)

            plt.savefig(args.output.replace('.pdf', f'_{mask}.pdf'))
            plt.close()

    # Corner plots
    if args.type == 'corner':
        ## Settings
        with open(args.input[0]) as f:
            settings = json.load(f)

        ## Simulator
        labels = simulator.labels
        low = simulator.low.numpy()
        high = simulator.high.numpy()

        ## Histograms
        data = []

        for item in settings['items']:
            data.append({})

            for f in match(item['files']):
                mask, hist = torch.load(f)
                hist._coalesced_(True)
                dims = mask.nonzero().squeeze(-1).tolist()

                pairs = {}

                if item.get('marginalize', False):
                    for i, a in reversed(list(enumerate(dims))):
                        for j, b in enumerate(dims[:i + 1]):
                            h = torchist.marginalize(hist, dim=[i, j], keep=True)

                            if h.is_sparse:
                                h = h.to_dense()

                            pairs[(a, b)] = h

                        hist = torchist.marginalize(hist, dim=i)
                else:
                    if len(dims) == 1:
                        i = j = dims[0]
                    elif len(dims) == 2:
                        j, i = dims
                    else:
                        continue

                    pairs[(i, j)] = hist

                for (i, j), h in pairs.items():
                    h = h.t().cpu().numpy()

                    if 'smooth' in item:
                        h = gaussian_filter(h, item['smooth'])

                    data[-1][(i, j)] = h + data[-1].get((i, j), 0)

            for (i, j), h in data[-1].items():
                h /= h.sum()

        ## Legend
        legend = settings.get('legend', None)

        ## Star
        if 'star' in settings:
            samples, index = settings['star']

            with h5py.File(samples) as f:
                star = f['theta'][index]
        else:
            star = None

        ## Masks
        if type(settings['masks']) is list:
            settings['masks'] = {mask: {} for mask in settings['masks']}

        for mask, kwargs in settings['masks'].items():
            dims = amsi.str2mask(mask).nonzero().squeeze(-1).tolist()

            corner(
                data, dims, low, high,
                labels=labels, legend=legend,
                star=star, **kwargs
            )

            plt.savefig(args.output.replace('.pdf', f'_{mask}.pdf'))
            plt.close()
