#!/usr/bin/env python

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch

from torchist import *
from typing import List, Union


Scalar = Union[int, float]
ArrayLike = Union[Scalar, List[Scalar], np.ndarray, torch.Tensor]


plt.rcParams['axes.axisbelow'] = True
plt.rcParams['axes.grid'] = True
plt.rcParams['figure.autolayout'] = True
plt.rcParams['font.size'] = 12.
plt.rcParams['legend.fontsize'] = 'small'
plt.rcParams['savefig.transparent'] = True


def get_pairs(hist: torch.Tensor) -> List[List[torch.Tensor]]:
    r"""Get pairwise histograms"""

    hists = []

    for i in reversed(range(hist.dim())):
        hists.insert(0, [])

        for j in range(i + 1):
            h = marginalize(hist, dim=[i, j], keep=True)
            if h.is_sparse:
                h = h.to_dense()

            hists[0].append(h.cpu())

        hist = marginalize(hist, dim=i)

    return hists


def corner(
    hists: List[List[ArrayLike]],
    low: ArrayLike,
    high: ArrayLike,
    truth: ArrayLike = None,
    percentiles: ArrayLike = [.1974, .3829, .6827, .8664, .9545, .9973],
    labels: List[str] = [],
    **fig_kwargs,
) -> mpl.figure.Figure:
    r"""Pairwise corner plot"""

    D = len(hists)

    fig_kwargs.setdefault('figsize', (D * 4.8,) * 2)
    fig, axs = plt.subplots(D, D, squeeze=False, **fig_kwargs)

    percentiles = np.sort(np.asarray(percentiles))
    percentiles = np.append(percentiles[::-1], 0.)

    for i in range(D):
        for j in range(D):
            # Only lower triangle
            if j > i:
                axs[i, j].axis('off')
                continue

            # Data
            ax = axs[i, j]
            hist = np.asarray(hists[i][j]).T
            x = np.linspace(low[j], high[j], hist.shape[-1])
            y = np.linspace(low[i], high[i], hist.shape[0])

            # Draw
            if i == j:
                ax.step(x, hist, color='k', linewidth=1.)
            else:
                levels = coverage(hist, percentiles)

                cf = ax.contourf(
                    x, y, hist,
                    levels=levels,
                    cmap=NonLinearColormap('Blues', levels),
                    alpha=0.8,
                )
                ax.contour(cf, colors='k', linewidths=1.)

                if i > 0:
                    ax.sharex(axs[i - 1, j])

                if j > 0:
                    ax.sharey(axs[i, j - 1])

            ax.label_outer()
            ax.set_box_aspect(1.)

            # Labels
            if labels:
                if i == D - 1:
                    ax.set_xlabel(labels[j])

                if j == 0 and i != j:
                    ax.set_ylabel(labels[i])

            # Truth
            if truth is not None:
                ax.axvline(truth[j], color='darkorange', linewidth=1.5)

                if i != j:
                    ax.axhline(truth[i], color='darkorange', linewidth=1.5)
                    ax.plot(truth[j], truth[i], color='darkorange', marker='s', markersize=3.)

    return fig


def coverage(x: np.ndarray, percentiles: ArrayLike) -> np.ndarray:
    r"""Coverage percentiles"""

    x = np.sort(x, axis=None)[::-1]
    cdf = np.cumsum(x)
    idx = np.searchsorted(cdf, np.asarray(percentiles) * cdf[-1])

    return x[idx]


class NonLinearColormap(mpl.colors.LinearSegmentedColormap):
    r"""Non-linear colormap"""

    def __init__(self, cmap: str, levels: np.ndarray):
        self.cmap = plt.get_cmap(cmap)

        self.dom = (levels - levels.min()) / (levels.max() - levels.min())
        self.img = np.linspace(0., 1., len(levels))

    def __getattr__(self, attr: str):
        return getattr(self.cmap, attr)

    def __call__(self, x: np.ndarray, alpha: float = 1.0, **kwargs) -> np.ndarray:
        y = np.interp(x, self.dom, self.img)
        return self.cmap(y, alpha)
