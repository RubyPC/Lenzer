#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 14:06:31 2023

@author: ruby
"""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def exporter():
    """Export utility modified from https://stackoverflow.com/a/41895194
    Returns export decorator, __all__ list
    """
    all_ = []

    def decorator(obj):
        all_.append(obj.__name__)
        return obj

    return decorator, all_


export, __all__ = exporter()
__all__.append("exporter")


@export
def image_grid(shape, pixel_width, x0=0, y0=0, edges=True):
    nx, ny = shape
    dx = nx * pixel_width
    dy = nx * pixel_width
    extra = 1 if edges else 0
    x = np.linspace(-dx / 2, dx / 2, nx + extra) + x0
    y = np.linspace(-dy / 2, dy / 2, ny + extra) + y0
    return x, y


@export
def plot_image(img, pixel_width, log_scale=True, label="HST F814W", colorbar=True,
               adjust_ax=True, vmin=None, vmax=None, **kwargs):

    # Set reasonable defaults
    if vmax is None:
        vmax = img.max()
    if vmin is None:
        if log_scale:
            vmin = vmax * 1e-3
        else:
            vmin = img.min()
    kwargs.setdefault(
        "norm",
        matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax)
        if log_scale
        else matplotlib.colors.Normalize(vmin=vmin, vmax=vmax),
    )
    kwargs.setdefault("cmap", "gray")

    # Plot image, note transposition
    plt.pcolormesh(
        *image_grid(img.shape, pixel_width), img.clip(vmin, None).T, **kwargs
    )
    if adjust_ax:
        plt.gca().set_aspect("equal")

    if colorbar:
        # Plot colorbar
        cbar = plt.colorbar(label=label, extend="both")
        cax = cbar.ax
        # Format colobar ticks as scalars (not 10^x)
        cax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
        if len(cbar.get_ticks()) < 2:
            # If there are very few colorbar ticks, show minor ticks too
            cax.yaxis.set_minor_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
            cax.tick_params(axis="y", labelsize=7, which="minor")

    plt.xlabel("[Arcseconds]")
    plt.ylabel("[Arcseconds]")


@export
def draw_box(x, y, **kwargs):
    """Draw rectangle, given x-y boundary tuples"""
    plt.gca().add_patch(
        matplotlib.patches.Rectangle(
            (x[0], y[0]), x[1] - x[0], y[1] - y[0], facecolor="none", **kwargs
        )
    )