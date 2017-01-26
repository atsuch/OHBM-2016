# *- encoding: utf-8 -*-
# Author: Ben Cipollini
# License: BSD

import os
import os.path as op

import numpy as np
from matplotlib import pyplot as plt
from nilearn import datasets
from nilearn.image import iter_img
from nilearn.plotting import plot_stat_map

from nilearn_ext.utils import get_n_terms, get_percentile_val

import math


def nice_number(value, round_=False):
    """
    Convert a number to a print-ready value.

    nice_number(value, round_=False) -> float
    """
    exponent = math.floor(math.log(value, 10))
    fraction = value / 10 ** exponent

    if round_:
        if fraction < 1.5:
            nice_fraction = 1.
        elif fraction < 3.:
            nice_fraction = 2.
        elif fraction < 7.:
            nice_fraction = 5.
        else:
            nice_fraction = 10.
    else:
        if fraction <= 1:
            nice_fraction = 1.
        elif fraction <= 2:
            nice_fraction = 2.
        elif fraction <= 5:
            nice_fraction = 5.
        else:
            nice_fraction = 10.

    return nice_fraction * 10 ** exponent


def nice_bounds(axis_start, axis_end, num_ticks=8):
    """
    Returns tuple as (nice_axis_start, nice_axis_end, nice_tick_width)
    """
    axis_width = axis_end - axis_start
    if axis_width == 0:
        nice_tick_w = 0
    else:
        nice_range = nice_number(axis_width)
        nice_tick_w = nice_number(nice_range / (num_ticks - 1), round_=True)
        axis_start = math.floor(axis_start / nice_tick_w) * nice_tick_w
        axis_end = math.ceil(axis_end / nice_tick_w) * nice_tick_w

    nice_tick = np.arange(axis_start, axis_end, nice_tick_w)[1:]
    return axis_start, axis_end, nice_tick


def rescale(arr, val_range=(10, 200)):
    """Rescale array to the given range of numbers"""
    new_arr = ((float(val_range[1]) - val_range[0]) * (arr - arr.min()) /
               (arr.max() - arr.min())) + float(val_range[0])

    return new_arr


def save_and_close(out_path, fh=None):
    fh = fh or plt.gcf()
    if not op.exists(op.dirname(out_path)):
        os.makedirs(op.dirname(out_path))
    fh.savefig(out_path)
    plt.close(fh)


def _title_from_terms(terms, ic_idx, label=None, n_terms=4, sign=1):

    if terms is None:
        return '%s[%d]' % (label, ic_idx)

    # Use the n terms weighted most as a positive title, n terms
    # weighted least as a negative title and return both

    pos_terms = get_n_terms(terms, ic_idx, n_terms=n_terms, sign=sign)
    neg_terms = get_n_terms(terms, ic_idx, n_terms=n_terms, top_bottom="bottom",
                            sign=sign)

    title = '%s[%d]: POS(%s) \n NEG(%s)' % (
        label, ic_idx, ', '.join(pos_terms), ', '.join(neg_terms))

    return title


def plot_components(ica_image, hemi='', out_dir=None,
                    bg_img=datasets.load_mni152_template()):
    print("Plotting %s components..." % hemi)

    # Determine threshoold and vmax for all the plots
    thr = get_percentile_val(ica_image, percentile=90.0)
    vmax = get_percentile_val(ica_image, percentile=99.99)
    for ci, ic_img in enumerate(iter_img(ica_image)):

        title = _title_from_terms(terms=ica_image.terms, ic_idx=ci, label=hemi)
        fh = plt.figure(figsize=(14, 6))
        plot_stat_map(ic_img, axes=fh.gca(), threshold=thr, vmax=vmax,
                      colorbar=True, title=title, black_bg=True, bg_img=bg_img)

        # Save images instead of displaying
        if out_dir is not None:
            save_and_close(out_path=op.join(
                out_dir, '%s_component_%i.png' % (hemi, ci)))


def plot_components_summary(ica_image, hemi='', out_dir=None,
                            bg_img=datasets.load_mni152_template()):
    print("Plotting %s components summary..." % hemi)

    n_components = ica_image.get_data().shape[3]

    # Determine threshoold and vmax for all the plots
    thr = get_percentile_val(ica_image, percentile=90.0)
    vmax = get_percentile_val(ica_image, percentile=99.99)
    for ii, ic_img in enumerate(iter_img(ica_image)):

        ri = (ii / 5) % 5  # row i
        ci = ii % 5  # column i
        pi = ii % 25 + 1  # plot i
        fi = ii / 25  # figure i

        if ri == 0 and ci == 0:
            fh = plt.figure(figsize=(30, 20))
            print('Plot %03d of %d' % (fi + 1, np.ceil(n_components / 25.)))
        ax = fh.add_subplot(5, 5, pi)

        title = _title_from_terms(terms=ica_image.terms, ic_idx=ii, label=hemi)

        colorbar = ci == 4

        plot_stat_map(
            ic_img, axes=ax, threshold=thr, vmax=vmax, colorbar=colorbar,
            title=title, black_bg=True, bg_img=bg_img)

        if (ri == 4 and ci == 4) or ii == n_components - 1:
            out_path = op.join(
                out_dir, '%s_components_summary%02d.png' % (hemi, fi + 1))
            save_and_close(out_path)
