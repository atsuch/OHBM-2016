# *- encoding: utf-8 -*-
# Author: Ben Cipollini
# License: BSD

import os
import os.path as op

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from nilearn import datasets
from nilearn.image import iter_img, index_img, math_img
from nilearn.plotting import plot_stat_map
from scipy import stats

from nilearn_ext.utils import reorder_mat, get_n_terms, get_match_idx_pair
from nilearn_ext.radar import radar_factory

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
    # get nonzero part of the image for proper thresholding of
    # r- or l- only component
    nonzero_img = ica_image.get_data()[np.nonzero(ica_image.get_data())]
    thr = stats.scoreatpercentile(np.abs(nonzero_img), 90)
    vmax = stats.scoreatpercentile(np.abs(nonzero_img), 99.99)
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
    # get nonzero part of the image for proper thresholding of
    # r- or l- only component
    nonzero_img = ica_image.get_data()[np.nonzero(ica_image.get_data())]
    thr = stats.scoreatpercentile(np.abs(nonzero_img), 90)
    vmax = stats.scoreatpercentile(np.abs(nonzero_img), 99.99)
    for ii, ic_img in enumerate(iter_img(ica_image)):

        ri = ii % 5  # row i
        ci = (ii / 5) % 5  # column i
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


def plot_matched_components(images, labels, score_mat, sign_mat,
                            force=False, out_dir=None):
    """
    Uses the score_mat to match up two images. If force, one-to-one matching
    is forced.
    Sign_mat is used to flip signs when comparing two images.
    """
    # Be careful
    assert len(images) == 2
    assert len(labels) == 2
    assert images[0].shape == images[1].shape
    n_components = images[0].shape[3]  # values @ 0 and 1 are the same
    assert score_mat.shape == sign_mat.shape
    assert len(score_mat[0]) == n_components

    # Get indices for matching components
    match, unmatch = get_match_idx_pair(score_mat, sign_mat, force=force)
    idx_pair = match["idx"]
    sign_pair = match["sign"]

    if not force and unmatch["idx"] is not None:
        idx_pair = np.hstack((idx_pair, unmatch["idx"]))
        sign_pair = np.hstack((sign_pair, unmatch["sign"]))

    n_comp = len(idx_pair[0])   # number of comparisons

    # Calculate a vmax optimal across all the plots
    # get nonzero part of the image for proper thresholding of
    # r- or l- only component
    nonzero_imgs = [img.get_data()[np.nonzero(img.get_data())]
                    for img in images]
    dat = np.append(nonzero_imgs[0], nonzero_imgs[1])
    vmax = stats.scoreatpercentile(np.abs(dat), 99.99)

    print("Plotting results.")
    for i in range(n_comp):
        c1i, c2i = idx_pair[0][i], idx_pair[1][i]
        cis = [c1i, c2i]

        prefix = "unmatched-" if i >= n_components else ""
        num = i - n_components if i >= n_components else i
        png_name = '%s%s_%s_%s.png' % (prefix, labels[0], labels[1], num)
        print "plotting %s" % png_name

        comp_imgs = [index_img(img, ci) for img, ci in zip(images, cis)]

        # flip the sign if sign_mat for the corresponding comparison is -1
        signs = [sign_pair[0][i], sign_pair[1][i]]
        comp_imgs = [math_img("%d*img" % (sign), img=img)
                     for sign, img in zip(signs, comp_imgs)]

        if ('R' in labels and 'L' in labels):
            # Combine left and right image, show just one.
            # terms are not combined here
            comp = math_img("img1+img2", img1=comp_imgs[0], img2=comp_imgs[1])
            titles = [_title_from_terms(
                terms=comp_imgs[labels.index(hemi)].terms,
                ic_idx=cis[labels.index(hemi)], label=hemi,
                sign=signs[labels.index(hemi)]) for hemi in labels]
            fh = plt.figure(figsize=(14, 8))
            plot_stat_map(
                comp, axes=fh.gca(), title="\n".join(titles), black_bg=True,
                symmetric_cbar=True, vmax=vmax)

        else:
            # Show two images, one above the other.
            fh = plt.figure(figsize=(14, 12))

            for ii in [0, 1]:  # Subplot per image
                ax = fh.add_subplot(2, 1, ii + 1)
                comp = comp_imgs[ii]

                title = _title_from_terms(
                    terms=images[ii].terms, ic_idx=cis[ii],
                    label=labels[ii], sign=signs[ii])

                if ii == 0:
                    display = plot_stat_map(comp, axes=ax, title=title,    # noqaax.matchow color map
                                            black_bg=True, symmetric_cbar=True,
                                            vmax=vmax)
                else:
                    # use same cut coords
                    cut_coords = display.cut_coords  # noqa
                    display = plot_stat_map(comp, axes=ax, title=title,
                                            black_bg=True, symmetric_cbar=True,
                                            vmax=vmax, display_mode='ortho',
                                            cut_coords=cut_coords)

        # Save images instead of displaying
        if out_dir is not None:
            save_and_close(out_path=op.join(out_dir, png_name), fh=fh)


def plot_comparison_matrix(score_mat, labels, normalize=True,
                           out_dir=None, vmax=None, colorbar=True, prefix=""):

    # Settings
    score_mat, x_idx, y_idx = reorder_mat(score_mat, normalize=normalize)
    idx = np.arange(score_mat.shape[0])
    vmax = vmax  # or min(scores.max(), 10 if normalize else np.inf)
    vmin = 0  # 1 if normalize else 0

    # Plotting
    fh = plt.figure(figsize=(10, 10))
    ax = fh.gca()
    cax = ax.matshow(score_mat, vmin=vmin, vmax=vmax, cmap="jet")
    plt.grid(False)
    ax.set_xlabel("%s components" % (labels[1]))
    ax.set_ylabel("%s components" % (labels[0]))
    ax.set_xticks(idx), ax.set_xticklabels(x_idx)
    ax.set_yticks(idx), ax.set_yticklabels(y_idx)
    if colorbar:
        fh.colorbar(cax)

    # Saving
    if out_dir is not None:
        save_and_close(out_path=op.join(out_dir, '%s%s_%s_simmat%s.png' % (
            prefix, labels[0], labels[1], '-normalized' if normalize else '')))


def plot_term_comparisons(termscores_summary, labels, plot_type="heatmap",
                          color_list=('g', 'r', 'b'), out_dir=None):
    """
    Take the termscores summary DF and plot the term values for the given labels
    as a heatmap or radar graph (plot_type="heatmap" or "rader").

    The labels should be found in the DF column names.
    """
    for label in labels:
        assert label in termscores_summary.columns
        assert "%s_idx" % label in termscores_summary.columns

    # iterate over the set of ic indices for each label and plot terms
    # and term scores
    idx_cols = ["%s_idx" % label for label in labels]
    for name, group in termscores_summary.groupby(idx_cols):
        comparison_name = "_".join("%s[%d]" % (label, idx) for label, idx in zip(labels, name))
        title = "Term comparisons for %scomponents" % (comparison_name.replace("_", " "))

        # Get term scores for all the labels
        data = group[["terms"] + list(labels)]
        data = data.set_index("terms")

        if plot_type == "rader":
            out_path = op.join(out_dir, '%sterm_comparisons_rader.png' % comparison_name)
            N = len(group)
            theta = radar_factory(N)
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(1, 1, 1, projection='radar')

            ax.set_title(title, weight='bold', size='medium', position=(0.5, 1.1),
                         horizontalalignment='center', verticalalignment='center')

            y_min, y_max, y_tick = nice_bounds(data.values.min(), data.values.max())
            ax.set_ylim(y_min, y_max)
            ax.set_yticks([0], minor=True)
            ax.set_yticks(y_tick)
            ax.yaxis.grid(which='major', linestyle=':')
            ax.yaxis.grid(which='minor', linestyle='-')

            for label, color in zip(labels, color_list):
                ax.plot(theta, data[label], color=color)
                ax.fill(theta, data[label], facecolor=color, alpha=0.25)
            ax.set_varlabels(data.index.values)

            legend = plt.legend(labels, loc=(1.1, 0.9), labelspacing=0.1)
            plt.setp(legend.get_texts(), fontsize='small')

        elif plot_type == "heatmap":
            out_path = op.join(out_dir, '%sterm_comparisons_hm.png' % comparison_name)
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.set_title(title, weight='bold', size='medium', position=(0.5, 1.1),
                         horizontalalignment='center', verticalalignment='center')
            sns.heatmap(data, center=0.0, ax=ax, cbar=True)
            plt.yticks(rotation=0)

        else:
            raise NotImplementedError(plot_type)

        # Saving
        if out_dir is not None:
            save_and_close(out_path=out_path)
