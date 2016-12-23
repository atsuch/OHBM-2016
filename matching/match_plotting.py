# *- encoding: utf-8 -*-
# Author: Ben Cipollini, Ami Tsuchida
# License: BSD

import os.path as op

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from nilearn.image import index_img, math_img
from nilearn.plotting import plot_stat_map

from nilearn_ext.utils import reorder_mat, get_match_idx_pair, get_percentile_val
from nilearn_ext.plotting import _title_from_terms, nice_bounds, save_and_close
from nilearn_ext.radar import radar_factory


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

    # Calculate a thr and vmax optimal across all the plots
    thr = get_percentile_val(*images, percentile=90.0)
    vmax = get_percentile_val(*images, percentile=99.99)

    for i in range(n_comp):
        c1i, c2i = idx_pair[0][i], idx_pair[1][i]
        cis = [c1i, c2i]

        prefix = "unmatched-" if i >= n_components else ""
        num = i - n_components if i >= n_components else i
        png_name = '%s%s_%s_%s.png' % (prefix, labels[0], labels[1], num)
        print "plotting %s, %sforcing one-to-one match: %s" % (png_name, '' if force else 'not ')

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
                symmetric_cbar=True, threshold=thr, vmax=vmax)

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
                    display = plot_stat_map(
                        comp, axes=ax, title=title, black_bg=True,
                        symmetric_cbar=True, threshold=thr, vmax=vmax)
                else:
                    # use same cut coords
                    cut_coords = display.cut_coords  # noqa
                    display = plot_stat_map(
                        comp, axes=ax, title=title, black_bg=True,
                        symmetric_cbar=True, threshold=thr, vmax=vmax,
                        display_mode='ortho', cut_coords=cut_coords)

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


### WIP ###
def plot_matching_results(wb_master, scoring, out_dir):

    # 4) Matching results: average matching scores and proportion of unmatched
    print "Plotting matching results"
    set2 = sns.color_palette("Set2")
    palette = [set2[2], set2[0], set2[1]]
    title = "Matching scores for the best-matched pairs"
    xlabel = "Number of components"
    ylabel = "Matching score using %s" % scoring

    out_path = op.join(out_dir, '4_Matching_results_box.png')
    score_cols = ["matchR_score", "matchL_score", "matchRL-unforced_score"]
    match_scores = pd.melt(wb_master[["n_comp"] + score_cols], id_vars="n_comp",
                           value_vars=score_cols)

    fh = plt.figure(figsize=(10, 6))
    plt.title(title)
    ax = sns.boxplot(x="n_comp", y="value", hue="variable", data=match_scores, palette=palette)
    ax.set(xlabel=xlabel, ylabel=ylabel)

    save_and_close(out_path, fh=fh)

    # Same data but in line plot: also add proportion of unmatched
    out_path = op.join(out_dir, '4_Matching_results_line.png')

    unmatch_cols = ["n_unmatchedR", "n_unmatchedL"]
    unmatched = pd.melt(wb_master[["n_comp"] + unmatch_cols], id_vars="n_comp",
                        value_vars=unmatch_cols)
    unmatched["proportion"] = unmatched.value / unmatched.n_comp.astype(float)

    fh = plt.figure(figsize=(10, 6))
    plt.title(title)
    ax = sns.pointplot(x="n_comp", y="value", hue="variable", palette=palette,
                       data=match_scores, dodge=0.3)
    sns.pointplot(x="n_comp", y="proportion", hue="variable", palette=palette,
                  data=unmatched, dodge=0.3, ax=ax, linestyles="--", markers="s")
    ax.set(xlabel=xlabel, ylabel=ylabel)
    fh.text(0.95, 0.5, "Proportion of unmatched R- or L- components", va="center", rotation=-90)

    save_and_close(out_path, fh=fh)
