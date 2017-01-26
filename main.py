# *- encoding: utf-8 -*-
# Author: Ami Tsuchida
# License: BSD
"""
How symmetric are the whole-brain ICA components? How are they similar to
half-brain ICA components?

For both WB and hal-brain components, find the sparsity, measured as L1 norm
and also as voxel count above a threshold, to compare whether they differ.
Sharper contrast (increase in vc-sparsity) in half-brain ICA indicate masking
of lateralized organization in WB.

To analyze symmetry of WB components, Calculate;
1) HPAI (Hemisphere Participation Asymmetry Index)
2) SSS (spatial symmetry score: similarity score between R and L, using correlation)
for each WB ICA component image to show the relationship between the two.

Then for each component, find the best-matching half-brain R&L components,
compare the SSS between them to see how much it  (increases relative to the
whole-brain SSS. Also compare terms associated with whole-brain and matching
half-brain components.

Do that with a hard loop on the # of components, then
plotting the mean SSS change.
"""

import os.path as op
import sys
sys.path.append(op.abspath(op.join(op.abspath(__file__), '..')))

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns

from nibabel_ext import NiftiImageWithTerms
from nilearn import datasets
from nilearn.image import index_img, math_img
from nilearn.plotting import plot_stat_map
from sklearn.externals.joblib import Memory
from textwrap import wrap

from nilearn_ext.datasets import fetch_neurovault
from nilearn_ext.decomposition import generate_components
from nilearn_ext.plotting import (save_and_close, rescale, plot_components,
                                  plot_components_summary)
from image_analysis.hpai import plot_hpai
from image_analysis.sparsity import SPARSITY_SIGNS, get_sparsity_threshold, plot_sparsity
from image_analysis.sss import plot_sss
from image_analysis.acni import plot_acni
from image_analysis.summary import load_or_generate_summary


def get_dataset(dataset, max_images=np.inf, **kwargs):
    """Retrieve & normalize dataset from nilearn"""
    # Download
    if dataset == 'neurovault':
        images, term_scores = fetch_neurovault(max_images=max_images, **kwargs)

    elif dataset == 'abide':
        dataset = datasets.fetch_abide_pcp(
            n_subjects=min(94, max_images), **kwargs)
        images = [{'absolute_path': p} for p in dataset['func_preproc']]
        term_scores = None

    elif dataset == 'nyu':
        dataset = datasets.fetch_nyu_rest(
            n_subjects=min(25, max_images), **kwargs)
        images = [{'absolute_path': p} for p in dataset['func']]
        term_scores = None

    else:
        raise ValueError("Unknown dataset: %s" % dataset)
    return images, term_scores


def load_or_generate_components(hemi, out_dir='.', force=False,
                                plot_dir=None, no_plot=False,
                                *args, **kwargs):
    """Load an image and return if it exists, otherwise compute via ICA"""
    # Only re-run if image doesn't exist.
    img_path = op.join(out_dir, '%s_ica_components.nii.gz' % hemi)
    generate_imgs = force or not op.exists(img_path)
    no_plot = no_plot or not generate_imgs

    if generate_imgs:
        img = generate_components(hemi=hemi, out_dir=out_dir, *args, **kwargs)
    else:
        img = NiftiImageWithTerms.from_filename(img_path)

    if not no_plot:
        plot_dir = plot_dir or op.join(out_dir, 'png')
        plot_components(img, hemi=hemi, out_dir=plot_dir)
        plot_components_summary(img, hemi=hemi, out_dir=plot_dir)

    return img


def generate_component_specific_plots(wb_master, components, scoring, out_dir=None):
    """Asdf"""
    start_idx = 0
    for c in components:
        wb_summary = wb_master[wb_master['n_comp'] == c]
        assert len(wb_summary) == c
        start_idx += c

        ### Generate component-specific plots ###
        # Save component-specific images in the component dir
        comp_outdir = op.join(out_dir, str(c))

        # 1) Relationship between positive and negative HPAI in wb components
        out_path = op.join(comp_outdir, "1_PosNegHPAI_%dcomponents.png" % c)

        # set color to the ACNI: ranging from 0 to 1 and reflects the proportion
        # of anti-correlated network (higher vals indicate strong ACN)
        color = wb_summary["ACNI_wb"]

        # size is proportional to vc-abs_wb
        size = wb_summary["rescaled_vc_abs"]
        ax = wb_summary.plot.scatter(x='posHPAI', y='negHPAI', c=color, s=size,
                                     xlim=(-1.1, 1.1), ylim=(-1.1, 1.1), edgecolors="grey",
                                     colormap='rainbow_r', colorbar=True, figsize=(7, 6))
        title = ax.set_title("\n".join(wrap("The relationship between HPAI on "
                                            "positive and negative side: "
                                            "n_components = %d" % c, 60)))
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        ax.yaxis.set_ticks_position('left')
        ax.yaxis.set_label_coords(-0.1, 0.5)
        ax.spines['left'].set_position(('data', 0))
        ax.xaxis.set_ticks_position('bottom')
        ax.spines['bottom'].set_position(('data', 0))
        ticks = [-1.1, -1.0, -0.5, 0, 0.5, 1.0, 1.1]
        labels = ['L', '-1.0', '-0.5', '0', '0.5', '1.0', 'R']
        plt.setp(ax, xticks=ticks, xticklabels=labels, yticks=ticks, yticklabels=labels)
        f = plt.gcf()
        title.set_y(1.05)
        f.subplots_adjust(top=0.8)
        cax = f.get_axes()[1]
        cax.set_ylabel('Proportion of anti-correlated network',
                       rotation=270, labelpad=20)

        save_and_close(out_path)

        # 2) Relationship between HPAI and SSS in wb components
        out_path = op.join(comp_outdir, "2_HPAIvsSSS_%dcomponents.png" % c)

        fh, axes = plt.subplots(1, 3, sharey=True, figsize=(18, 6))
        fh.suptitle("The relationship between HPAI values and SSS: "
                    "n_components = %d" % c, fontsize=16)
        colors = sns.color_palette("Paired", 6)
        hpai_colors = {'pos': (colors[4], colors[5]),
                       'neg': (colors[0], colors[1]),
                       'abs': (colors[2], colors[3])}
        for ax, sign in zip(axes, SPARSITY_SIGNS):
            size = wb_summary['rescaled_vc_%s' % sign]
            ax.scatter(wb_summary['%sHPAI' % sign], wb_summary['wb_SSS'],
                       c=hpai_colors[sign][0], s=size,
                       edgecolors=hpai_colors[sign][1])
            ax.set_xlabel("%s HPAI" % sign)
            ax.set_xlim(-1.1, 1.1)
            ax.set_ylim(0, 1)
            ax.spines['right'].set_color('none')
            ax.spines['top'].set_color('none')
            ax.spines['left'].set_position(('data', 0))
            ax.xaxis.set_ticks_position('bottom')
            ax.spines['bottom'].set_position(('data', 0))
            plt.setp(ax, xticks=ticks, xticklabels=labels)
            fh.text(0.04, 0.5, "Spatial Symmetry Score using %s" % scoring,
                    va='center', rotation='vertical')

        save_and_close(out_path)


def plot_variations_wb_vs_RL(imgs, wb_summary, metric='correlation', out_dir=None):
    components = np.unique(wb_summary['n_comp'])
    out_file = out_dir and op.join(
        out_dir,
        'comparison_img_%s.%s.nii' % (
            '_'.join([str(c) for c in components]),
            metric))

    # Load or generate the comparison image.
    if out_file and op.exists(out_file):
        comparison_img = nib.load(out_file)

    else:
        # Reorder r and l such that they best match wb.
        reordered = {}
        for k in ['R', 'L']:
            # Reorder & flip sign on the components within each image
            img = imgs[k]
            idx = wb_summary['matched%s' % k].astype(int)
            idx, sign = np.abs(idx), np.sign(idx)
            print idx, sign

            reordered[k] = nib.concat_images(
                [math_img('img * %d' % si, img=index_img(img, ii))
                 for ii, si in zip(idx, sign)])
            plot_components_summary(reordered[k], hemi='reordered%s' % k, out_dir=out_dir)

        # Make a comparison for wb vs. rl
        imgs['rl'] = math_img('R+L', **reordered)

        # Now do the dot product.
        if metric == 'l1':
            comparison_img = math_img('np.sum(np.abs(rl - wb), axis=3)', **imgs)
        elif metric == 'l2':
            comparison_img = math_img('np.sqrt(np.sum((rl - wb)**2, axis=3))', **imgs)
        elif metric == 'correlation':
            from nilearn.masking import apply_mask, unmask
            from nilearn_ext.masking import get_mask_by_key

            wb_mask = get_mask_by_key('wb')
            wb_data = apply_mask(imgs['wb'], wb_mask)
            rl_data = apply_mask(imgs['rl'], wb_mask)
            img_data = np.zeros((rl_data.shape[1],))
            for vi in range(img_data.size):
                img_data[vi] = stats.stats.pearsonr(
                    wb_data[:, vi], rl_data[:, vi])[0]
            # Do 1 - correlation, so that higher is more dissimilar.
            comparison_img = unmask(1 - img_data, wb_mask)
        else:
            raise ValueError("Unknown metric: %s" % metric)
        nib.save(comparison_img, out_file)

    # Plot components to make sure it's working...
    for hemi in ('wb', 'R', 'L', 'rl'):
        plot_components_summary(imgs[hemi], hemi=hemi, out_dir=out_dir)

    # Plot the comparison image
    for mode in ['x', 'y', 'z']:
        plot_stat_map(
            comparison_img, display_mode=mode, cut_coords=15, black_bg=True,
            title="rl vs. wb similarity (%s)" % mode, colorbar=True)
        if out_file:
            plot_file = out_file.replace('.nii', '-%s.png' % mode)
            save_and_close(plot_file)


### WIP ###
def loop_main_and_plot(components, scoring, dataset, query_server=True,
                       force=False, plot=True, max_images=np.inf,
                       memory=Memory(cachedir='nilearn_cache')):
    """
    Loop main.py to plot summaries of WB vs hemi ICA components
    """
    out_dir = op.join('ica_imgs', dataset)

    # Get data once
    images, term_scores = get_dataset(dataset, max_images=max_images,
                                      query_server=query_server)

    # Perform ICA for WB, R and L for each n_component once and
    # run match analysis. Also store each component images
    hemis = ("wb", "R", "L")
    imgs = {hemi: [] for hemi in hemis}
    for c in components:
        for hemi in hemis:
            print("Generating or loading ICA components for %s,"
                  " n=%d components" % (hemi, c))
            nii_dir = op.join('ica_nii', dataset, '%d' % c)
            kwargs = dict(images=[im['absolute_path'] for im in images],
                          n_components=c, term_scores=term_scores,
                          out_dir=nii_dir, memory=memory)

            img = load_or_generate_components(
                hemi=hemi, force=force, no_plot=not plot, **kwargs)
            imgs[hemi].append(img)

        print ("Running match analysis for n=%d components" % c)
        ica_images = {hemi: imgs[hemi][-1] for hemi in hemis}
        # Call match.py with ica_images to perform matching. Decide
        # what output to keep (score_mat for all the combination of matching?)

        # Generate plots of variations in wb vs RL
        # This depends on the matching method, so make use of the output from
        # above call to match.py

        # print "Examining differences in wb vs. rl for %d components..." % c
        # comp_dir = op.join(out_dir, str(c))
        # comp_imgs = {hemi: imgs[hemi][ci] for hemi in hemis}
        # plot_variations_wb_vs_RL(comp_imgs, wb_summary, out_dir=comp_dir)

    # Use wb images to determine threshold for voxel count sparsity
    print("Getting sparsity threshold.")
    global_percentile = 99.9
    sparsity_threshold = get_sparsity_threshold(
        images=imgs["wb"], global_percentile=global_percentile)
    print("Using global sparsity threshold of %0.8f for sparsity calculation"
          % sparsity_threshold)

    # Loop again this time to perform image analyses on wb, R, L component images
    master_DFs = {"%s_master" % hemi: pd.DataFrame() for hemi in hemis}
    for ci, c in enumerate(components):
        for hemi in hemis:
            print("Running image analysis for %s ICA, %d components" % (hemi, c))
            summary = load_or_generate_summary(
                imgs[hemi][ci], hemi=hemi, n_components=c, dataset=dataset,
                sparsity_threshold=sparsity_threshold, acni_percentile=95.0,
                hpai_percentile=95.0)
            master_DFs["%s_master" % hemi] = master_DFs["%s_master" % hemi].append(summary)

    # Reset indices of master DFs and save
    for key in master_DFs:
        master_DFs[key].reset_index(inplace=True)
        master_DFs[key].to_csv(op.join(out_dir, '%sICA_im_analsis_summary.csv' % key))

    # To set size proportional to vc sparsity in several graphs,
    # add columns with vc vals
    wb_master = master_DFs["wb_master"]
    for sign in SPARSITY_SIGNS:
        wb_master["rescaled_vc_%s" % sign] = rescale(wb_master["vc-%s_wb" % sign])

    # 1) Component-specific plots
    print "Generating plots for each n_components."
    generate_component_specific_plots(
        wb_master=wb_master, components=components, scoring=scoring, out_dir=out_dir)

    # 2) Main summary plots over the range of n_components
    print "Generating summary plots.."
    plot_hpai(wb_master=wb_master, sparsity_threshold=sparsity_threshold, out_dir=out_dir)
    plot_sparsity(out_dir=out_dir, **master_DFs)

    # SSS plot is the only one that use matching results to compare
    # wb and matched RL SSS, so decide later what to do...
    # plot_sss(wb_master=wb_master, scoring=scoring, out_dir=out_dir)
    plot_acni(out_dir=out_dir, **master_DFs)

    # Need to figure out what summary/plots should be generated for matching performance
    # plot_matching(wb_master=wb_master, scoring=scoring, out_dir=out_dir)


if __name__ == '__main__':
    import warnings
    from argparse import ArgumentParser

    # Look for image computation errors
    warnings.simplefilter('ignore', DeprecationWarning)
    warnings.simplefilter('error', RuntimeWarning)  # Detect bad NV images

    # Arg parsing
    hemi_choices = ['R', 'L', 'wb']
    parser = ArgumentParser(description="Really?")
    parser.add_argument('--force', action='store_true', default=False)
    parser.add_argument('--offline', action='store_true', default=False)
    parser.add_argument('--no-plot', action='store_true', default=False)
    parser.add_argument('--components', nargs='?',
                        default="5,10,15,20,30,40,50,75,100")
    parser.add_argument('--dataset', nargs='?', default='neurovault',
                        choices=['neurovault', 'abide', 'nyu'])
    parser.add_argument('--scoring', nargs='?', default='correlation',
                        choices=['l1norm', 'l2norm', 'correlation'])
    parser.add_argument('--max-images', nargs='?', type=int, default=np.inf)
    args = vars(parser.parse_args())

    # Alias args
    query_server = not args.pop('offline')
    plot = not args.pop('no_plot')
    components = [int(c) for c in args.pop('components').split(',')]

    loop_main_and_plot(
        components=components, query_server=query_server, plot=plot, **args)
    plt.show()  # make sure any remaining plots are shown.
