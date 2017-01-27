# *- encoding: utf-8 -*-
# Author: Ami Tsuchida
# License: BSD
"""
"""

import os
import os.path as op
import pandas as pd

from .acni import calculate_acni
from .hpai import calculate_hpai
from .sparsity import get_hemi_sparsity, SPARSITY_SIGNS
from nilearn_ext.decomposition import compare_RL


def load_or_generate_summary(ica_image, hemi, n_components, dataset,
                             sparsity_threshold, acni_percentile=95.0,
                             hpai_percentile=95.0, force=False, out_dir=None):
    """
    For a given ICA image, load the image analysis csv if it already exist, or
    run image analysis to get and save necessary summary data.

    Returns a DF containing 1) sparsity 2) ACNI, and for wb, 3) HPAI and 4) SSS.
    """
    # Directory to find or save the summary csvs
    out_dir = out_dir or op.join('ica_imgs', dataset, str(n_components))
    summary_csv = "%s_ICA_im_analysis.csv" % hemi

    # If summary data are already saved as csv file, simply load them
    if not force and op.exists(op.join(out_dir, summary_csv)):
        summary = pd.read_csv(op.join(out_dir, summary_csv))

    # Otherwise run image analysis and save as csv
    else:
        # Initialize a DF
        summary = pd.DataFrame({"n_comp": [n_components] * n_components})
        if not op.exists(out_dir):
            os.makedirs(out_dir)

        # 1) Get sparsity
        sparsityTypes = ("l1", "vc-pos", "vc-neg", "vc-abs")
        hemi_labels = ["R", "L", "wb"] if hemi == "wb" else [hemi]

        # sparsity_results = {label: sparsity_dict}
        sparsity_results = {label: get_hemi_sparsity(ica_image, label,
                            thr=sparsity_threshold) for label in hemi_labels}

        for s_type in sparsityTypes:
            for label in hemi_labels:
                summary["%s_%s" % (s_type, label)] = sparsity_results[label][s_type]

        # 2) Get ACNI (Anti-Correlated Network index).
        for label in hemi_labels:
            summary["ACNI_%s" % label] = calculate_acni(
                ica_image, hemi=label, percentile=acni_percentile)

        # For "wb" only 3) HPAI (Hemispheric participation asymmetry index).
        if hemi == "wb":
            hpai_d = calculate_hpai(ica_image, percentile=hpai_percentile)
            for sign in SPARSITY_SIGNS:
                summary["%sHPAI" % sign] = hpai_d[sign]

            # Also for "wb" only 4) Get SSS (Spatial symmetry score).
            summary["SSS"] = compare_RL(ica_image)

        # Save DF
        summary.to_csv(op.join(out_dir, summary_csv))

    return summary
