# *- encoding: utf-8 -*-
# Author: Ami Tsuchida
# License: BSD
"""
"""

import os.path as op

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from nilearn_ext.plotting import save_and_close


def plot_sss(wb_master, scoring, out_dir):
    components = np.unique(wb_master['n_comp'])

    # 5) SSS for wb components and matched RL components
    print "Plotting SSS for wb components"
    pastel2 = sns.color_palette("Pastel2")
    set2 = sns.color_palette("Set2")
    palette = [set2[1], pastel2[1]]
    title = "Spatial Symmetry Score for WB and the matched RL components"
    xlabel = "Number of components"
    ylabel = "Spatial Symmetry Score using %s " % scoring

    out_path = op.join(out_dir, '5_wb_RL_SSS_box.png')

    sss_cols = ["wb_SSS", "matchedRL_SSS"]
    sss = pd.melt(wb_master[["n_comp"] + sss_cols], id_vars="n_comp",
                  value_vars=sss_cols)

    fh = plt.figure(figsize=(10, 6))
    plt.title(title)
    ax = sns.boxplot(x="n_comp", y="value", hue="variable", data=sss, palette=palette)
    ax.set(xlabel=xlabel, ylabel=ylabel)

    save_and_close(out_path, fh=fh)

    # Same data but with paired dots and lines
    out_path = op.join(out_dir, '5_wb_RL_SSS_dots.png')

    fh = plt.figure(figsize=(10, 6))
    plt.title(title)

    # first plot lines between individual plots
    for i in range(len(wb_master.index)):
        linestyle = "-" if (wb_master.wb_SSS[i] - wb_master.matchedRL_SSS[i]) < 0 else "--"
        plt.plot([wb_master.n_comp.astype(int)[i] - 1, wb_master.n_comp.astype(int)[i] + 1],
                 [wb_master.wb_SSS[i], wb_master.matchedRL_SSS[i]],
                 c="grey", linestyle=linestyle, linewidth=1.0)

    # add scatter points
    plt.scatter(wb_master.n_comp.astype(int) - 1, wb_master.wb_SSS, s=80,
                edgecolor="orange", facecolor=set2[1], label="WB")
    plt.scatter(wb_master.n_comp.astype(int) + 1, wb_master.matchedRL_SSS, s=80,
                edgecolor="orange", facecolor=pastel2[1], label="matched RL")
    plt.legend()

    # add mean change
    by_comp = wb_master.groupby("n_comp")
    for c, grouped in by_comp:
        linestyle = "-" if (grouped.wb_SSS.mean() - grouped.matchedRL_SSS.mean()) < 0 else "--"
        plt.plot([int(c) - 1, int(c) + 1], [grouped.wb_SSS.mean(), grouped.matchedRL_SSS.mean()],
                 c="black", linestyle=linestyle)
    comp_arr = np.asarray(map(int, components))
    plt.scatter(comp_arr - 1, by_comp.wb_SSS.mean(), c="orange", s=100, marker="+")
    plt.scatter(comp_arr + 1, by_comp.matchedRL_SSS.mean(), c="orange", s=100, marker="+")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    save_and_close(out_path, fh=fh)
