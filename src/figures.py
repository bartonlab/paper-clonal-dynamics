"""
This Python file contains code for generating plots in:
    Title of the paper
    Yunxiao Li, and
    John P. Barton (john.barton@ucr.edu)

Additional code to pre-process the data and pass it to these plotting
routines is contained in the Jupyter notebook `figures.ipynb`.
"""

#############  PACKAGES  #############

import sys

import numpy as np
import math

from copy import deepcopy

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec

import mplot as mp

import seaborn as sns

from scipy import stats

sys.path.append('../paper-clade-reconstruction/src')
import MPL as MPL
import analyze_and_plot as AP
import reconstruct_clades as RC
import infer_fitness as IF
import print_info as PI
import LTEE
import LTEE_helper as LH
import lolipop_helper
import PALTEanalysis
import tobramycin_analysis

############# PARAMETERS #############

# GLOBAL VARIABLES

# METHODS = ['SL', 'true_cov', 'recovered', 'est_cov', 'Lolipop']
METHODS = ['true_cov', 'recovered', 'Lolipop', 'est_cov', 'SL']

# Standard color scheme

BKCOLOR  = '#252525'
LCOLOR   = '#969696'
C_BEN    = '#EB4025' #'#F16913'
C_BEN_LT = '#F08F78' #'#fdd0a2'
C_NEU    =  LCOLOR   #'#E8E8E8' # LCOLOR
C_NEU_LT = '#E8E8E8' #'#F0F0F0' #'#d9d9d9'
C_DEL    = '#3E8DCF' #'#604A7B'
C_DEL_LT = '#78B4E7' #'#dadaeb'

GREY_COLOR_RGB = (0.5, 0.5, 0.5)
GREY_COLOR_HEX = '#808080'

USE_COLOR_PALETTE_FROM_MATPLOTLIB = False
USE_COLOR_PALETTE_FROM_SEABORN = True

if USE_COLOR_PALETTE_FROM_MATPLOTLIB:
    cmap = matplotlib.cm.get_cmap("Set3")
    CLADE_COLORS = (GREY_COLOR_HEX,) + cmap.colors[2:8] + cmap.colors[9:]
    CLADE_COLORS = [GREY_COLOR_HEX, '#1f77b4', '#ff7f0e', '#bd81be', '#b4df69', '#fdcee6', '#bfbbdb', '#cdecc6', '#fc8172' '#10ee6f']
    # CLADE_COLORS = [GREY_COLOR_RGB, '#81b2d4', '#feb562', '#bfbbdb', '#bd81be', '#b4df69', '#fdcee6', '#cdecc6', '#fc8172' '#10ee6f']

    # cmap = matplotlib.cm.get_cmap("tab10")
    # ALLELE_COLORS = cmap.colors
    cmap = matplotlib.cm.get_cmap("hsv")
    ALLELE_CMAP = cmap
elif USE_COLOR_PALETTE_FROM_SEABORN:
    ALLELE_COLORS = sns.husl_palette(20)
    CLADE_COLORS = [GREY_COLOR_RGB] + sns.husl_palette(10)
    # SEABORN_COLORS = sns.husl_palette(30)
    # ALLELE_COLORS = SEABORN_COLORS[:20]
    # CLADE_COLORS = SEABORN_COLORS[20:]

CMAP = sns.diverging_palette(145, 300, as_cmap=True)
CMAP_NONCLONAL = sns.diverging_palette(0, 145, as_cmap=True)

LTEE_MAJOR_FIXED_COLOR = '#1f77b4'
LTEE_MINOR_FIXED_COLOR = '#ff7f0e'
LTEE_EXTINCT_COLOR = GREY_COLOR_HEX
LTEE_ANCESTOR_FIXED_COLOR = '#d62728'
LTEE_ANCESTOR_POLYMORPHIC_COLOR = '#e377c2'
LTEE_MAJOR_POLYMORPHIC_COLOR = '#2ca02c'

LTEE_COLORS = {
    1: LTEE_EXTINCT_COLOR, # Extinct
    2: LTEE_ANCESTOR_FIXED_COLOR, # Ancestor fixed
    3: LTEE_MAJOR_FIXED_COLOR, # Major fixed
    4: LTEE_MINOR_FIXED_COLOR, # Minor fixed
    5: LTEE_ANCESTOR_POLYMORPHIC_COLOR, # Ancestor polymorphic
    6: LTEE_MAJOR_POLYMORPHIC_COLOR, # Major polymorphic
    7: 'yellow', # NA
}

# Plot conventions

def cm2inch(x): return float(x)/2.54
SINGLE_COLUMN   = cm2inch(8.5)
ONE_FIVE_COLUMN = cm2inch(11.4)
DOUBLE_COLUMN   = cm2inch(17.4)

GOLDR        = (1.0 + np.sqrt(5)) / 2.0
TICKLENGTH   = 3
TICKPAD      = 3
AXWIDTH      = 0.4
SUBLABELS    = ['A', 'B', 'C', 'D', 'E', 'F']
FREQUENCY_YLIM = [-0.04, 1.04]
GENERATION_XLIM = [-40, 1010]

# paper style

FONTFAMILY   = 'Arial'
SIZESUBLABEL = 8
SIZELABEL    = 6
SIZELEGEND   = 6
SIZEANNOTATION = 6
SIZEANNOTATION_SINGLE = 4
SIZEANNOTATION_HEATMAP = 6
SIZETICK     = 6
SMALLSIZEDOT = 6.
SIZEDOT      = 12
SIZELINE     = 0.6

DEF_FIGPROPS = {
    'transparent' : True,
    'edgecolor'   : None,
    'dpi'         : 1000,
    # 'bbox_inches' : 'tight',
    'pad_inches'  : 0.05,
}

DEF_SUBLABELPROPS = {
    'family'  : FONTFAMILY,
    'size'    : SIZESUBLABEL+1,
    'weight'  : 'bold',
    'ha'      : 'center',
    'va'      : 'center',
    'color'   : 'k',
    'clip_on' : False,
}

DEF_SUBLABELPOSITION_1x2 = {
    'x'       : -0.15,
    'y'       : 1.05,
}

DEF_SUBLABELPOSITION_2x1 = {
    'x'       : -0.175,
    'y'       : 1.05,
}

DEF_SUBLABELPOSITION_2x2 = {
    'x'       : -0.07,
    'y'       : 1.07,
}

DEF_SUBPLOTS_ADJUST_1x2 = {
    'wspace'  : 0.4,
}

DEF_SUBPLOTS_ADJUST_2x2 = {
    'left'    : 0.076,
    'bottom'  : 0.067,
    'right'   : 0.998,
    'top'     : 0.949,
    'wspace'  : 0.21,
    'hspace'  : 0.3,
}

DEF_LABELPROPS = {
    'family' : FONTFAMILY,
    'size'   : SIZELABEL,
    'color'  : BKCOLOR
}

DEF_TICKPROPS = {
    'length'    : TICKLENGTH,
    'width'     : AXWIDTH/2,
    'pad'       : TICKPAD,
    'axis'      : 'both',
    'which'     : 'both',
    'direction' : 'out',
    'colors'    : BKCOLOR,
    'labelsize' : SIZETICK,
    'bottom'    : True,
    'left'      : True,
    'top'       : False,
    'right'     : False,
}

DEF_TICKPROPS_TOP = {
    'length'    : TICKLENGTH,
    'width'     : AXWIDTH/2,
    'pad'       : TICKPAD,
    'axis'      : 'both',
    'which'     : 'both',
    'direction' : 'out',
    'colors'    : BKCOLOR,
    'labelsize' : SIZETICK,
    'bottom'    : False,
    'left'      : True,
    'top'       : True,
    'right'     : False,
}

DEF_TICKPROPS_HEATMAP = {
    'length'    : TICKLENGTH,
    'width'     : AXWIDTH/2,
    'pad'       : TICKPAD,
    'axis'      : 'both',
    'which'     : 'both',
    'direction' : 'out',
    'colors'    : BKCOLOR,
    'labelsize' : SIZETICK,
    'bottom'    : False,
    'left'      : False,
    'top'       : False,
    'right'     : False,
    'pad'       : 2,
}

DEF_TICKPROPS_COLORBAR = {
    'length'    : TICKLENGTH,
    'width'     : AXWIDTH/2,
    'pad'       : TICKPAD,
    'axis'      : 'both',
    'which'     : 'both',
    'direction' : 'out',
    'colors'    : BKCOLOR,
    'labelsize' : SIZETICK,
    'bottom'    : False,
    'left'      : False,
    'top'       : False,
    'right'     : True,
}

DEF_AXPROPS = {
    'linewidth' : AXWIDTH,
    'linestyle' : '-',
    'color'     : BKCOLOR
}

DEF_HEATMAP_KWARGS = {
    'vmin'      : 0.65,
    'vmax'      : 0.9,
    'cmap'      : 'GnBu',
    'alpha'     : 0.75,
    'cbar'      : False,
    'annot'     : True,
    'annot_kws' : {"fontsize": 6}
}

FIG4_LEFT, FIG4_BOTTOM, FIG4_RIGHT, FIG4_TOP = 0.14, 0.223, 0.965, 0.935
GOLD_RATIO = 1.4
LABEL_SPEARMANR = "Rank correlation between true and inferred\nselection coefficients" + r" (Spearman’s $\rho$)"
LABEL_SPEARMANR_COVARIANCE = "Rank correlation between true and inferred\ncovariances" + r" (Spearman’s $\rho$)"

LABEL_SPEARMANR_THREE = "Rank correlation between true\nand inferred selection coefficients\n" + r"(Spearman’s $\rho$)"
LABEL_SPEARMANR_COVARIANCE_THREE = "Rank correlation between true\nand inferred covariances\n" + r"(Spearman’s $\rho$)"

LABEL_SPEARMANR_FOUR = "Rank correlation\nbetween true and inferred\nselection coefficients\n" + r"(Spearman’s $\rho$)"
LABEL_SPEARMANR_COVARIANCE_FOUR = "Rank correlation\nbetween true and\ninferred covariances\n" + r"(Spearman’s $\rho$)"
LABEL_SPEARMANR_FITNESS_FOUR = "Rank correlation\nbetween measured\nand inferred fitness\n" + r"(Spearman’s $\rho$)"

LABEL_SPEARMANR_FOUR_2 = "Rank correlation\nbetween true and inferred\nselection coefficients\n" + r"(Spearman’s $\rho$)"
LABEL_SPEARMANR_COVARIANCE_FOUR_2 = "Rank correlation\nbetween true and\ninferred covariances\n" + r"(Spearman’s $\rho$)"
LABEL_SPEARMANR_FITNESS_FOUR_2 = "Rank correlation\nbetween measured\nand inferred fitness\n" + r"(Spearman’s $\rho$)"

LABEL_PEARSONR_FOUR = "Linear correlation\nbetween true and inferred\nselection coefficients\n" + r"(Pearson’s $r$)"
LABEL_PEARSONR_COVARIANCE_FOUR = "Linear correlation\nbetween true and\ninferred covariances\n" + r"(Pearson’s $r$)"
LABEL_PEARSONR_FITNESS_FOUR = "Linear correlation\nbetween measured and\ninferred fitness\n" + r"(Pearson’s $r$)"

PARAMS = {'text.usetex': False, 'mathtext.fontset': 'stixsans', 'mathtext.default': 'regular', 'pdf.fonttype': 42, 'ps.fonttype': 42}
plt.rcParams.update(matplotlib.rcParamsDefault)
plt.rcParams.update(PARAMS)

EXCHANGE_FIRST_TWO_CLADES_LTEE = {
    'm1': True, 'm2': True, 'm4': True, 'm5': True, 'm6': False, 'p1': False, 'p3': False, 'p5': True, 'p6': True,
}


############# HELPER  FUNCTIONS #############

def swap(a, b):
    a, b = b, a


def rgb_to_hex(rgb):
    rgb = tuple([round(256 * _) for _ in rgb])
    return '#%02x%02x%02x' % (rgb)


def set_ticks_labels_axes(xlim=None, ylim=None, xticks=None, xticklabels=None, yticks=None, yticklabels=None, xlabel=None, ylabel=None):

    ax = plt.gca()

    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)

    if xticks is not None and xticklabels is not None:
        plt.xticks(ticks=xticks, labels=xticklabels, fontsize=SIZELABEL, rotation=0)
    else:
        plt.xticks(fontsize=SIZELABEL)

    if yticks is not None and yticklabels is not None:
        plt.yticks(ticks=yticks, labels=yticklabels, fontsize=SIZELABEL)
    else:
        plt.yticks(fontsize=SIZELABEL)

    if xlabel is not None:
        plt.xlabel(xlabel, fontsize=SIZELABEL)

    if ylabel is not None:
        plt.ylabel(ylabel, fontsize=SIZELABEL)

    ax.tick_params(**DEF_TICKPROPS)
    plt.setp(ax.spines.values(), **DEF_AXPROPS)


def get_mapping_clade_LTEE_to_identity_index():
    clade_hmm_states = {'A':0,'E':1,'FB':2,'FM':3, 'Fm':4,'PB':5,'PM':6,'Pm':7,'PB*':8}
    major, minor, abe = 0, 1, 2
    clade_LTEE_to_index = {}
    for clade in [0, 1, 2, 5, 8]:
        clade_LTEE_to_index[clade] = abe
    for clade in [3, 6]:
        clade_LTEE_to_index[clade] = major
    for clade in [4, 7]:
        clade_LTEE_to_index[clade] = minor
    return clade_LTEE_to_index


def add_shared_label(fig, xlabel=None, ylabel=None, xlabelpad=-3, ylabelpad=-3, fontsize=SIZELABEL):
    """Adds a xlabel shared by subplots."""

    # add a big axes, hide frame in order to generate a shared x-axis label
    fig.add_subplot(111, frameon=False)
    # hide tick and tick label of the big axes
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.grid(False)
    if xlabel is not None:
        plt.xlabel(xlabel, fontsize=fontsize)
        plt.gca().xaxis.labelpad = xlabelpad
    if ylabel is not None:
        plt.ylabel(ylabel, fontsize=fontsize)
        plt.gca().yaxis.labelpad = ylabelpad


def ticks_for_heatmap(length):
    """Returns ticks of a certain length, with each tick at position (i + 0.5)."""
    return np.arange(0.5, length + 0.5, 1)


def get_color_by_value_percentage(value, vmin=0, vmax=1, colors=None):
    if value <= vmin:
        index = 0
    elif value >= vmax:
        index = -1
    else:
        index = int((value - vmin) / (vmax - vmin) * len(colors))
    return colors[index]


############# PLOTTING  FUNCTIONS #############

############# MAIN FIGURES #############

def parse_info_from_simulation(simulation):

    T, L = simulation['traj'].shape
    N = np.sum(simulation['nVec'][0])
    genotypes, genotype_frequencies, pop = AP.getDominantGenotypeTrajectories(simulation, T=T, threshold=0, totalPopulation=N)
    return T, L, N, genotypes, genotype_frequencies, pop


def get_signature_mutation_index(genotype):
    muts = get_mutations_of_genotype(genotype)
    if not muts:
        return None
    return muts[-1]


def map_genotype_index_to_clade(genotypes, reconstruction):

    genotype_index_to_clade = {}
    clades_muts = reconstruction.cladeMuts
    other_muts = reconstruction.otherMuts
    for i, genotype in enumerate(genotypes):
        muts = get_mutations_of_genotype(genotype)
        if not muts:
            genotype_index_to_clade[i] = 0 # All wild type. Ancestor clade.
            continue
        sig_mut = muts[-1]
        for k, clade_muts in enumerate(clades_muts):
            if sig_mut in clade_muts:
                genotype_index_to_clade[i] = k + 1
        # for k, clade_muts in enumerate(clades_muts):
        #     if np.all([mut in clade_muts + other_muts for mut in muts]):
        #         genotype_index_to_clade[i] = k
        if i not in genotype_index_to_clade:
            # print(f'Can not find a clade that includes all mutations of\n\tgenotype %2d - {genotype}.' % i)
            genotype_index_to_clade[i] = 0
            # print(f'Genotype %2d - {genotype} assigned to ancestor clade' % i)
    clade_to_genotype_indices = {clade: [] for clade in range(reconstruction.numClades + 1)}
    for genotype_index, clade in genotype_index_to_clade.items():
        clade_to_genotype_indices[clade].append(genotype_index)

    return genotype_index_to_clade, clade_to_genotype_indices


def get_mutations_of_genotype(genotype):

    return [i for i in range(len(genotype)) if genotype[i] > 0]


def get_genotype_frequencies_colors_for_stackplot(genotype_frequencies, genotype_colors, clade_to_genotype_indices, clade_colors, lightness_difference=0.1, divider_frequency=0.01, divider_color='white'):

    T = len(genotype_frequencies[0])
    genotype_frequencies_for_stackplot = []
    genotype_colors_for_stackplot = []
    genotype_indices_for_stackplot = []
    for clade, genotype_indices in clade_to_genotype_indices.items():
        for i, genotype_index in enumerate(genotype_indices):
            padded_frequencies = np.zeros(T)
            for t in range(T):
                padded_frequencies[t] = max(genotype_frequencies[genotype_index, t] - divider_frequency, 0)
            genotype_frequencies_for_stackplot.append(padded_frequencies)
            genotype_frequencies_for_stackplot.append(genotype_frequencies[genotype_index] - padded_frequencies)

            # genotype_colors_for_stackplot.append(genotype_colors[genotype_index])
            genotype_colors_for_stackplot.append(adjust_lightness(clade_colors[clade], lightness_difference * (i - len(genotype_indices) // 2)))
            genotype_colors_for_stackplot.append(divider_color)

            genotype_indices_for_stackplot.append(genotype_index)
    return np.array(genotype_frequencies_for_stackplot), genotype_colors_for_stackplot, genotype_indices_for_stackplot


def adjust_lightness(rgb, adjustment):
    adjusted = np.zeros_like(rgb, dtype=float)
    for i in range(len(adjusted)):
        adjusted[i] = min(0.99, max(0.01, rgb[i] + adjustment))
    return tuple(adjusted)


def plot_genotype_annotation(x, y, genotype, max_freq, allele_colors, min_max_freq_to_annotate=0.3, dx=10, plot_dot_for_WT_locus=True, fontsize=SIZEANNOTATION, plot_single_column=False, add_background_behind_annotation=False, background_color='white', background_height=0.038):
    if max_freq < min_max_freq_to_annotate:
        return
    ax = plt.gca()
    pre_is_dot = False
    x_0 = x
    if plot_single_column:
        if plot_dot_for_WT_locus:
            dx *= 1.3
        else:
            dx *= 1.4
    for locus, allele in enumerate(genotype):
        color = allele_colors[locus]
        if plot_dot_for_WT_locus:
            if allele > 0:
                if x > x_0:
                    x += 0.3 * dx if pre_is_dot else 0.8 * dx
                plt.text(x, y, locus, color=color, fontsize=fontsize, zorder=10)
                pre_is_dot = False
                if locus >= 10:
                    x += 1.3 * dx
                else:
                    x += 0.4 * dx

            else:
                x += 0.35 * dx if pre_is_dot else 0.55 * dx
                plt.text(x, y, '.', color='black', alpha=0.7, fontsize=fontsize, zorder=10)
                pre_is_dot = True
                x += 0.35 * dx
        else:
            if allele > 0:
                x += 0.6 * dx
                plt.text(x, y, locus, color=color, fontsize=fontsize, zorder=10)
                if locus >= 10:
                    x += 1.4 * dx
                else:
                    x += 0.6 * dx
    if add_background_behind_annotation:
        patch = mpatches.FancyBboxPatch((x_0 + dx / 2, y + 0.015), x - x_0, background_height, color=background_color, zorder=5, boxstyle=mpatches.BoxStyle("Round", pad=0.02, rounding_size=0), clip_on=False)
        ax.add_patch(patch)
    pass


def get_colors_accordingly(genotypes, genotype_frequencies, clade_to_genotype_indices, reconstruction, genotype_color_indices_to_skip=None, number_redundant_colors=4, weight_by_freq_sum=True, weight_by_freq_at_max_freq_time=False, verbose=False):

    # genotype_colors = [GREY_COLOR_RGB] + sns.husl_palette(len(genotypes) - 1)
    mutant_genotype_colors = sns.husl_palette(len(genotypes) - 1 + number_redundant_colors)
    if genotype_color_indices_to_skip is not None:
        mutant_genotype_colors = [mutant_genotype_colors[i] for i in range(len(mutant_genotype_colors)) if i not in genotype_color_indices_to_skip]
    genotype_colors = [GREY_COLOR_RGB] + mutant_genotype_colors  # Redundant colors to separate colors of the first and the last clade.

    allele_colors = [None for _ in range(len(genotypes) - 1)]
    for i, genotype in enumerate(genotypes):
        sig_mut = get_signature_mutation_index(genotype)
        if sig_mut is not None:
            allele_colors[sig_mut] = genotype_colors[i]

    clade_colors = []
    for i in range(1 + reconstruction.numClades):
        weights = 0
        clade_color = np.array((0, 0, 0), dtype=float)
        max_clade_freq_time = np.argmax(reconstruction.cladeFreqWithAncestor[:, i])
        if verbose:
            print(f'Clade {i}, max freq at time {max_clade_freq_time}')
        for genotype_index in clade_to_genotype_indices[i]:
            if weight_by_freq_sum:
                weight = np.sum(genotype_frequencies[genotype_index])
            else:
                weight = genotype_frequencies[genotype_index, max_clade_freq_time]
            if verbose:
                print(f'Genotype {genotype_index}, max freq {weight}')
                print(np.array(genotype_colors[genotype_index]), weight)
            clade_color += weight * np.array(genotype_colors[genotype_index])
            weights += weight
        clade_color /= weights
        if verbose:
            print(f'Clade {i} clade_color {clade_color}')
        if np.any(clade_color > 1):
            print(f'RGB not valid. ')
        clade_colors.append(tuple(clade_color))

    genotype_colors = [None for _ in range(len(genotypes))]
    for clade, genotype_indices in clade_to_genotype_indices.items():
        for i, genotype_index in enumerate(genotype_indices):
            genotype_colors[genotype_index] = adjust_lightness(clade_colors[clade], 0.05 * (i - len(genotype_indices) // 2))

    return allele_colors, genotype_colors, clade_colors


def get_annotation_positions(max_gen, max_freq, plotted_positions, plot_dot_for_WT_locus=True, initial_y=1.15, y_step=0.05, x_collision_threshold=100, y_collision_threshold=0.1):

    if plot_dot_for_WT_locus:
        x = max(0, max_gen - 30)
    else:
        x = max(0, max_gen - 14)

    y = initial_y
    for (px, py) in plotted_positions.values():
        if abs(px - x) <= x_collision_threshold:
            while abs(py - y) <= y_collision_threshold:
                y += y_step

    return x, y


def plot_figure_reconstruction_example(simulation, reconstruction, nRow=3, nCol=1, hspace=0.2, hspace_annotate_together=0.4, alpha=0.6, xlim=GENERATION_XLIM, ylim=FREQUENCY_YLIM, alpha_for_stackplot=0.8, ylim_for_stackplot=[-0.03, 1.02], use_color_palette_accordingly=True, genotype_color_indices_to_skip=None, plot_single_column=False, annotate_together=False, annotation_ys=None, annotate_genotype_sequence=True, annotation_line_size=SIZELINE * 1.2, annotation_line_offset=0.04, min_max_freq_to_annotate=0.3, plot_dot_for_WT_locus=True, add_background_behind_annotation=False, background_color='#f2f2f2', save_file=None):

    times = simulation['times']
    T, L, N, genotypes, genotype_frequencies, pop = parse_info_from_simulation(simulation)
    genotype_index_to_clade, clade_to_genotype_indices = map_genotype_index_to_clade(genotypes, reconstruction)

    if use_color_palette_accordingly:
        allele_colors, genotype_colors, clade_colors = get_colors_accordingly(genotypes, genotype_frequencies, clade_to_genotype_indices, reconstruction, genotype_color_indices_to_skip=genotype_color_indices_to_skip)
    else:
        clade_colors = CLADE_COLORS
        allele_colors = ALLELE_COLORS if USE_COLOR_PALETTE_FROM_SEABORN else ALLELE_CMAP(np.linspace(0, 0.8, L))

    genotype_frequencies_for_stackplot, genotype_colors_for_stackplot, genotype_indices_for_stackplot = get_genotype_frequencies_colors_for_stackplot(genotype_frequencies, genotype_colors, clade_to_genotype_indices, clade_colors)

    w = SINGLE_COLUMN if plot_single_column else DOUBLE_COLUMN
    goldh = w / GOLD_RATIO

    gs_base = plt.GridSpec(3, 1, hspace=hspace_annotate_together if annotate_together else hspace)
    fig = plt.figure(figsize=(w, goldh))
    # fig, axes = plt.subplots(nRow, nCol, figsize=(w, goldh))
    ylabel = 'Frequency'
    xlabel = 'Generation'
    size_annotation = SIZEANNOTATION_SINGLE if plot_single_column else SIZEANNOTATION
    sublabel_x, sublabel_y = -0.115, 1.05

    plt.sca(fig.add_subplot(gs_top[0,:]))
    traj = simulation['traj']
    ax = plt.gca()
    AP.plotTraj(traj, linewidth=SIZELINE, colors=allele_colors, alpha=alpha, plotFigure=False, plotShow=False, title=None)
    set_ticks_labels_axes(xlim=xlim, ylim=ylim, xticks=[], xticklabels=[], ylabel=ylabel)
    plt.text(x=sublabel_x, y=sublabel_y, s=SUBLABELS[0], transform=ax.transAxes, **DEF_SUBLABELPROPS)

    bottom_axes = [fig.add_subplot(gs_base[1, :]), fig.add_subplot(gs_base[2, :])]

    plt.sca(bottom_axes[1 if annotate_together else 0])
    traj = simulation['traj']
    ax = plt.gca()
    AP.plotTraj(genotype_frequencies.T, linewidth=SIZELINE, linestyle='dashed', alpha=alpha, colors=genotype_colors, plotFigure=False, plotShow=False, title=None)
    AP.plotTraj(reconstruction.cladeFreqWithAncestor, linewidth=SIZELINE * 2, alpha=alpha, colors=clade_colors, plotFigure=False, plotShow=False, title=None)
    plt.plot([0, 0.0001], [-1, -1], linewidth=SIZELINE, linestyle='dashed', color=GREY_COLOR_RGB, alpha=alpha, label='Genotype')
    plt.plot([0, 0.0001], [-1, -1], linewidth=SIZELINE * 2, color=GREY_COLOR_RGB, alpha=alpha, label='Clade')
    if annotate_together:
        ax.legend(fontsize=SIZELEGEND * 0.7, loc='center left', bbox_to_anchor=(-0.01, 0.45), frameon=False)
        plt.text(x=sublabel_x, y=sublabel_y, s=SUBLABELS[2], transform=ax.transAxes, **DEF_SUBLABELPROPS)
    else:
        ax.legend(fontsize=SIZELEGEND * 0.7, loc='center left', bbox_to_anchor=(-0.01, 0.45), frameon=False)
        plt.text(x=sublabel_x, y=sublabel_y, s=SUBLABELS[1], transform=ax.transAxes, **DEF_SUBLABELPROPS)

    plotted_positions = {}
    if annotate_together:
        count = 0
        for i, genotype in enumerate(genotypes):
            if np.all(genotype < 1) or np.max(genotype_frequencies[i]) < min_max_freq_to_annotate:
                continue  # Wild-type or frequency remains small

            max_freq = np.max(genotype_frequencies[i])
            max_gen = np.argmax(genotype_frequencies[i])
            if annotation_ys is None:
                x, y = get_annotation_positions(max_gen, max_freq, plotted_positions, plot_dot_for_WT_locus=plot_dot_for_WT_locus, initial_y=1.15)
            else:
                x = max(0, max_gen - 14)
                y = annotation_ys[count]
            plotted_positions[i] = (max_gen, y)

            plt.plot([max_gen, max_gen], [y - annotation_line_offset, max_freq + annotation_line_offset], linewidth=annotation_line_size, linestyle='dotted', color=genotype_colors[i], alpha=1, clip_on=False)

            plot_genotype_annotation(x, y, genotype, max_freq, allele_colors, min_max_freq_to_annotate=min_max_freq_to_annotate, fontsize=size_annotation, plot_dot_for_WT_locus=plot_dot_for_WT_locus, plot_single_column=plot_single_column, add_background_behind_annotation=add_background_behind_annotation, background_color=background_color)
            count += 1

    elif annotate_genotype_sequence:
        for i, genotype in enumerate(genotypes):
            if np.all(genotype < 1) or np.max(genotype_frequencies[i]) < min_max_freq_to_annotate:
                continue  # Wild-type or frequency remains small

            max_freq = np.max(genotype_frequencies[i])
            max_gen = np.argmax(genotype_frequencies[i])
            x, y = get_annotation_positions(max_gen, max_freq, plotted_positions, plot_dot_for_WT_locus=plot_dot_for_WT_locus, initial_y=min(1.1, max_freq + 0.2), y_step=0.15)
            plotted_positions[i] = (max_gen, y)

            plt.plot([max_gen, max_gen], [max_freq + annotation_line_offset, y - annotation_line_offset], linewidth=annotation_line_size, linestyle='dotted', color=genotype_colors[i], alpha=1, clip_on=False)

            plot_genotype_annotation(x, y, genotype, max_freq, allele_colors, min_max_freq_to_annotate=min_max_freq_to_annotate, fontsize=size_annotation, plot_dot_for_WT_locus=plot_dot_for_WT_locus, plot_single_column=plot_single_column, add_background_behind_annotation=add_background_behind_annotation, background_color=background_color)

    set_ticks_labels_axes(xlim=xlim, xlabel=xlabel if annotate_together else None, ylim=ylim, xticks=[] if not annotate_together else None, xticklabels=[] if not annotate_together else None, ylabel=ylabel)

    plt.sca(bottom_axes[0 if annotate_together else 1])
    ax = plt.gca()
    plt.stackplot(times, genotype_frequencies_for_stackplot, colors=genotype_colors_for_stackplot, alpha=alpha_for_stackplot)
    if annotate_together:
        plt.text(x=sublabel_x, y=sublabel_y, s=SUBLABELS[1], transform=ax.transAxes, **DEF_SUBLABELPROPS)
        genotype_frequencies_for_stackplot_without_divider = genotype_frequencies_for_stackplot[::2]
        for i, frequencies in enumerate(genotype_frequencies_for_stackplot_without_divider):
            index = genotype_indices_for_stackplot[i]
            genotype = genotypes[index]
            max_freq = np.max(genotype_frequencies[index])
            if np.all(genotype < 1) or max_freq < min_max_freq_to_annotate:
                continue  # Wild-type or frequency remains small

            max_gen = plotted_positions[index][0]
            y_annotation = -(hspace_annotate_together - (plotted_positions[index][1] - 1)) + annotation_line_offset - 0.025
            y_up = np.sum(genotype_frequencies_for_stackplot_without_divider[:i, max_gen])
            y_down = np.sum(genotype_frequencies_for_stackplot_without_divider[:i+1, max_gen])
            y = (y_up + y_down) / 2

            plt.plot([max_gen, max_gen], [y, y_annotation], linewidth=annotation_line_size, linestyle='dotted', color=genotype_colors[index], alpha=1, clip_on=False)
    elif annotate_genotype_sequence:
        genotype_frequencies_for_stackplot_without_divider = genotype_frequencies_for_stackplot[::2]
        for i, frequencies in enumerate(genotype_frequencies_for_stackplot_without_divider):
            index = genotype_indices_for_stackplot[i]
            genotype = genotypes[index]
            max_freq = np.max(genotype_frequencies[index])
            if np.all(genotype < 1) or max_freq < min_max_freq_to_annotate:
                continue  # Wild-type or frequency remains small

            x = max(0, np.argmax(frequencies) - 30)
            y_up = np.sum(genotype_frequencies_for_stackplot_without_divider[:i, x])
            y_down = np.sum(genotype_frequencies_for_stackplot_without_divider[:i+1, x])
            y = (y_up + y_down) / 2

            plot_genotype_annotation(x, y, genotype, max_freq, allele_colors, min_max_freq_to_annotate=min_max_freq_to_annotate, fontsize=size_annotation, plot_dot_for_WT_locus=plot_dot_for_WT_locus, plot_single_column=plot_single_column, add_background_behind_annotation=add_background_behind_annotation, background_color=background_color)

    set_ticks_labels_axes(xlim=xlim, ylim=ylim_for_stackplot, ylabel=ylabel, xlabel=xlabel if not annotate_together else None, xticks=[] if annotate_together else None, xticklabels=[] if annotate_together else None)

    if plot_single_column:
        plt.subplots_adjust(0.11, 0.13, 0.92, 0.95)
    else:
        plt.subplots_adjust(0.055, 0.06, 0.93, 0.99)
    plt.show()
    if save_file:
        fig.savefig(save_file, facecolor=fig.get_facecolor(), **DEF_FIGPROPS)


def plot_figure_performance_example(simulation, reconstruction, evaluation, nRow=3, nCol=2, plot_genotype_fitness=True, threshold=0, compare_with_true=False, compare_with_SL=False, plot_true_cov=False, ylim=(0.95, 1.25), yticks=(1.0, 1.1, 1.2), alpha=0.6, title_pad=4, save_file=None):

    T, L, N, genotypes, genotype_frequencies, pop = parse_info_from_simulation(simulation)

    w = SINGLE_COLUMN
    goldh = w * 1 if nRow == 2 else w * 1.3
    fig, axes = plt.subplots(nRow, nCol, figsize=(w, goldh))
    sublabel_x, sublabel_y = -0.25, 1.08

    if nRow == 2:
        heatmaps = [axes[0, 0], axes[0, 1], axes[1, 0]]
    else:
        heatmaps = [axes[0, 0], axes[1, 0], axes[2, 0]]
    cov_list = [reconstruction.intCov, reconstruction.recoveredIntCov, reconstruction.recoveredIntCov - reconstruction.intCov]
    # title_list = [r"True covariance, $C$", r"Recovered covariance, $\hat{C}$", "Error of recovered\ncovariance, $\hat{C}-C$"]
    title_list = [r"True covariance, $C$", r"Recovered covariance, $\hat{C}$", "Error matrix, $\hat{C}-C$"]
    vmin = min(np.min(cov_list[0]), np.min(cov_list[1]))
    vmax = max(np.max(cov_list[0]), np.max(cov_list[1]))
    cbar_ax = fig.add_axes(rect=[.106, .04, .34, .01])
    cbar_ticks = np.arange(int(vmin/5)*5, int(vmax/5)*5, 50)
    cbar_ticks -= cbar_ticks[np.argmin(np.abs(cbar_ticks))]
    matrix_labels = np.arange(1, L+1, 2)
    matrix_ticks = [l - 0.5 for l in matrix_labels]

    for i, cov in enumerate(cov_list):
        plt.sca(heatmaps[i])
        at_left = (i % nCol == 0) if nRow == 2 else True
        ax = plt.gca()
        plot_cbar = (i == 2)
        sns.heatmap(cov, center=0, vmin=vmin - 5, vmax=vmax + 5, cmap=CMAP, square=True, cbar=plot_cbar, cbar_ax=cbar_ax if plot_cbar else None, cbar_kws=dict(ticks=cbar_ticks, orientation="horizontal", shrink=1))
        plt.title(title_list[i], fontsize=SIZELABEL, pad=title_pad)
        ax.xaxis.set_label_position('top')
        if at_left:
            plt.yticks(ticks=matrix_ticks, labels=matrix_labels, fontsize=SIZELABEL)
            plt.ylabel('Locus index', fontsize=SIZELABEL)
        else:
            plt.yticks(ticks=[], labels=[], fontsize=SIZELABEL)
        plt.xticks(ticks=[], labels=[], fontsize=SIZELABEL)
        if plot_cbar:
            cbar_ax.tick_params(labelsize=SIZELABEL)
        ax.tick_params(**DEF_TICKPROPS)
        plt.text(x=sublabel_x, y=sublabel_y, s=SUBLABELS[i], transform=ax.transAxes, **DEF_SUBLABELPROPS)

    cbar_ax.tick_params(**DEF_TICKPROPS)

    inference = evaluation[1]
    selection_true = inference['true'][2]
    seleciton_recovered = inference['recovered'][2]
    selection_trueCov = inference['true_cov'][2]
    selection_estCov = inference['est_cov'][2]
    selection_SL = inference['SL'][2]
    selection_list = [selection_trueCov, seleciton_recovered, selection_SL]
    xlabel = 'True selections'
    if compare_with_SL:
        ys_list = [selection_trueCov, seleciton_recovered, selection_SL]
        xs_list = [selection_true, selection_true, selection_true]
    else:
        ys_list = [selection_trueCov, seleciton_recovered, seleciton_recovered]
        xs_list = [selection_true, selection_true, selection_trueCov]
    selection_values = np.concatenate([selection_true] + selection_list)
    if ylim is None:
        ylim = [np.min(selection_values) - 0.01, np.max(selection_values) + 0.01]

    if plot_genotype_fitness:
        T, L = simulation['traj'].shape
        N = np.sum(simulation['nVec'][0])
        genotypes, _, _ = AP.getDominantGenotypeTrajectories(simulation, threshold=threshold, T=T, totalPopulation=N)
        fitness_true = RC.computeFitnessOfGenotypes(genotypes, selection_true)
        fitness_estCov = RC.computeFitnessOfGenotypes(genotypes, selection_estCov)
        xlabel, ylabel = 'True genotype fitness', 'Inferred genotype fitness'
        fitness_list = [RC.computeFitnessOfGenotypes(genotypes, selection) for selection in selection_list]
        fitness_trueCov, fitness_recovered, fitness_SL = tuple(fitness_list)
        ys_fitness_list = fitness_list
        # ys_fitness_list = [fitness_trueCov, fitness_recovered, fitness_estCov]
        xs_fitness_list = [fitness_true] * 3
        fitness_values = np.concatenate([fitness_true] + fitness_list)
        if ylim is None:
            ylim = [np.min(fitness_values) - 0.01, np.max(fitness_values) + 0.01]

    if nRow == 2:
        plt.sca(axes[1, 1])
        ax = plt.gca()
        ylabel = 'Inferred selections'
        if plot_genotype_fitness:
            fitness_recovered = RC.computeFitnessOfGenotypes(genotypes, seleciton_recovered)
            plot_comparison(fitness_true, fitness_recovered, xlabel, ylabel, label='Recovered' if plot_true_cov else None, alpha=alpha, ylim=ylim, yticks=yticks, xticks=yticks, plot_title=not plot_true_cov)
        else:
            plot_comparison(selection_true, seleciton_recovered, xlabel, ylabel, label='Recovered' if plot_true_cov else None, alpha=alpha, ylim=ylim, yticks=yticks, xticks=yticks, plot_title=not plot_true_cov)
        if plot_true_cov:
            plt.scatter(selection_true, selection_trueCov, s=SIZEDOT, alpha=alpha, label='True_cov')
            plt.legend(fontsize=SIZELEGEND)
    elif compare_with_true:
        ylabels = ['Selections inferred\nwith true covariance', 'Selections inferred with\nreconstructed covariance', 'Selections inferred when\nignoring linkage']
        for i, ys in enumerate(selection_list):
            ylabel = ylabels[i]
            plt.sca(axes[i, 1])
            plot_comparison(selection_true, ys, xlabel, ylabel, alpha=alpha, ylim=ylim, yticks=yticks, xticks=yticks if i == nRow - 1 else [])
    elif plot_genotype_fitness:
        ylabels = ['Fitness inferred\nwith true covariance', 'Fitness inferred with\nreconstructed covariance', 'Fitness inferred when\nignoring linkage']
        xlabel = 'True fitness'
        for i, (xs, ys) in enumerate(zip(xs_fitness_list, ys_fitness_list)):
            ylabel = ylabels[i]
            plt.sca(axes[i, 1])
            ax = plt.gca()
            plot_comparison(xs, ys, xlabel if i == nRow - 1 else None, ylabel, alpha=alpha, ylim=ylim, yticks=yticks, xticks=yticks, xticklabels=yticks if i == nRow - 1 else [], plot_title=False)
            plt.text(x=0.05, y=0.9, s="Spearman's " + r'$\rho$' + ' = %.2f' % stats.spearmanr(xs, ys)[0], transform=ax.transAxes, fontsize=SIZELABEL)
            plt.text(x=sublabel_x, y=sublabel_y, s=SUBLABELS[3 + i], transform=ax.transAxes, **DEF_SUBLABELPROPS)
    else:
        if compare_with_SL:
            ylabels = ['Selections inferred\nwith true covariance', 'Selections inferred with\nreconstructed covariance', 'Selections inferred when\nignoring covariance']
            xlabels = ['True selections'] * 3
        else:
            ylabels = ['Selections inferred\nwith true covariance', 'Selections inferred with\nreconstructed covariance', 'Selections inferred with\nreconstructed covariance']
            xlabels = ['True selections', 'True selections', 'Selections inferred\nwith true covariance']
        for i, (xs, ys) in enumerate(zip(xs_list, ys_list)):
            ylabel = ylabels[i]
            xlabel = xlabels[i]
            plt.sca(axes[i, 1])
            plot_comparison(xs, ys, xlabel, ylabel, alpha=alpha, ylim=ylim, yticks=yticks, xticks=yticks if i == nRow - 1 else [], title_pad=title_pad)

    ax.tick_params(**DEF_TICKPROPS)
    plt.setp(ax.spines.values(), **DEF_AXPROPS)

    plt.subplots_adjust(0.1, 0.07, 0.98, 0.96, hspace=0.2, wspace=0.5)
    plt.show()

    if save_file:
        fig.savefig(save_file, facecolor=fig.get_facecolor(), **DEF_FIGPROPS)


def plot_comparison(xs, ys, xlabel, ylabel, label=None, annot_texts=None, alpha=0.6, ylim=None, xlim=None, yticks=None, xticks=None, xticklabels=None, plot_title=True, title_pad=4):
    plt.scatter(xs, ys, marker='o', edgecolors='none', s=SIZEDOT, alpha=alpha, label=label)
    if annot_texts is not None:
        for x, y, text in zip(xs, ys, annot_texts):
            plt.annotate(x, y, text)
    if xlim is None:
        xlim = ylim
        plt.plot(ylim, ylim, color=GREY_COLOR_RGB, alpha=alpha, linestyle='dashed')
        plt.gca().set_aspect('equal', adjustable='box')
    set_ticks_labels_axes(xlim=xlim, ylim=ylim, xticks=xticks, yticks=yticks, xticklabels=xticklabels, yticklabels=yticks, xlabel=xlabel, ylabel=ylabel)
    if plot_title:
        plt.title("Spearman's " + r'$\rho$' + ' = %.2f' % stats.spearmanr(xs, ys)[0], fontsize=SIZELABEL, pad=title_pad)


def plot_figure_performance_on_simulated_data_helper(ax, yspine_position=0.05, xspine_position=-0.1):

    ax.tick_params(**DEF_TICKPROPS)
    plt.setp(ax.spines.values(), **DEF_AXPROPS)
    # ax.spines['left'].set_position(('data', yspine_position))
    # ax.spines['bottom'].set_position(('data', xspine_position))
    ax.spines['left'].set_position(('axes', yspine_position))
    ax.spines['bottom'].set_position(('axes', xspine_position))


def plot_figure_performance_on_simulated_data(MAE_cov, Spearmanr_cov, MAE_selection, Spearmanr_selection, MAE_fitness=None, Spearmanr_fitness=None, two_columns=False, plot_legend=False, use_pearsonr=False, evaluate_fitness=False, annot=False, save_file=None):
    """
    Comparisons versus existing methods of covariance recovery and selection inference.
    """

    w       = SINGLE_COLUMN
    hshrink = 1.5
    if MAE_fitness is not None and Spearmanr_fitness is not None:
        if two_columns:
            goldh   = 0.70 * w * hshrink
        else:
            goldh   = 1.10 * w * hshrink
    else:
        goldh   = 0.90 * w * hshrink
    fig     = plt.figure(figsize=(w, goldh))

    box_top   = 0.94
    box_left  = 0.2
    box_right = 0.995 if two_columns else 0.94
    wspace    = 0.6

    if MAE_fitness is not None and Spearmanr_fitness is not None:
        if two_columns:
            ddy = 0.17 / hshrink
            dy = 0.27 / hshrink  # Adjusts height of each subplot, & height of white space below the subplots.
        else:
            ddy = 0.1 / hshrink
            dy = 0.145 / hshrink  # Adjusts height of each subplot, & height of white space below the subplots.
    else:
        ddy = 0.14 / hshrink
        dy = 0.23 / hshrink  # Adjusts height of each subplot, & height of white space below the subplots.

    if MAE_fitness is not None and Spearmanr_fitness is not None:
        metrics_list = [MAE_cov, Spearmanr_cov, MAE_selection, Spearmanr_selection, MAE_fitness, Spearmanr_fitness]
    else:
        metrics_list = [MAE_cov, Spearmanr_cov, MAE_selection, Spearmanr_selection]

    if two_columns:
        boxes = [dict(left=box_left,
                      right=box_right,
                      bottom=box_top-((i+1)*dy)-(i*ddy),
                      top=box_top-(i*dy)-(i*ddy)) for i in range(len(metrics_list)//2)]
        gridspecs = [gridspec.GridSpec(1, 2, wspace=wspace, **box) for box in boxes]
        axes = []
        for _ in gridspecs:
            axes += [plt.subplot(_[0, 1]), plt.subplot(_[0, 0])]
    else:
        boxes = [dict(left=box_left,
                      right=box_right,
                      bottom=box_top-((i+1)*dy)-(i*ddy),
                      top=box_top-(i*dy)-(i*ddy)) for i in range(len(metrics_list))]
        gridspecs = [gridspec.GridSpec(1, 1, wspace=wspace, **box) for box in boxes]
        axes = [plt.subplot(gridspec[0, 0]) for gridspec in gridspecs]

    if MAE_fitness is not None and Spearmanr_fitness is not None:
        ylim_list = [[0, 4.4], [0, 1.1], [0, 0.044], [0, 1.1], [0, 0.088], [0, 1.1]]
        yticks_list = [[0, 2, 4], [0, 0.5, 1], [0, 0.02, 0.04], [0, 0.5, 1], [0, 0.04, 0.08], [0, 0.5, 1]]
        yticklabels_list = [['0', '2', r'$\geq 4$'], ['0', '0.5', '1'], ['0', '0.02', '0.04'], ['0', '0.5', '1'], ['0', '0.04', '0.08'], ['0', '0.5', '1']]
        ceil_list = [4, None, 0.04, None, 0.08, None]
        floor_list = [None, None, None, None, None, None]
    else:
        ylim_list = [[0, 4.4], [0, 1.1], [0, 0.088] if evaluate_fitness else [0, 0.044], [0, 1.1]]
        yticks_list = [[0, 2, 4], [0, 0.5, 1], [0, 0.04, 0.08] if evaluate_fitness else [0, 0.02, 0.04], [0, 0.5, 1]]
        yticklabels_list = [['0', '2', r'$\geq 4$'], ['0', '0.5', '1'], ['0', '0.04', '0.08'] if evaluate_fitness else ['0', '0.02', '0.04'], ['0', '0.5', '1']]
        ceil_list = [4, None, 0.08 if evaluate_fitness else 0.04, None]
        floor_list = [None, None, None, None]
    if MAE_fitness is not None and Spearmanr_fitness is not None:
        if two_columns:
            ylabel_list = ['MAE of recovered\ncovariances', LABEL_SPEARMANR_COVARIANCE_FOUR_2, 'MAE of inferred\nselection coefficients', LABEL_SPEARMANR_FOUR_2, 'MAE of inferred\ngenotype fitness', LABEL_SPEARMANR_FITNESS_FOUR_2]
        else:
            ylabel_list = ['MAE of recovered\ncovariances', LABEL_SPEARMANR_COVARIANCE_FOUR, 'MAE of inferred\nselection coefficients', LABEL_SPEARMANR_FOUR, 'MAE of inferred\ngenotype fitness', LABEL_SPEARMANR_FITNESS_FOUR]
    else:
        if use_pearsonr:
            if evaluate_fitness:
                ylabel_list = ['MAE of recovered\ncovariances', LABEL_PEARSONR_COVARIANCE_FOUR, 'MAE of inferred\ngenotype fitness', LABEL_PEARSONR_FITNESS_FOUR]
            else:
                ylabel_list = ['MAE of recovered\ncovariances', LABEL_PEARSONR_COVARIANCE_FOUR, 'MAE of inferred\nselection coefficients', LABEL_PEARSONR_FOUR]
        else:
            if evaluate_fitness:
                ylabel_list = ['MAE of recovered\ncovariances', LABEL_SPEARMANR_COVARIANCE_FOUR, 'MAE of inferred\ngenotype fitness', LABEL_SPEARMANR_FITNESS_FOUR]
            else:
                ylabel_list = ['MAE of recovered\ncovariances', LABEL_SPEARMANR_COVARIANCE_FOUR, 'MAE of inferred\nselection coefficients', LABEL_SPEARMANR_FOUR]

    method_list = METHODS
    xs = np.arange(0, len(method_list))
    xlim = [-0.75, 4.4] if two_columns else [-0.8, 4.4]
    sublabel_x, sublabel_y = -0.6, 1.15

    ## set colors and methods list

    fc        = '#ff6666'  #'#EB4025'
    ffc       = '#ff6666'  #'#EB4025'
    hc        = '#FFB511'
    nc        = '#E8E8E8'
    hfc       = '#ffcd5e'
    nfc       = '#f0f0f0'
    methods   = METHODS
    xticklabels = methods
    # xticklabels = ['0', '1', '2', '3', '4' ] if two_columns else METHODS
    colorlist   = [   fc,    hc,    nc,      nc,      nc,         nc,      nc,    nc]
    fclist      = [  ffc,   hfc,   nfc,     nfc,     nfc,        nfc,     nfc,   nfc]
    eclist      = [BKCOLOR for k in range(len(methods))]

    hist_props = dict(lw=SIZELINE/2, width=0.5, align='center', orientation='vertical',
                      edgecolor=[BKCOLOR for i in range(len(methods))])

    for row, metrics in enumerate(metrics_list):
        ylim = ylim_list[row]
        yticks, yticklabels, ylabel = yticks_list[row], yticklabels_list[row], ylabel_list[row]
        floor, ceil = floor_list[row], ceil_list[row]

        ys = [metrics[method] for method in method_list]
        y_avgs = np.mean(ys, axis=1)
        ax = axes[row]

        if row == 1:
            scatter_indices = np.array([_ for _ in range(len(ys)) if _ not in [0, 4]])  # Spearmanr of covariances does not apply to the SL/MPL method
            # scatter_indices = np.arange(len(ys))
            plt.sca(ax)
            na_x, na_y = 3.6, 0.15
            plt.plot([4, 4], [0, na_y - 0.05], linewidth=SIZELINE, color=BKCOLOR)
            plt.text(na_x, na_y, 'NA', fontsize=SIZESUBLABEL)

            na_x, na_y = -0.4, 0.15
            plt.plot([0, 0], [0, na_y - 0.05], linewidth=SIZELINE, color=BKCOLOR)
            plt.text(na_x, na_y, 'NA', fontsize=SIZESUBLABEL)

            y_avgs[0] = None
            y_avgs[4] = None
        elif row == 0:
            scatter_indices = np.array([_ for _ in range(len(ys)) if _ not in [0, 4]])
            plt.sca(ax)

            na_x, na_y = 3.6, 0.6
            plt.plot([4, 4], [0, na_y - 0.2], linewidth=SIZELINE, color=BKCOLOR)
            plt.text(na_x, na_y, 'NA', fontsize=SIZESUBLABEL)

            na_x, na_y = -0.4, 0.6
            plt.plot([0, 0], [0, na_y - 0.2], linewidth=SIZELINE, color=BKCOLOR)
            plt.text(na_x, na_y, 'NA', fontsize=SIZESUBLABEL)
            y_avgs[0] = None
            y_avgs[4] = None
        else:
            scatter_indices = np.arange(0, len(ys))

        pprops = { 'colors':      [colorlist],
                   'xlim':        deepcopy(xlim),
                   'ylim':        ylim,
                   'xticks':      xs,
                   'xticklabels': [] if two_columns and row < 4 else xticklabels,
                   'yticks':      [],
                   'theme':       'open',
                   'hide':        ['left','right'] }

        pprops['yticks'] = yticks
        pprops['yticklabels'] = yticklabels
        pprops['ylabel'] = ylabel
        pprops['hide']   = []

        mp.plot(type='bar', ax=ax, x=[xs], y=[y_avgs], plotprops=hist_props, **pprops)
        if annot and (row == 3 or row == 2):
            for x, y in zip(xs, y_avgs):
                plt.sca(ax)
                annotation = '%.2f' % y if row == 3 else ('%.3f' % y)
                plt.text(x - 0.3, min(y * 1.2, ceil * 1.2) if ceil is not None else y * 1.2, annotation, fontsize=SIZESUBLABEL)

        del pprops['colors']

        ys = np.array(ys)
        if floor is not None:
            ys[ys <= floor] = floor
        if ceil is not None:
            ys[ys >= ceil] = ceil

        pprops['facecolor'] = ['None' for _c1 in range(len(xs))]
        pprops['edgecolor'] = [eclist[_] for _ in scatter_indices]
        # size_of_point = 2.
        size_of_point = 1.5 if two_columns else 4.
        sprops = dict(lw=AXWIDTH, s=size_of_point, marker='o', alpha=1.0)
        temp_x = [[xs[_c1] + np.random.normal(0, 0.04) for _c2 in range(len(ys[_c1]))] for _c1 in scatter_indices]
        mp.scatter(ax=ax, x=temp_x, y=ys[scatter_indices], plotprops=sprops, **pprops)

        pprops['facecolor'] = [fclist[_] for _ in scatter_indices]
        sprops = dict(lw=0, s=size_of_point, marker='o', alpha=1)
        mp.scatter(ax=ax, x=temp_x, y=ys[scatter_indices], plotprops=sprops, **pprops)
        if row % 2 > 0:
            plt.sca(ax)
            plt.ylabel(ylabel, labelpad=1)
        plot_figure_performance_on_simulated_data_helper(ax)
        if two_columns and row >= 4:
            plt.sca(ax)
            # plt.xticks(ticks=xs, labels=xticklabels, rotation=38)
            plt.xticks(ticks=xs, labels=xticklabels, rotation=90)
        if two_columns and row % 2 == 1:
            plt.text(x=sublabel_x, y=sublabel_y, s=SUBLABELS[row // 2], transform=ax.transAxes, **DEF_SUBLABELPROPS)

    if two_columns and plot_legend:
        legend_dy =  1.3
        legend_l  =  2  # num of lines
        legend_x  = -7.4
        legend_y  = -24.2
        legend_dx =  3.7
        legend_d  = -0.5
        for k in range(len(methods)):
            axes[0].text(legend_x + legend_d + (k//legend_l * legend_dx), legend_y - (k%legend_l * legend_dy), labels[k], ha='center', va='center', **DEF_LABELPROPS)
            axes[0].text(legend_x + (k//legend_l * legend_dx), legend_y - (k%legend_l * legend_dy), methods[k], ha='left', va='center', **DEF_LABELPROPS)

    plt.show()
    if save_file:
        fig.savefig(save_file, facecolor=fig.get_facecolor(), **DEF_FIGPROPS)


def plot_figure_clusterization_example_LTEE(pop, clusterization, save_file=None):
    w = SINGLE_COLUMN
    fig = plt.figure(figsize=(w, w))
    ax = plt.gca()
    segmented_int_D, groups_sorted = clusterization['segmentedIntDxdx']
    L = len(segmented_int_D)
    for l in range(L):
        segmented_int_D[l, l] = 0
    # groups = clusterization['groups']
    ticks, ylabels, group_sizes = get_ticks_and_labels_for_clusterization(groups_sorted)
    vmin = np.min(segmented_int_D)
    vmax = np.max(segmented_int_D)
    sns.heatmap(segmented_int_D, center=0, vmin=vmin, vmax=vmax, cmap=CMAP, square=True, cbar=False)
    plt.xticks(ticks=ticks[:-1])
    plt.yticks(ticks=ticks[:-1])

    plot_ticks_and_labels_for_clusterization(ticks, ylabels, group_sizes)
    ax.tick_params(**DEF_TICKPROPS)
    plt.setp(ax.spines.values(), **DEF_AXPROPS)
    plt.title(f'Segmented Matrix ' + r'$D$' + f' for population {pop}', fontsize=SIZELABEL)
    plt.show()
    if save_file:
        fig.savefig(save_file, facecolor=fig.get_facecolor(), **DEF_FIGPROPS)


def get_ticks_and_labels_for_clusterization(groups_sorted, name='Group', note_size=False):
    group_sizes = [int(len(group)) for group in groups_sorted]
    # if group_sizes[0] == 0:
    #     group_sizes = group_sizes[1:]
    ticks = [np.sum(group_sizes[:i]) for i in range(2, len(group_sizes) + 1)]
    labels = [f'{name} {i}' for i in range(1, len(groups_sorted))]
    if len(groups_sorted[0]) > 0:
        labels = ['Shared'] + labels
        ticks = [group_sizes[0]] + ticks
    size_index_offset = (len(groups_sorted[0])==0)
    if note_size:
        labels = [f'{_} ({group_sizes[i+size_index_offset]})' for i, _ in enumerate(labels)]
    return ticks, labels, group_sizes


def plot_ticks_and_labels_for_clusterization(ticks, ylabels, group_sizes, ylabel_x=-0.15):
    ax = plt.gca()
    if group_sizes[0] == 0:
        group_sizes = group_sizes[1:]
    L = np.sum(group_sizes)
    for g, size in enumerate(group_sizes):
        tick = ticks[g]
        ylabel_y = 0.99 - (tick - size / 2) / L
        ax.text(ylabel_x, ylabel_y, ylabels[g], transform=ax.transAxes, fontsize=SIZELABEL)


def plot_figure_clusterizations_LTEE(populations, clusterizations):
    pass


def get_axes_for_reconstruction_example_LTEE():
    ratio   = 0.33
    w       = DOUBLE_COLUMN
    h       = ratio * w
    fig     = plt.figure(figsize=(w, h))

    divider_width = 0.1
    box_left = 0.065
    # box_bottom = 0.14
    box_top = 0.95
    ddy = 0.07  # Adjust hspace
    dy = 0.37  # Adjusts height of each subplot. Also adjusts height of white space below the subplots.
    # Height of heatmap = cov_top - cov_bottom
    # Width of heatmap = (cov_top - cov_bottom) * (h / w)

    box_bottom = box_top - (2*dy)-(1*ddy)

    cov_right = 0.99
    cov_bottom = box_bottom
    cov_top = box_top

    cbar_width = 0.1
    cov_width = ratio * (cov_top - cov_bottom) + cbar_width
    box_middle = cov_right - divider_width - cov_width
    cov_left = box_middle + divider_width

    traj_boxes = [dict(left=box_left,
                  right=box_middle,
                  bottom=box_top-((i+1)*dy)-(i*ddy),
                  top=box_top-(i*dy)-(i*ddy)) for i in range(2)]
    cov_box = dict(left=cov_left,
                   right=cov_right,
                   bottom=cov_bottom,
                   top=cov_top)

    boxes = traj_boxes + [cov_box]
    gridspecs = [gridspec.GridSpec(1, 1, **boxes[i]) for i in range(len(boxes))]
    axes = [plt.subplot(gridspecs[i][0, 0]) for i in range(len(gridspecs))]
    return fig, axes


def plot_figure_reconstruction_example_LTEE(pop, reconstruction, data, alpha=0.4, plotIntpl=True, ylim=(-0.03, 1.03), bbox_to_anchor_legend=(-0.005, 0.5), annotation_line_size=SIZEANNOTATION, legend_frameon=False, plot_legend_out_on_the_right=True, save_file=None):

    if False:
        pass
        # nRow, nCol = 7, 5
        # divider_rowspan = 1
        # fig = plt.figure(figsize=(w, w))
        #
        # ax1_rowspan = int((nRow - num_divider * divider_rowspan) / (2 * ax2_ax1_rowspan_ratio + 1))
        # ax2_rowspan = int(ax1_rowspan * ax2_ax1_rowspan_ratio)
        # ax2_colspan = nCol//2 - 1
        #
        # ax1 = plt.subplot2grid((nRow, nCol), (0, 0), rowspan=ax1_rowspan, colspan=nCol)
        # ax2 = plt.subplot2grid((nRow, nCol), (ax1_rowspan + divider_rowspan, 0), rowspan=ax2_rowspan, colspan=ax2_colspan)
        # ax3 = plt.subplot2grid((nRow, nCol), (ax1_rowspan + divider_rowspan, nCol//2 + 1), rowspan=ax2_rowspan, colspan=ax2_colspan)
        # ax4 = plt.subplot2grid((nRow, nCol), (ax1_rowspan + ax2_rowspan + divider_rowspan, 0), rowspan=ax2_rowspan, colspan=ax2_colspan)
        # res = reconstructions[pop]
        # alpha_ = alpha if len(data[pop]['sites_intpl']) <= 1000 else 0.1

    # w = DOUBLE_COLUMN
    # fig, axes = plt.subplots(2, 1, figsize=(w, 0.3 * w))
    fig, axes = get_axes_for_reconstruction_example_LTEE()
    xlabel, ylabel = 'Generation', 'Frequency'
    xticks = range(0, 60001, 10000)
    tStart, tEnd = reconstruction.times[0], reconstruction.times[-1]
    if pop in EXCHANGE_FIRST_TWO_CLADES_LTEE and EXCHANGE_FIRST_TWO_CLADES_LTEE[pop]:
        clade_colors = {
            0: LTEE_EXTINCT_COLOR,
            1: LTEE_MINOR_FIXED_COLOR,
            2: LTEE_MAJOR_FIXED_COLOR,
            3: '#64af31',
            4: '#cd7af5',
        }
    else:
        clade_colors = {
            0: LTEE_EXTINCT_COLOR,
            1: LTEE_MAJOR_FIXED_COLOR,
            2: LTEE_MINOR_FIXED_COLOR,
            3: '#64af31',
            4: '#cd7af5',
        }

    if plot_legend_out_on_the_right:
        if reconstruction.numClades >= 4:
            bbox_to_anchor_legend = (1, 0.45)
        else:
            bbox_to_anchor_legend = (1, 0.5)
        xlim = (tStart - 500, tEnd + 500)
        sublabel_x, sublabel_y = -0.112, 0.957
    else:
        xlim = (tStart - 2500, tEnd + 500)
        sublabel_x, sublabel_y = -0.05, 1

    plt.sca(axes[0])
    ax = plt.gca()
    AP.plotTotalCladeFreq(reconstruction.cladeFreqWithAncestor, cladeMuts=reconstruction.cladeMuts, otherMuts=reconstruction.otherMuts, colorAncestorFixedRed=True, traj=reconstruction.traj, times=LH.TIMES_INTPL, colors=clade_colors, alpha=alpha, plotFigure=False, plotLegend=False, plotClade=False, plotShow=False, fontsize=SIZELABEL, linewidth=SIZELINE, legendsize=SIZELEGEND)
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.xticks(ticks=xticks, labels=[])

    labels = [f'Clade {c+1}' for c in range(reconstruction.numClades)]
    handles = [matplotlib.lines.Line2D([], [], color=LTEE_ANCESTOR_FIXED_COLOR, label='Ancestor')] + [matplotlib.lines.Line2D([], [], color=LTEE_MAJOR_FIXED_COLOR, label=labels[0]),
     matplotlib.lines.Line2D([], [], color=LTEE_MINOR_FIXED_COLOR, label=labels[1])] + [matplotlib.lines.Line2D([], [], color=clade_colors[c + 1], label=labels[c]) for c in range(2, reconstruction.numClades)] + [matplotlib.lines.Line2D([], [], color=LTEE_EXTINCT_COLOR, label='Extinct')]
    plt.legend(handles=handles, fontsize=SIZELEGEND, loc='center left', bbox_to_anchor=bbox_to_anchor_legend, frameon=legend_frameon)
    ax.tick_params(**DEF_TICKPROPS)
    plt.ylabel(ylabel, fontsize=SIZELABEL)
    plt.setp(ax.spines.values(), **DEF_AXPROPS)
    plt.text(x=sublabel_x, y=sublabel_y, s=SUBLABELS[0], transform=ax.transAxes, **DEF_SUBLABELPROPS)
    # plt.title(f"Population {pop}", fontsize=SIZELABEL)

    plt.sca(axes[1])
    ax = plt.gca()
    plotted_clades = set()
    if plotIntpl:
        sites = data[pop]['sites_intpl']
    else:
        sites = data[pop]['sites']
    for l, site in enumerate(sites):
        if plotIntpl:
            times = LH.TIMES_INTPL
            freqs = data[pop]['traj'][:, l]
        else:
            times = data[pop]['times'][l]
            freqs = data[pop]['freqs'][l]
        c = data[pop]['site_to_clade'][site]
        if c == 6:
            c = 3
        plt.plot(times, freqs, linewidth=SIZELINE * 0.5, alpha=alpha, color=LH.COLORS[c])

    if pop in LH.populations_nonclonal:
        handles = [matplotlib.lines.Line2D([], [], color=LTEE_ANCESTOR_FIXED_COLOR, label='Ancestor'),
                   matplotlib.lines.Line2D([], [], color=LTEE_ANCESTOR_POLYMORPHIC_COLOR, label='Ancestor\npolymorphic'),
                   matplotlib.lines.Line2D([], [], color=LTEE_EXTINCT_COLOR, label='Extinct')]
    else:
        handles = [matplotlib.lines.Line2D([], [], color=LTEE_ANCESTOR_FIXED_COLOR, label='Ancestor'),
                   matplotlib.lines.Line2D([], [], color=LTEE_MAJOR_FIXED_COLOR, label='Major'),
                   matplotlib.lines.Line2D([], [], color=LTEE_MINOR_FIXED_COLOR, label='Minor'),
                   matplotlib.lines.Line2D([], [], color=LTEE_EXTINCT_COLOR, label='Extinct')]
    plt.legend(handles=handles, fontsize=SIZELEGEND, loc='center left', bbox_to_anchor=bbox_to_anchor_legend, frameon=legend_frameon)
    plt.xlabel(xlabel, fontsize=SIZELABEL)
    plt.ylabel(ylabel, fontsize=SIZELABEL)
    plt.xlim(xlim)
    plt.ylim(ylim)
    ax.tick_params(**DEF_TICKPROPS)
    plt.setp(ax.spines.values(), **DEF_AXPROPS)
    plt.text(x=sublabel_x, y=sublabel_y, s=SUBLABELS[1], transform=ax.transAxes, **DEF_SUBLABELPROPS)

    plt.sca(axes[2])
    ax = plt.gca()
    segmentedIntDxdx, groups = LH.load_dxdx_for_LTEE(pop)
    plot_dxdx_heatmap(segmentedIntDxdx, groups, as_subplot=True)
    plt.setp(ax.spines.values(), **DEF_AXPROPS)
    plt.text(x=-0.07, y=0.97, s=SUBLABELS[2], transform=ax.transAxes, **DEF_SUBLABELPROPS)

    # if legend_frameon:
    #     plt.subplots_adjust(0.06, 0.15, 0.83, 0.96)
    # else:
    #     plt.subplots_adjust(0.06, 0.15, 0.99, 0.96)
    plt.show()
    if save_file:
        fig.savefig(save_file, facecolor=fig.get_facecolor(), **DEF_FIGPROPS)


def plot_dxdx_heatmap(dxdx, groups, as_subplot=False, ylabel_x=-0.21, alpha=0.5, grid_line_offset=0, cbar_shrink=1, xtick_distance=40, figsize=(4, 3), rasterized=True, save_file=None):

    L = len(dxdx)
    if L > 1000:
        xtick_distance = 400
    if not as_subplot:
        fig = plt.figure(figsize=figsize)
    ax = plt.gca()
    for l in range(L):
        dxdx[l, l] = 0
    heatmap = sns.heatmap(dxdx, center=0, cmap=CMAP, square=True, cbar=True, cbar_kws={"shrink": cbar_shrink}, rasterized=rasterized)
    cbar = heatmap.collections[0].colorbar
    # cbar.ax.tick_params(labelsize=SIZELABEL, length=2, color=)
    cbar.ax.tick_params(**DEF_TICKPROPS_COLORBAR)
    xticklabels = np.arange(0, L + xtick_distance // 2, xtick_distance)
    xticks = [l for l in xticklabels]
    ticks, ylabels, group_sizes = get_ticks_and_labels_for_clusterization(groups, name='Group', note_size=True)
    # plot_ticks_and_labels_for_clusterization(ticks, ylabels, group_sizes, ylabel_x=ylabel_x)
    ax.hlines([_ + grid_line_offset for _ in ticks], *ax.get_xlim(), color=GREY_COLOR_HEX, alpha=alpha, linewidth=SIZELINE * 1.2)
    ax.vlines([_ + grid_line_offset for _ in ticks], *ax.get_ylim(), color=GREY_COLOR_HEX, alpha=alpha, linewidth=SIZELINE * 1.2)
    set_ticks_labels_axes(yticks=[], yticklabels=[], xticks=xticks, xticklabels=xticklabels)
    plt.xlabel('Locus index', fontsize=SIZELABEL)
    if not as_subplot:
        plt.show()
    if not as_subplot and save_file:
        fig.savefig(save_file, facecolor=fig.get_facecolor(), **DEF_FIGPROPS)


def plot_identity_helper(pop, reconstruction, data, square=True, cmap=CMAP, reduce_vmax=False, plot_xticks=True, plot_yticks=True, xticklabels_rotation=0, yticklabels_rotation=0):
    pop_to_vmax = {
        'm1': 1000, 'm2': 1000, 'm4': 600, 'm5': 90, 'm6': 60, 'p1': 100, 'p3': 500, 'p5': 60
    }

    ax = plt.gca()
    num_clades = reconstruction.numClades
    num_clades_LTEE = 2
    clade_LTEE_to_identity_index = get_mapping_clade_LTEE_to_identity_index()
    yticks = np.arange(0, num_clades + 1) + 0.5
    yticklabels_complete = [f'Clade {k}' for k in range(1, num_clades + 1)] + ['Other']
    xticks = np.arange(0, num_clades_LTEE + 1) + 0.5
    # xticklabels = ['Major clade', 'Minor clade', 'Ancestral, Basal & Extinct']
    xticklabels = ['Major', 'Minor', 'Other']
    identity_counts = np.zeros((num_clades + 1, num_clades_LTEE + 1))
    for k, muts in enumerate(reconstruction.cladeMuts + [reconstruction.otherMuts]):
        for index in muts:
            site = data[pop]['sites_intpl'][index]
            clade_LTEE = data[pop]['site_to_clade'][site]
            identity_index = clade_LTEE_to_identity_index[clade_LTEE]
            identity_counts[k, identity_index] += 1

    if pop in EXCHANGE_FIRST_TWO_CLADES_LTEE and EXCHANGE_FIRST_TWO_CLADES_LTEE[pop]:
        temp = np.copy(identity_counts[0])
        identity_counts[0, :] = identity_counts[1]
        identity_counts[1, :] = temp

    sns.heatmap(identity_counts, center=0, vmin=0, vmax=pop_to_vmax[pop] if reduce_vmax else None, cmap=cmap, square=square, cbar=False, annot=True, fmt='.0f', annot_kws={"size": SIZEANNOTATION_HEATMAP})
    if plot_xticks:
        plt.xticks(ticks=xticks, labels=xticklabels, fontsize=SIZELABEL, rotation=xticklabels_rotation)
    else:
        plt.xticks(ticks=[], labels=[], fontsize=SIZELABEL)
    if plot_yticks:
        plt.yticks(ticks=yticks, labels=yticklabels_complete, fontsize=SIZELABEL, rotation=yticklabels_rotation)
    else:
        plt.yticks(ticks=[], labels=[], fontsize=SIZELABEL)

    ax.tick_params(**DEF_TICKPROPS_HEATMAP)
    plt.setp(ax.spines.values(), **DEF_AXPROPS)


def plot_figure_identity_example_LTEE(pop, reconstruction, data, save_file=None):
    w = SINGLE_COLUMN * 0.5
    fig = plt.figure(figsize=(w * 1.5, w))
    plot_identity_helper(pop, reconstruction, data)
    plt.title(f"Population {pop}", fontsize=SIZELABEL)
    plt.show()
    if save_file:
        fig.savefig(save_file, facecolor=fig.get_facecolor(), **DEF_FIGPROPS)


def plot_figure_identities_LTEE(populations, reconstructions, data, custom_arrangement=True, mark_nonclonal_with_rectangle=False, plot_ylabel_for_all_second_row=True, mark_nonclonal_with_horizontal_text=True, mark_nonclonal_with_vertical_text=False, nRow=2, nCol=4, wspace=0.3, hspace=0.3, square=False, reduce_vmax=False, ticklabels_rotation=0, plot_single_column=False, plot_sublabel=True, save_file=None):
    w = SINGLE_COLUMN if plot_single_column else DOUBLE_COLUMN

    if custom_arrangement:
        h = 0.8 * w if mark_nonclonal_with_horizontal_text else 0.33 * w
        fig = plt.figure(figsize=(w, 0.33 * w))

        populations_2_clades = ['m1', 'm2', 'm4', 'm5', 'm6', 'p5', 'p6']
        populations_3_clades = ['p1', 'p3']
        special_pop = 'p2'
        populations_4_clades = ['m3', 'p4']
        populations_nonclonal = ['p2', 'm3', 'p4']

        heatmap_height = 0.28 if mark_nonclonal_with_horizontal_text else 0.32
        heatmap_width = 0.115
        heatmap_distance = 0.02
        heatmap_distance_with_label = 0.074
        wspace = heatmap_distance / heatmap_width
        wspace_with_label = heatmap_distance_with_label / heatmap_width

        global_top = 0.92
        global_bottom = 0.18 if mark_nonclonal_with_horizontal_text else 0.08
        top_bottom = global_top - heatmap_height
        bottom_top = global_bottom + heatmap_height

        # first row
        fr_num_heatmaps = len(populations_2_clades)
        fr_left = 0.06
        fr_right = fr_left + heatmap_width * fr_num_heatmaps + heatmap_distance * (fr_num_heatmaps - 1)
        fr_box = dict(left=fr_left, right=fr_right, bottom=top_bottom, top=global_top)
        fr_gs = gridspec.GridSpec(1, fr_num_heatmaps, wspace=wspace, **fr_box)

        # second row left
        srl_num_heatmaps = len(populations_3_clades)
        if plot_ylabel_for_all_second_row:
            heatmap_distance_, wspace_ = heatmap_distance_with_label, wspace_with_label
        else:
            heatmap_distance_, wspace_ = heatmap_distance, wspace
        srl_right = fr_left + heatmap_width * srl_num_heatmaps + heatmap_distance_ * (srl_num_heatmaps - 1)
        srl_box = dict(left=fr_left, right=srl_right, bottom=global_bottom, top=bottom_top)
        srl_gs = gridspec.GridSpec(1, srl_num_heatmaps, wspace=wspace_, **srl_box)

        # second row right
        srr_num_heatmaps = len(populations_4_clades)
        if plot_ylabel_for_all_second_row:
            srr_right = fr_right
            srr_left = srr_right - (srr_num_heatmaps - 1) * heatmap_distance_ - srr_num_heatmaps * heatmap_width
        else:
            srr_left = fr_left + (fr_num_heatmaps - srr_num_heatmaps) * (heatmap_width + heatmap_distance_)
            srr_right = srr_left + heatmap_width * srr_num_heatmaps + heatmap_distance_ * (srr_num_heatmaps - 1)
        srr_box = dict(left=srr_left, right=srr_right, bottom=global_bottom, top=bottom_top)
        srr_gs = gridspec.GridSpec(1, srr_num_heatmaps, wspace=wspace_, **srr_box)

        # second row middle
        srm_left = srr_left - heatmap_width - heatmap_distance_ if plot_ylabel_for_all_second_row else 0.55
        srm_right = srm_left + heatmap_width
        srm_box = dict(left=srm_left, right=srm_right, bottom=global_bottom, top=bottom_top)
        srm_gs = gridspec.GridSpec(1, 1, wspace=wspace, **srm_box)

        # sublabels
        sublabel_x = -0.14
        sublabel_y = 1.15

        for i, pop in enumerate(populations_2_clades):
            ax = plt.subplot(fr_gs[0, i])
            # plot_xticks = True if i in [3, 4] else False
            plot_xticks = True
            plot_identity_helper(pop, reconstructions[pop], data, square=square, reduce_vmax=reduce_vmax, plot_xticks=plot_xticks, plot_yticks=(i == 0), xticklabels_rotation=ticklabels_rotation, yticklabels_rotation=ticklabels_rotation)
            plt.title(f"Population {pop}", fontsize=SIZELABEL)
            if i == 0:
                if plot_sublabel:
                    plt.text(x=sublabel_x, y=sublabel_y, s=SUBLABELS[0], transform=ax.transAxes, **DEF_SUBLABELPROPS)

        for i, pop in enumerate(populations_3_clades):
            ax = plt.subplot(srl_gs[0, i])
            plot_identity_helper(pop, reconstructions[pop], data, square=square, reduce_vmax=reduce_vmax, plot_xticks=True, plot_yticks=True if plot_ylabel_for_all_second_row else (i == 0), xticklabels_rotation=ticklabels_rotation, yticklabels_rotation=ticklabels_rotation)
            plt.title(f"Population {pop}", fontsize=SIZELABEL)
            if i == 0:
                if plot_sublabel:
                    plt.text(x=sublabel_x, y=sublabel_y, s=SUBLABELS[1], transform=ax.transAxes, **DEF_SUBLABELPROPS)

        pop = special_pop
        ax = plt.subplot(srm_gs[0, 0])
        plot_identity_helper(pop, reconstructions[pop], data, square=square, cmap=CMAP_NONCLONAL, reduce_vmax=reduce_vmax, plot_xticks=True, plot_yticks=True, xticklabels_rotation=ticklabels_rotation, yticklabels_rotation=ticklabels_rotation)
        plt.title(f"Population {pop}", fontsize=SIZELABEL)
        if plot_sublabel:
            plt.text(x=sublabel_x, y=sublabel_y, s=SUBLABELS[2], transform=ax.transAxes, **DEF_SUBLABELPROPS)

        for i, pop in enumerate(populations_4_clades):
            ax = plt.subplot(srr_gs[0, i])
            plot_identity_helper(pop, reconstructions[pop], data, square=square, cmap=CMAP_NONCLONAL, reduce_vmax=reduce_vmax, plot_xticks=True, plot_yticks=True if plot_ylabel_for_all_second_row else (i == 0), xticklabels_rotation=ticklabels_rotation, yticklabels_rotation=ticklabels_rotation)
            plt.title(f"Population {pop}", fontsize=SIZELABEL)

        if mark_nonclonal_with_rectangle:
            rect = matplotlib.patches.Rectangle((-9.8, -1.3), 13, 7.5, clip_on=False, linewidth=0.6, edgecolor=GREY_COLOR_HEX, facecolor='none')
            ax.add_patch(rect)

        # bbox_color = '#E5CDE9'
        bbox_color = '#C7DBD0'
        bbox_alpha = 0.8

        if mark_nonclonal_with_horizontal_text:
            t = plt.text(-2.25, -0.46, 'No clonal structure inferred previously', transform=ax.transAxes, fontsize=SIZELABEL, color='black')
            bbox = matplotlib.patches.FancyBboxPatch((-9.55, 6.7), 12.25, 0.8, clip_on=False, linewidth=0.6, edgecolor=bbox_color, facecolor=bbox_color, alpha=bbox_alpha)
            ax.add_patch(bbox)

        if mark_nonclonal_with_vertical_text:
            # t = plt.text(-3.65, -0.12, 'No clonal structure\ninferred previously', transform=ax.transAxes, fontsize=(SIZELABEL + SIZESUBLABEL) / 2, color='black', rotation='vertical')
            text_x = -11.2
            bbox_x = text_x + 0.14
            t = plt.text(text_x, 5.5, 'No clonal structure\ninferred previously', fontsize=(SIZELABEL + SIZESUBLABEL) / 2, color='black', rotation='vertical')
            bbox = matplotlib.patches.FancyBboxPatch((bbox_x, -0.75), 0.43, 6.45, clip_on=False, linewidth=0.6, edgecolor=bbox_color, facecolor=bbox_color, alpha=bbox_alpha)
            ax.add_patch(bbox)

    else:
        fig, axes = plt.subplots(nRow, nCol, figsize=(w, nRow / nCol * w))
        for i, pop in enumerate(populations):
            plt.sca(axes[i//nCol, i%nCol])
            at_bottom = (i//nCol == nRow - 1)
            at_left = (i%nCol == 0)
            plot_identity_helper(pop, reconstructions[pop], data, square=square, reduce_vmax=reduce_vmax, plot_xticks=at_bottom, plot_yticks=at_left, xticklabels_rotation=ticklabels_rotation, yticklabels_rotation=ticklabels_rotation)
            plt.title(f"Population {pop}", fontsize=SIZELABEL)
        if nCol == 6 and nRow == 2:
            plt.subplots_adjust(0.06, 0.1, 0.98, 0.9, wspace=wspace, hspace=hspace)
        else:
            plt.subplots_adjust(wspace=wspace, hspace=hspace)
    plt.show()
    if save_file:
        fig.savefig(save_file, facecolor=fig.get_facecolor(), **DEF_FIGPROPS)


def get_axes_for_performance_on_real_data(plot_single_column=True):
    if plot_single_column:
        ratio   = 1.8
        w       = SINGLE_COLUMN
        h       = ratio * w
        fig     = plt.figure(figsize=(w, h))

        box_top   = 0.96
        box_left  = 0.18
        box_right = 0.96
        ddy       = 0.03  # Adjust hspace
        dy        = 0.1  # Adjusts height of each subplot, & height of white space below the subplots.
        # Width of heatmap = box_right - box_left = 0.72 * w
        cov_top = 0.68
        cov_bottom = 0.2

        perf_left = box_left + 0.07
        perf_right = box_right
        perf_top = 0.19
        perf_bottom = 0.07
        wspace   = 0.5

        traj_boxes = [dict(left=box_left,
                           right=box_right,
                           bottom=box_top-((i+1)*dy)-(i*ddy),
                           top=box_top-(i*dy)-(i*ddy)) for i in range(2)]
        cov_box = dict(left=box_left,
                       right=box_right,
                       bottom=cov_bottom,
                       top=cov_top)
        perf_box = dict(left=perf_left,
                        right=perf_right,
                        bottom=perf_bottom,
                        top=perf_top)
        boxes = traj_boxes + [cov_box, perf_box]
        gridspecs = [gridspec.GridSpec(1, 1, wspace=wspace, **boxes[i]) for i in range(len(boxes))]
        axes = [plt.subplot(gridspecs[i][0, 0]) for i in range(len(gridspecs))]

    else:
        ratio   = 0.33
        w       = DOUBLE_COLUMN
        h       = ratio * w
        fig     = plt.figure(figsize=(w, h))

        divider_width = 0.1
        box_left = 0.08
        box_bottom = 0.14
        box_top = 0.95
        ddy = 0.06  # Adjust hspace
        dy = 0.23  # Adjusts height of each subplot. Also adjusts height of white space below the subplots.
        # Height of heatmap = cov_top - cov_bottom
        # Width of heatmap = (cov_top - cov_bottom) * (h / w)
        cov_right = 0.99
        cov_bottom = box_bottom
        cov_top = box_top

        cbar_width = 0.1
        cov_width = ratio * (cov_top - cov_bottom) + cbar_width
        box_middle = cov_right - divider_width - cov_width
        cov_left = box_middle + divider_width

        perf_left = box_left + 0.1
        perf_right = box_middle - 0.1
        perf_bottom = box_bottom - 0.04
        perf_top = 0.28

        traj_boxes = [dict(left=box_left,
                      right=box_middle,
                      bottom=box_top-((i+1)*dy)-(i*ddy),
                      top=box_top-(i*dy)-(i*ddy)) for i in range(2)]
        cov_box = dict(left=cov_left,
                       right=cov_right,
                       bottom=cov_bottom,
                       top=cov_top)
        perf_box = dict(left=perf_left,
                        right=perf_right,
                        bottom=perf_bottom,
                        top=perf_top)

        boxes = traj_boxes + [cov_box] + [perf_box]
        gridspecs = [gridspec.GridSpec(1, 1, **boxes[i]) for i in range(len(boxes))]
        axes = [plt.subplot(gridspecs[i][0, 0]) for i in range(len(gridspecs))]
    return fig, axes


def get_clade_colors_and_labels(num_clades):
    clade_colors = [GREY_COLOR_RGB] + sns.husl_palette(num_clades + 1)[:num_clades]
    clade_labels = ['Ancestor'] + [f'Clade {i + 1}' for i in range(num_clades)]
    return clade_colors, clade_labels


def get_allele_colors(num_alleles):
    allele_colors = sns.husl_palette(min(num_alleles, 50))
    return allele_colors


def plot_cov_heatmap(reconstruction, ylabel_x=-0.21, alpha=0.5, grid_line_offset=0, cbar_shrink=1, xtick_distance=40):
    ax = plt.gca()
    int_cov = reconstruction.recoveredIntCov
    clades = [reconstruction.otherMuts] + reconstruction.cladeMuts
    segmented_int_cov = RC.segmentMatrix(int_cov, clades)[0]

    L = len(segmented_int_cov)
    xticklabels = np.arange(0, L + xtick_distance // 2, xtick_distance)
    xticks = [l for l in xticklabels]

    heatmap = sns.heatmap(segmented_int_cov, center=0, cmap=CMAP, square=True, cbar=True, cbar_kws={"shrink": cbar_shrink})
    cbar = heatmap.collections[0].colorbar
    # cbar.ax.tick_params(labelsize=SIZELABEL, length=2, color=)
    cbar.ax.tick_params(**DEF_TICKPROPS_COLORBAR)
    ticks, ylabels, group_sizes = get_ticks_and_labels_for_clusterization(clades, name='Clade', note_size=True)
    plot_ticks_and_labels_for_clusterization(ticks, ylabels, group_sizes, ylabel_x=ylabel_x)
    ax.hlines([_ + grid_line_offset for _ in ticks], *ax.get_xlim(), color=GREY_COLOR_HEX, alpha=alpha, linewidth=SIZELINE * 1.2)
    ax.vlines([_ + grid_line_offset for _ in ticks], *ax.get_ylim(), color=GREY_COLOR_HEX, alpha=alpha, linewidth=SIZELINE * 1.2)
    set_ticks_labels_axes(yticks=[], yticklabels=[], xticks=xticks, xticklabels=xticklabels)
    plt.xlabel('Locus index', fontsize=SIZELABEL)
    # set_ticks_labels_axes(yticks=ticks[:-1], yticklabels=[], xticks=[], xticklabels=[])


def plot_perf_bars(measured_fitness, inferred_fitness_list, ylabel=LABEL_SPEARMANR_FITNESS_FOUR, methods=METHODS, use_pearsonr=False):
    ax = plt.gca()
    hist_props = dict(lw=SIZELINE/2, width=0.5, align='center', orientation='vertical',
                      edgecolor=[BKCOLOR for i in range(len(methods))])

    xlim, ylim = [-0.6, 4], [0, 1]
    xs, labels = np.arange(0, len(methods)), methods
    if use_pearsonr:
        ys = [stats.pearsonr(measured_fitness, inferred_fitness)[0] for inferred_fitness in inferred_fitness_list]
    else:
        ys = [stats.spearmanr(measured_fitness, inferred_fitness)[0] for inferred_fitness in inferred_fitness_list]
        print(measured_fitness)
        print(ys)

    colorlist = ['#FFB511', '#E8E8E8', '#E8E8E8', '#E8E8E8']
    yticks = [0, 0.5, 1]
    yticklabels = [str(_) for _ in yticks]
    pprops = { 'colors':      [colorlist],
               'xlim':        xlim,
               'ylim':        ylim,
               'xticks':      xs,
               'xticklabels': labels,
               'yticks':      [],
               'theme':       'open',
               'hide':        ['left','right'] }

    pprops['yticks'] = yticks
    pprops['yticklabels'] = yticklabels
    pprops['ylabel'] = ylabel
    pprops['hide']   = []

    mp.plot(type='bar', ax=ax, x=[xs], y=[ys], plotprops=hist_props, **pprops)


def plot_clade_annotation(annotXs, annotYs, clade_colors, clade_labels, background_width=10.8, background_width_for_ancestor=12.8, background_height=0.09, background_color='#f2f2f2'):
    ax = plt.gca()
    for x, y, color, label in zip(annotXs, annotYs, clade_colors, clade_labels):
        plt.text(x, y, label, fontsize=SIZEANNOTATION, color=color, zorder=10)
        patch = mpatches.FancyBboxPatch((x, y + 0.01), background_width_for_ancestor if label == 'Ancestor' else background_width, background_height, color=background_color, zorder=5, boxstyle=mpatches.BoxStyle("Round", pad=0.02, rounding_size=0), clip_on=False)
        ax.add_patch(patch)


def plot_figure_performance_on_data_PALTE(traj, reconstruction, measured_fitness_by_pop, inferred_fitness_by_pop_list, times=PALTEanalysis.TIMES, plot_single_column=False, use_pearsonr=False, plot_muller_plot=True, background_width=83, background_width_for_ancestor=93, background_height=0.09, background_color='#f2f2f2', methods=PALTEanalysis.METHODS, alpha=0.5, ylim=(-0.03, 1.03), grid_line_offset=-0.5, save_file=None):

    fig, axes = get_axes_for_performance_on_real_data(plot_single_column=plot_single_column)

    measured_fitness, inferred_fitness_list = PALTEanalysis.parse_fitness_by_pop_into_list(measured_fitness_by_pop, inferred_fitness_by_pop_list)

    xlim = (-5, 605)
    ylim = (-0.04, 1.04)
    ylabel = 'Frequency'
    xticks = np.arange(0, 700, 100)
    sublabel_x, sublabel_y = -0.15, 1.06

    plt.sca(axes[0])
    ax = plt.gca()
    allele_colors = get_allele_colors(len(traj[0]))
    AP.plotTraj(traj, times=times, linewidth=SIZELINE, colors=allele_colors, alpha=alpha, plotFigure=False, plotShow=False, title=None)
    set_ticks_labels_axes(ylim=ylim, xlim=xlim, xticklabels=[], ylabel=ylabel, xticks=xticks)
    plt.text(x=sublabel_x, y=sublabel_y, s=SUBLABELS[0], transform=ax.transAxes, **DEF_SUBLABELPROPS)

    plt.sca(axes[1])
    ax = plt.gca()
    clade_colors, clade_labels = get_clade_colors_and_labels(reconstruction.numClades)
    clade_colors[4] = 'orange'
    if plot_muller_plot:
        reconstruction.getMullerCladeFreq(mullerColors=clade_colors)
        plt.stackplot(reconstruction.times, reconstruction.mullerCladeFreq.T, colors=reconstruction.mullerColors)
        annotXs = [20, 80, 100, 300, 260, 350]
        annotYs = [0.8, 0.15, 0.45, 0.15, 0.5, 0.7]
        plot_clade_annotation(annotXs, annotYs, clade_colors, clade_labels, background_width=background_width, background_width_for_ancestor=background_width_for_ancestor, background_height=background_height, background_color=background_color)
    else:
        annotXs = [50, 100, 150, 350, 280, 500]
        annotYs = [0.8, 0.1, 0.5, 0.6, 0.15, 0.866]
        AP.plotTraj(reconstruction.cladeFreqWithAncestor, times=times, linewidth=SIZELINE * 2, alpha=alpha, colors=clade_colors, annot=True, annotTexts=clade_labels, annotXs=annotXs, annotYs=annotYs, fontsize=SIZELABEL, labels=None, plotFigure=False, plotShow=False, title=None)
    set_ticks_labels_axes(ylim=ylim, xlim=xlim, ylabel=ylabel, xlabel='Generation')
    plt.text(x=sublabel_x, y=sublabel_y, s=SUBLABELS[1], transform=ax.transAxes, **DEF_SUBLABELPROPS)

    plt.sca(axes[2])
    ax = plt.gca()
    plot_cov_heatmap(reconstruction, ylabel_x=-0.21 if plot_single_column else -0.32, alpha=alpha, grid_line_offset=grid_line_offset, xtick_distance=40)
    plt.text(x=1.08, y=sublabel_y, s=SUBLABELS[3], transform=axes[0].transAxes, **DEF_SUBLABELPROPS)

    plt.sca(axes[3])
    ax = plt.gca()
    ylabel = LABEL_PEARSONR_FITNESS_FOUR if use_pearsonr else LABEL_SPEARMANR_FITNESS_FOUR
    plot_perf_bars(measured_fitness, inferred_fitness_list, ylabel=ylabel, methods=methods, use_pearsonr=use_pearsonr)
    plt.text(x=sublabel_x, y=sublabel_y - 1.5, s=SUBLABELS[2], transform=axes[1].transAxes, **DEF_SUBLABELPROPS)

    plt.show()
    if save_file:
        fig.savefig(save_file, facecolor=fig.get_facecolor(), **DEF_FIGPROPS)


def plot_figure_fitness_biplot_PALTE(measured_fitness_by_pop, inferred_fitness_by_pop_list, methods=PALTEanalysis.METHODS, alpha=1, xlim=(-1, 10), ylim=None, yticks=None, plot_title=True, save_file=None, figsize=(8, 3)):

    measured_fitness, inferred_fitness_list = PALTEanalysis.parse_fitness_by_pop_into_list(measured_fitness_by_pop, inferred_fitness_by_pop_list)
    # print(inferred_fitness_list)

    fig, axes = plt.subplots(1, 4, figsize=figsize)
    xlabel = 'Measured fitness'
    if ylim is None:
        ylim = xlim
        xlim = None

    for i, inferred_fitness in enumerate(inferred_fitness_list):
        plt.sca(axes[i])
        ylabel = methods[i]
        plot_comparison(measured_fitness, inferred_fitness, xlabel, ylabel, alpha=alpha, ylim=ylim, xlim=xlim, yticks=yticks, xticks=yticks, plot_title=plot_title)

    plt.subplots_adjust(wspace=0.4)
    plt.show()
    if save_file:
        fig.savefig(save_file, facecolor=fig.get_facecolor(), **DEF_FIGPROPS)



def plot_figure_performance_on_data_tobramycin(traj, reconstruction, measured_MIC_list, median_inferred_fitness_lists, times=tobramycin_analysis.TIMES_PA, plot_single_column=False, use_pearsonr=False, plot_muller_plot=True, background_width=9, background_height=0.09, background_color='#f2f2f2', methods=tobramycin_analysis.METHODS, alpha=0.5, xticks=np.arange(0, 90, 10), ylim=(-0.03, 1.03), grid_line_offset=-0.02, save_file=None):

    fig, axes = get_axes_for_performance_on_real_data(plot_single_column=plot_single_column)

    xlim = (-2, 80)
    ylim = (-0.04, 1.04)
    ylabel = 'Frequency'
    sublabel_x, sublabel_y = -0.15, 1.06

    plt.sca(axes[0])
    ax = plt.gca()
    allele_colors = get_allele_colors(len(traj[0]))
    AP.plotTraj(traj, times=times, linewidth=SIZELINE, colors=allele_colors, alpha=alpha, plotFigure=False, plotShow=False, title=None)
    set_ticks_labels_axes(ylim=ylim, xlim=xlim, xticklabels=[], ylabel=ylabel, xticks=xticks)
    plt.text(x=sublabel_x, y=sublabel_y, s=SUBLABELS[0], transform=ax.transAxes, **DEF_SUBLABELPROPS)

    plt.sca(axes[1])
    ax = plt.gca()
    clade_colors, clade_labels = get_clade_colors_and_labels(reconstruction.numClades)
    if plot_muller_plot:
        reconstruction.getMullerCladeFreq(mullerColors=clade_colors)
        plt.stackplot(reconstruction.times, reconstruction.mullerCladeFreq.T, colors=reconstruction.mullerColors)
        annotXs = [2, 65, 15, 70, 30]
        annotYs = [0.8, 0.25, 0.45, 0.8, 0.7]
        plot_clade_annotation(annotXs, annotYs, clade_colors, clade_labels, background_width=background_width, background_width_for_ancestor=background_width + 1.5, background_height=background_height, background_color=background_color)
        # for x, y, color, label in zip(annotXs, annotYs, clade_colors, clade_labels):
        #     plt.text(x, y, label, fontsize=SIZEANNOTATION, color=color, zorder=10)
        #     patch = mpatches.FancyBboxPatch((x, y + 0.01), 12.8 if label == 'Ancestor' else background_width, background_height, color=background_color, zorder=5, boxstyle=mpatches.BoxStyle("Round", pad=0.02, rounding_size=0), clip_on=False)
        #     ax.add_patch(patch)
    else:
        annotXs = [7, 60, 20, 67, 35]
        annotYs = [0.8, 0.75, 0.65, 0.35, 0.75]
        AP.plotTraj(reconstruction.cladeFreqWithAncestor, times=times, linewidth=SIZELINE * 2, alpha=alpha, colors=clade_colors, annot=True, annotTexts=clade_labels, annotXs=annotXs, annotYs=annotYs, fontsize=SIZELABEL, labels=None, plotFigure=False, plotShow=False, title=None)
        # plt.legend(fontsize=SIZELEGEND)
    set_ticks_labels_axes(ylim=ylim, xlim=xlim, ylabel=ylabel, xlabel='Generation', xticks=xticks, xticklabels=xticks)
    plt.text(x=sublabel_x, y=sublabel_y, s=SUBLABELS[1], transform=ax.transAxes, **DEF_SUBLABELPROPS)

    plt.sca(axes[2])
    plot_cov_heatmap(reconstruction, ylabel_x=-0.21 if plot_single_column else -0.3, alpha=alpha, grid_line_offset=grid_line_offset, xtick_distance=2)
    plt.text(x=1.1, y=sublabel_y, s=SUBLABELS[3], transform=axes[0].transAxes, **DEF_SUBLABELPROPS)

    plt.sca(axes[3])
    ylabel = "Rank correlation\nbetween MIC and\ninferred fitness\n" + r"(Spearman’s $\rho$)"
    plot_perf_bars(measured_MIC_list, median_inferred_fitness_lists, ylabel=ylabel, methods=methods, use_pearsonr=use_pearsonr)
    plt.text(x=sublabel_x, y=sublabel_y - 1.5, s=SUBLABELS[2], transform=axes[1].transAxes, **DEF_SUBLABELPROPS)

    plt.show()
    if save_file:
        fig.savefig(save_file, facecolor=fig.get_facecolor(), **DEF_FIGPROPS)
