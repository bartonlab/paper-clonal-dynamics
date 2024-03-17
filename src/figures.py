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

import copy
from copy import deepcopy

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec

import mplot as mp

import seaborn as sns

from scipy import stats
from sklearn.linear_model import LinearRegression

import estimate_covariance as EC

# sys.path.append('../paper-clade-reconstruction/src')

############# PARAMETERS #############

# GLOBAL VARIABLES

# METHODS = ['SL', 'true_cov', 'recovered', 'est_cov', 'Lolipop']
# METHODS = ['true_cov', 'recovered', 'Lolipop', 'Evoracle', 'haploSep', 'est_cov', 'SL']
OUR_METHOD_NAME = 'dxdx'  # 'recovered'
TRUE_COV_NAME = 'True'  # 'true_cov'
EST_METHOD_NAME = 'LB'  # 'est_cov'
METHODS = [TRUE_COV_NAME, OUR_METHOD_NAME, 'Lolipop', 'Evoracle', EST_METHOD_NAME, 'SL']
METHODS_COMPARE_RUN_TIME = [OUR_METHOD_NAME, 'Lolipop', 'Evoracle']
MARKERS_METHODS_COMPARE_RUN_TIME = ['o', '^', 'v']

COLOR_PALETTES = {
    key: sns.color_palette(key) for key in ['deep', 'tab10', 'bright', 'pastel']
}
COLORS = COLOR_PALETTES['bright']
METHOD_COLORS = ['#ff6666', '#FFB511', COLORS[0], COLORS[2], 'grey', 'grey']
METHOD_MARKERS = ['o', 'd', 'v', '^', '<', '>']
DELTA_G = r"$\Delta g$"

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
    # METHODS_COLORS = sns.husl_palette(len(METHODS) + 1)
    METHODS_COMPARE_RUN_TIME_COLORS = sns.husl_palette(len(METHODS_COMPARE_RUN_TIME) + 1)
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

LTEE_INFERRED_SHARED_COLOR = '#fdd017'

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
SINGLE_COLUMN = cm2inch(8.5)
ONE_FIVE_COLUMN = cm2inch(11.4)
DOUBLE_COLUMN = cm2inch(17.4)

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
EDGECOLORS   = 'none'

# DEF_FIGPROPS = {
#     'transparent' : True,
#     'edgecolor'   : None,
#     'dpi'         : 1000,
#     # 'bbox_inches' : 'tight',
#     'pad_inches'  : 0.05,
# }
DEF_FIGPROPS = {
    'transparent' : True,
    'edgecolor'   : None,
    'dpi'         : 300, # 300,  # 1000  # Comment this out for generating vector figures
    # 'bbox_inches' : 'tight',
    'pad_inches'  : 0.05,
    # 'backend'     : 'PDF',
    # 'backend'     : 'PGF',
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

DEF_TICKPROPS_NO_AXES = {
    'length'    : 0,
    'width'     : 0,
    'pad'       : 0,
    'axis'      : 'both',
    'which'     : 'both',
    'direction' : 'out',
    'colors'    : BKCOLOR,
    'labelsize' : SIZETICK,
    'bottom'    : False,
    'left'      : False,
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

DEF_TICKPROPS_COV_TOP = {
    'length'    : 0,
    'width'     : AXWIDTH/2,
    'pad'       : TICKPAD,
    'axis'      : 'both',
    'which'     : 'both',
    'direction' : 'out',
    'colors'    : BKCOLOR,
    'labelsize' : SIZETICK,
    'bottom'    : False,
    'left'      : False,
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
LABEL_SPEARMANR_FITNESS_FOUR = "Rank correlation\nbetween true\nand inferred fitness\n" + r"(Spearman’s $\rho$)"

LABEL_SPEARMANR_FOUR_2 = "Rank correlation\nbetween true and inferred\nselection coefficients\n" + r"(Spearman’s $\rho$)"
LABEL_SPEARMANR_COVARIANCE_FOUR_2 = "Rank correlation\nbetween true and\ninferred covariances\n" + r"(Spearman’s $\rho$)"
LABEL_SPEARMANR_FITNESS_FOUR_2 = "Rank correlation\nbetween true\nand inferred fitness\n" + r"(Spearman’s $\rho$)"

LABEL_PEARSONR_FOUR = "Linear correlation\nbetween true and inferred\nselection coefficients\n" + r"(Pearson’s $r$)"
LABEL_PEARSONR_COVARIANCE_FOUR = "Linear correlation\nbetween true and\ninferred covariances\n" + r"(Pearson’s $r$)"
LABEL_PEARSONR_FITNESS_FOUR = "Linear correlation\nbetween true and\ninferred fitness\n" + r"(Pearson’s $r$)"

LABEL_MAE_FITNESS = ' MAE of inferred\nhaplotype fitness'

PARAMS = {'text.usetex': False, 'mathtext.fontset': 'stixsans', 'mathtext.default': 'regular', 'pdf.fonttype': 42, 'ps.fonttype': 42}
plt.rcParams.update(matplotlib.rcParamsDefault)
plt.rcParams.update(PARAMS)

EXCHANGE_FIRST_TWO_CLADES_LTEE = {
    'm1': True, 'm2': True, 'm4': True, 'm5': True, 'm6': False, 'p1': False, 'p3': False, 'p5': True, 'p6': True,
}

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
        locus += 1  # Convert to 1-based index for plotting
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


def get_axes_for_method_overview(plot_single_column=False, plot_arrow=True, plot_arrow_from_traj_center=False):

    if not plot_single_column:

        ratio   = 0.5
        w       = DOUBLE_COLUMN
        h       = ratio * w
        fig     = plt.figure(figsize=(w, h))

        global_left = 0.08
        global_right = 0.98
        global_top = 0.95
        global_bottom = 0.05

        arrow_width = 0.07
        full_period_ratio = 0.7
        full_width = (global_right - global_left - 3 * arrow_width) / (2 + 2 * full_period_ratio)
        period_width = full_period_ratio * full_width

        step_1_width = full_width
        arrow_12_width = arrow_width
        step_2_width = period_width
        arrow_23_width = arrow_width
        step_3_width = period_width
        arrow_34_width = arrow_width
        step_4_width = full_width

        traj_height = 0.2
        divider_height = 0.07
        heatmap_height = 0.12
        heatmap_width = heatmap_height * ratio
        step_1_divider_width = step_1_width - 2 * heatmap_width
        step_2_divider_width = step_2_width - 2 * heatmap_width

        step_1_left = global_left
        step_1_right = global_left + step_1_width
        step_1_height = traj_height + divider_height + heatmap_height
        step_1_top = 0.5 + step_1_height / 2
        step_1_bottom = 0.5 - step_1_height / 2
        step_1_center = 0.5

        step_2_left = step_1_right + arrow_12_width
        step_2_right = step_2_left + step_2_width
        step_2_upper_top = global_top
        step_2_upper_bottom = global_top - traj_height - divider_height - heatmap_height
        step_2_lower_top = global_bottom + traj_height + divider_height + heatmap_height
        step_2_lower_bottom = global_bottom
        step_2_upper_center = (step_2_upper_top + step_2_upper_bottom) / 2
        step_2_lower_center = (step_2_lower_top + step_2_lower_bottom) / 2

        step_3_left = step_2_right + arrow_23_width
        step_3_right = step_3_left + step_3_width
        step_3_upper_top = step_2_upper_center + traj_height / 2
        step_3_lower_top = step_2_lower_center + traj_height / 2
        step_3_upper_bottom = step_2_upper_center - traj_height / 2
        step_3_lower_bottom = step_2_lower_center - traj_height / 2
        step_3_upper_center = (step_3_upper_top + step_3_upper_bottom) / 2
        step_3_lower_center = (step_3_lower_top + step_3_lower_bottom) / 2

        step_4_left = step_3_right + arrow_34_width
        step_4_right = step_4_left + step_4_width
        step_4_top = step_1_center + traj_height / 2
        step_4_bottom = step_1_center - traj_height / 2
        step_4_center = (step_4_top + step_4_bottom) / 2

        step_1_traj_box = dict(left=step_1_left,
                               right=step_1_right,
                               bottom=step_1_top - traj_height,
                               top=step_1_top)

        step_1_D_box = dict(left=step_1_left,
                            right=step_1_left + heatmap_width,
                            bottom=step_1_bottom,
                            top=step_1_bottom + heatmap_height)

        step_1_segmentedD_box = dict(left=step_1_right - heatmap_width,
                                     right=step_1_right,
                                     bottom=step_1_bottom,
                                     top=step_1_bottom + heatmap_height)

        step_2_traj_upper_box = dict(left=step_2_left,
                                     right=step_2_right,
                                     bottom=step_2_upper_top - traj_height,
                                     top=step_2_upper_top)

        step_2_D_upper_box = dict(left=step_2_left,
                                  right=step_2_left + heatmap_width,
                                  bottom=step_2_upper_bottom,
                                  top=step_2_upper_top - traj_height - divider_height)

        step_2_segmentedD_upper_box = dict(left=step_2_right - heatmap_width,
                                           right=step_2_right,
                                           bottom=step_2_upper_bottom,
                                           top=step_2_upper_top - traj_height - divider_height)

        step_2_traj_lower_box = dict(left=step_2_left,
                                     right=step_2_right,
                                     bottom=step_2_lower_top - traj_height,
                                     top=step_2_lower_top,)

        step_2_D_lower_box = dict(left=step_2_left,
                                  right=step_2_left + heatmap_width,
                                  bottom=step_2_lower_bottom,
                                  top=step_2_lower_top - traj_height - divider_height)

        step_2_segmentedD_lower_box = dict(left=step_2_right - heatmap_width,
                                           right=step_2_right,
                                           bottom=step_2_lower_bottom,
                                           top=step_2_lower_top - traj_height - divider_height)

        step_3_traj_upper_box = dict(left=step_3_left,
                                     right=step_3_right,
                                     bottom=step_3_upper_bottom,
                                     top=step_3_upper_top,)

        step_3_traj_lower_box = dict(left=step_3_left,
                                     right=step_3_right,
                                     bottom=step_3_lower_bottom,
                                     top=step_3_lower_top,)

        step_4_traj_box = dict(left=step_4_left,
                               right=step_4_right,
                               bottom=step_4_bottom,
                               top=step_4_top,)

        traj_boxes = [step_1_traj_box, step_2_traj_upper_box, step_2_traj_lower_box, step_3_traj_upper_box, step_3_traj_lower_box, step_4_traj_box]
        D_boxes = [step_1_D_box, step_2_D_upper_box, step_2_D_lower_box]
        segmentedD_boxes = [step_1_segmentedD_box, step_2_segmentedD_upper_box, step_2_segmentedD_lower_box]

        # boxes = traj_boxes + D_boxes + segmentedD_boxes
        traj_gridspecs = [gridspec.GridSpec(1, 1, **box) for box in traj_boxes]
        D_gridspecs = [gridspec.GridSpec(1, 1, **box) for box in D_boxes]
        segmentedD_gridspecs = [gridspec.GridSpec(1, 1, **box) for box in segmentedD_boxes]

        traj_axes = [plt.subplot(gs[0, 0]) for gs in traj_gridspecs]
        D_axes = [plt.subplot(gs[0, 0]) for gs in D_gridspecs]
        segmentedD_axes = [plt.subplot(gs[0, 0]) for gs in segmentedD_gridspecs]

    if plot_arrow:
        def_arrowprops = {
            'arrowstyle': '<-',
            'color': '0.5',
            'shrinkA': 2, 
            'shrinkB': 2,
            'patchA': None, 
            'patchB': None,
        }
        annotprops = {
            'xy': None,
            'xytext': None,
            'xycoords': 'axes fraction',
            'textcoords': 'axes fraction',
        }

        ax = traj_axes[0]
        if plot_arrow_from_traj_center:
            connectionstyle = 'bar,angle=90,fraction=0.29'
            annotprops['xy'] = 1, 0.5
            annotprops['xytext'] = (arrow_12_width + step_1_width) / step_1_width, 0.5 + (step_2_upper_top - step_1_top) / traj_height
            arrowprops = {**def_arrowprops, 'connectionstyle': connectionstyle}
            ax.annotate('', **annotprops, arrowprops=arrowprops,)

            annotprops['xytext'] = (arrow_12_width + step_1_width) / step_1_width, (step_2_lower_top - traj_height / 2 - (step_1_top - traj_height)) / traj_height
            connectionstyle = 'bar,angle=90,fraction=-0.29'
            arrowprops = {**def_arrowprops, 'connectionstyle': connectionstyle}
            ax.annotate('', **annotprops, arrowprops=arrowprops,)
        else:
            connectionstyle = 'bar,angle=90,fraction=0.29'
            annotprops['xy'] = 1, (step_1_center - (step_1_top - traj_height)) / traj_height
            annotprops['xytext'] = (arrow_12_width + step_1_width) / step_1_width, (step_2_upper_center - (step_1_top - traj_height)) / traj_height
            arrowprops = {**def_arrowprops, 'connectionstyle': connectionstyle}
            ax.annotate('', **annotprops, arrowprops=arrowprops,)

            annotprops['xytext'] = (arrow_12_width + step_1_width) / step_1_width, (step_2_lower_center - (step_1_top - traj_height)) / traj_height
            connectionstyle = 'bar,angle=90,fraction=-0.29'
            arrowprops = {**def_arrowprops, 'connectionstyle': connectionstyle}
            ax.annotate('', **annotprops, arrowprops=arrowprops,)

        ax = traj_axes[1]
        annotprops['xy'] = 1, (step_2_upper_center - (step_2_upper_top - traj_height)) / traj_height
        annotprops['xytext'] = (arrow_23_width + step_2_width) / step_2_width, (step_2_upper_center - (step_2_upper_top - traj_height)) / traj_height
        connectionstyle = 'bar,angle=90,fraction=-0.29'
        arrowprops = {**def_arrowprops, 'connectionstyle': connectionstyle}
        ax.annotate('', **annotprops, arrowprops=arrowprops,)


        ax = traj_axes[2]
        # annotprops['xy'] = 1, (step_2_upper_center - (step_2_upper_top - traj_height)) / traj_height
        # annotprops['xytext'] = (arrow_23_width + step_2_width) / step_2_width, (step_2_upper_center - (step_2_upper_top - traj_height)) / traj_height
        connectionstyle = 'bar,angle=90,fraction=-0.29'
        arrowprops = {**def_arrowprops, 'connectionstyle': connectionstyle}
        ax.annotate('', **annotprops, arrowprops=arrowprops,)

        ax = traj_axes[3]
        annotprops['xy'] = 1, 0.5
        annotprops['xytext'] = (arrow_34_width + step_3_width) / step_3_width, (step_4_center - step_3_upper_bottom) / traj_height
        connectionstyle = 'bar,angle=90,fraction=-0.29'
        arrowprops = {**def_arrowprops, 'connectionstyle': connectionstyle}
        ax.annotate('', **annotprops, arrowprops=arrowprops,)

        ax = traj_axes[4]
        annotprops['xy'] = 1, 0.5
        annotprops['xytext'] = (arrow_34_width + step_3_width) / step_3_width, (step_4_center - step_3_lower_bottom) / traj_height
        connectionstyle = 'bar,angle=90,fraction=0.29'
        arrowprops = {**def_arrowprops, 'connectionstyle': connectionstyle}
        ax.annotate('', **annotprops, arrowprops=arrowprops,)



    return fig, traj_axes, D_axes, segmentedD_axes


def get_axes_for_method_overview_2(plot_single_column=False, plot_arrow=True, reconstruction=None):

    if plot_single_column and reconstruction is not None:
        ratio   = 1.6
        w       = SINGLE_COLUMN
        h       = ratio * w
        fig     = plt.figure(figsize=(w, h))

        global_left = 0.04
        global_right = 0.97
        global_top = 0.98
        global_bottom = 0.04

        arrow_height = 0.06
        short_arrow_height = arrow_height * 0.75
        traj_height = 0.12
        traj_width = 0.7
        heatmap_height = 0.1

        # Normalize
        total = 2 * arrow_height + 3 * short_arrow_height + 4 * traj_height + 2 * heatmap_height
        arrow_height /= total / (global_top - global_bottom)
        short_arrow_height /= total / (global_top - global_bottom)
        traj_height /= total / (global_top - global_bottom)
        heatmap_height /= total / (global_top - global_bottom)

        heatmap_width = heatmap_height * ratio
        pb_1 = reconstruction.periodBoundaries[0]
        pb_2 = reconstruction.periodBoundaries[1]
        pl_1 = pb_1[1] - pb_1[0]
        pl_2 = pb_2[1] - pb_2[0]
        period_1_width = pl_1 / (pl_1 + pl_2) * traj_width
        period_2_width = pl_2 / (pl_1 + pl_2) * traj_width

        step_1_height = traj_height + short_arrow_height + heatmap_height
        arrow_12_height = arrow_height
        step_2_height = traj_height + short_arrow_height + heatmap_height
        arrow_23_height = short_arrow_height
        step_3_height = traj_height
        arrow_34_height = arrow_height
        step_4_height = traj_height

        step_1_center = (global_left + global_right) / 2
        step_1_left = step_1_center - traj_width / 2
        step_1_right = step_1_center + traj_width / 2
        step_1_width = step_1_right - step_1_left
        step_1_traj_top = global_top
        step_1_traj_bottom = step_1_traj_top - traj_height
        step_1_heatmap_top = step_1_traj_bottom - short_arrow_height
        step_1_heatmap_bottom = step_1_heatmap_top - heatmap_height
        step_1_heatmap_left = step_1_center - heatmap_width / 2
        step_1_heatmap_right = step_1_center + heatmap_width / 2
        step_1_top = step_1_traj_top
        step_1_bottom = step_1_heatmap_bottom

        step_2_traj_top = step_1_bottom - arrow_12_height
        step_2_traj_bottom = step_2_traj_top - traj_height
        step_2_heatmap_top = step_2_traj_bottom - short_arrow_height
        step_2_heatmap_bottom = step_2_heatmap_top - heatmap_height
        step_2_period_1_left = global_left
        step_2_period_1_right = step_2_period_1_left + period_1_width
        step_2_period_2_right = global_right
        step_2_period_2_left = step_2_period_2_right - period_2_width
        step_2_period_1_center = (step_2_period_1_left + step_2_period_1_right) / 2
        step_2_period_2_center = (step_2_period_2_left + step_2_period_2_right) / 2
        step_2_period_1_heatmap_left = step_2_period_1_center - heatmap_width / 2
        step_2_period_1_heatmap_right = step_2_period_1_center + heatmap_width / 2
        step_2_period_2_heatmap_right = step_2_period_2_center + heatmap_width / 2
        step_2_period_2_heatmap_left = step_2_period_2_center - heatmap_width / 2
        step_2_top = step_2_traj_top
        step_2_bottom = step_2_heatmap_bottom
        step_2_left = step_2_period_1_left
        step_2_right = step_2_period_2_right

        step_3_top = step_2_bottom - arrow_23_height
        step_3_bottom = step_3_top - step_3_height
        step_3_period_1_left = step_2_period_1_left
        step_3_period_1_right = step_2_period_1_right
        step_3_period_2_right = step_2_period_2_right
        step_3_period_2_left = step_2_period_2_left
        step_3_period_1_center = (step_3_period_1_left + step_3_period_1_right) / 2
        step_3_period_2_center = (step_3_period_2_left + step_3_period_2_right) / 2

        step_4_top = step_3_bottom - arrow_34_height
        step_4_bottom = step_4_top - step_4_height
        step_4_center = step_1_center
        step_4_left = step_4_center - traj_width / 2
        step_4_right = step_4_center + traj_width / 2

        step_1_traj_box = dict(left=step_1_left,
                               right=step_1_right,
                               bottom=step_1_traj_bottom,
                               top=step_1_traj_top)

        step_1_segmentedD_box = dict(left=step_1_heatmap_left,
                                     right=step_1_heatmap_right,
                                     bottom=step_1_heatmap_bottom,
                                     top=step_1_heatmap_top)

        step_2_period_1_traj_box = dict(left=step_2_period_1_left,
                                     right=step_2_period_1_right,
                                     bottom=step_2_traj_bottom,
                                     top=step_2_traj_top)

        step_2_period_1_segmentedD_box = dict(left=step_2_period_1_heatmap_left,
                                           right=step_2_period_1_heatmap_right,
                                           bottom=step_2_heatmap_bottom,
                                           top=step_2_heatmap_top)

        step_2_period_2_traj_box = dict(left=step_2_period_2_left,
                                     right=step_2_period_2_right,
                                     bottom=step_2_traj_bottom,
                                     top=step_2_traj_top)

        step_2_period_2_segmentedD_box = dict(left=step_2_period_2_heatmap_left,
                                           right=step_2_period_2_heatmap_right,
                                           bottom=step_2_heatmap_bottom,
                                           top=step_2_heatmap_top)

        step_3_period_1_traj_box = dict(left=step_3_period_1_left,
                                     right=step_3_period_1_right,
                                     bottom=step_3_bottom,
                                     top=step_3_top,)

        step_3_period_2_traj_box = dict(left=step_3_period_2_left,
                                     right=step_3_period_2_right,
                                     bottom=step_3_bottom,
                                     top=step_3_top,)

        step_4_traj_box = dict(left=step_4_left,
                               right=step_4_right,
                               bottom=step_4_bottom,
                               top=step_4_top,)

        traj_boxes = [step_1_traj_box, step_2_period_1_traj_box, step_2_period_2_traj_box, step_3_period_1_traj_box, step_3_period_2_traj_box, step_4_traj_box]
        segmentedD_boxes = [step_1_segmentedD_box, step_2_period_1_segmentedD_box, step_2_period_2_segmentedD_box]

        # boxes = traj_boxes + D_boxes + segmentedD_boxes
        traj_gridspecs = [gridspec.GridSpec(1, 1, **box) for box in traj_boxes]
        segmentedD_gridspecs = [gridspec.GridSpec(1, 1, **box) for box in segmentedD_boxes]

        traj_axes = [plt.subplot(gs[0, 0]) for gs in traj_gridspecs]
        D_axes = None
        segmentedD_axes = [plt.subplot(gs[0, 0]) for gs in segmentedD_gridspecs]

        if plot_arrow:
            y_offset = 0
            def_arrowprops = {
                'arrowstyle': '<-',
                'color': '0.5',
                'shrinkA': 2, 
                'shrinkB': 2,
                'patchA': None, 
                'patchB': None,
            }
            annotprops = {
                'xy': None,
                'xytext': None,
                'xycoords': 'axes fraction',
                'textcoords': 'axes fraction',
            }

            traj_ax_ratio = ratio * traj_height / traj_width
            # print('traj_ax_ratio = ', traj_ax_ratio)
            ax = traj_axes[0]
            connectionstyle = 'arc3,rad=0'
            annotprops['xy'] = 0.5, 0
            annotprops['xytext'] = 0.5, (-short_arrow_height) / traj_height + y_offset
            arrowprops = {**def_arrowprops, 'connectionstyle': connectionstyle}
            ax.annotate('', **annotprops, arrowprops=arrowprops,)

            annotprops['xy'] = 0.5, (-short_arrow_height - heatmap_height) / traj_height
            annotprops['xytext'] = (step_2_period_1_center - step_1_left) / step_1_width, (-short_arrow_height - heatmap_height - arrow_12_height) / traj_height
            # fraction = 0.5 * (annotprops['xytext'][1] - annotprops['xy'][1])
            # fraction = get_fraction_to_connect_at_mid(annotprops['xy'], annotprops['xytext'], 0, ax_ratio=traj_ax_ratio)
            fraction = -0.25
            connectionstyle = f"bar,angle=0,fraction={fraction}"
            arrowprops = {**def_arrowprops, 'connectionstyle': connectionstyle}
            ax.annotate('', **annotprops, arrowprops=arrowprops,)

            annotprops['xytext'] = (step_2_period_2_center - step_1_left) / step_1_width, (-short_arrow_height - heatmap_height - arrow_12_height) / traj_height
            fraction = -0.13
            connectionstyle = f'bar,angle=180,fraction={fraction}'
            arrowprops = {**def_arrowprops, 'connectionstyle': connectionstyle}
            ax.annotate('', **annotprops, arrowprops=arrowprops,)

            connectionstyle = 'arc3,rad=0'
            ax = traj_axes[1]
            annotprops['xy'] = 0.5, 0
            annotprops['xytext'] = 0.5, (-short_arrow_height) / traj_height + y_offset
            arrowprops = {**def_arrowprops, 'connectionstyle': connectionstyle}
            ax.annotate('', **annotprops, arrowprops=arrowprops,)

            annotprops['xy'] = 0.5, (-short_arrow_height - heatmap_height) / traj_height
            annotprops['xytext'] = 0.5, (-short_arrow_height - heatmap_height - arrow_23_height) / traj_height
            arrowprops = {**def_arrowprops, 'connectionstyle': connectionstyle}
            ax.annotate('', **annotprops, arrowprops=arrowprops,)

            ax = traj_axes[2]
            annotprops['xy'] = 0.5, 0
            annotprops['xytext'] = 0.5, (-short_arrow_height) / traj_height + y_offset
            arrowprops = {**def_arrowprops, 'connectionstyle': connectionstyle}
            ax.annotate('', **annotprops, arrowprops=arrowprops,)

            annotprops['xy'] = 0.5, (-short_arrow_height - heatmap_height) / traj_height
            annotprops['xytext'] = 0.5, (-short_arrow_height - heatmap_height - arrow_23_height) / traj_height
            arrowprops = {**def_arrowprops, 'connectionstyle': connectionstyle}
            ax.annotate('', **annotprops, arrowprops=arrowprops,)

            ax = traj_axes[3]
            annotprops['xy'] = 0.5, 0
            annotprops['xytext'] = (step_4_center - step_3_period_1_left) / period_1_width, (-arrow_34_height) / traj_height
            fraction = -0.25
            connectionstyle = f'bar,angle=180,fraction={fraction}'
            arrowprops = {**def_arrowprops, 'connectionstyle': connectionstyle}
            ax.annotate('', **annotprops, arrowprops=arrowprops,)

            ax = traj_axes[4]
            annotprops['xy'] = 0.5, 0
            annotprops['xytext'] = (step_4_center - step_3_period_2_left) / period_2_width, (-arrow_34_height) / traj_height
            fraction = -0.13
            connectionstyle = f'bar,angle=0,fraction={fraction}'
            arrowprops = {**def_arrowprops, 'connectionstyle': connectionstyle}
            ax.annotate('', **annotprops, arrowprops=arrowprops,)

    else:
        ratio   = 0.3
        w       = DOUBLE_COLUMN
        h       = ratio * w
        fig     = plt.figure(figsize=(w, h))

        global_left = 0.06
        global_right = 0.99
        global_top = 0.95
        global_bottom = 0.08

        arrow_width = 0.07
        short_arrow_width = arrow_width * 0.75
        full_period_ratio = 0.7
        traj_height = 0.25
        divider_height = 0.07
        heatmap_height = 0.2
        heatmap_width = heatmap_height * ratio
        full_width = (global_right - global_left - 2 * arrow_width - 2 * short_arrow_width - heatmap_width) / (2 + 2 * full_period_ratio)
        period_width = full_period_ratio * full_width

        step_1_width = full_width
        arrow_12_width = arrow_width
        step_2_width = period_width
        arrow_23_width = 2 * short_arrow_width + heatmap_width
        step_3_width = period_width
        arrow_34_width = arrow_width
        step_4_width = full_width

        step_1_left = global_left
        step_1_right = global_left + step_1_width
        step_1_top = global_top
        step_1_bottom = global_bottom
        step_1_height = step_1_top - step_1_bottom
        step_1_center = (step_1_top + step_1_bottom) / 2
        step_1_lower_center = (step_1_bottom + step_1_bottom + traj_height) / 2

        step_2_left = step_1_right + arrow_12_width
        step_2_right = step_2_left + step_2_width
        step_2_upper_top = global_top
        step_2_upper_bottom = global_top - traj_height
        step_2_lower_top = global_bottom + traj_height
        step_2_lower_bottom = global_bottom
        step_2_upper_center = (step_2_upper_top + step_2_upper_bottom) / 2
        step_2_lower_center = (step_2_lower_top + step_2_lower_bottom) / 2

        step_3_left = step_2_right + arrow_23_width
        step_3_right = step_3_left + step_3_width
        step_3_upper_top = step_2_upper_center + traj_height / 2
        step_3_lower_top = step_2_lower_center + traj_height / 2
        step_3_upper_bottom = step_2_upper_center - traj_height / 2
        step_3_lower_bottom = step_2_lower_center - traj_height / 2
        step_3_upper_center = (step_3_upper_top + step_3_upper_bottom) / 2
        step_3_lower_center = (step_3_lower_top + step_3_lower_bottom) / 2

        step_4_left = step_3_right + arrow_34_width
        step_4_right = step_4_left + step_4_width
        step_4_top = step_1_center + traj_height / 2
        step_4_bottom = step_1_center - traj_height / 2
        step_4_center = (step_4_top + step_4_bottom) / 2

        step_1_traj_box_1 = dict(left=step_1_left,
                                 right=step_1_right,
                                 bottom=step_1_top - traj_height,
                                 top=step_1_top)

        step_1_traj_box_2 = dict(left=step_1_left,
                                 right=step_1_right,
                                 bottom=step_1_bottom,
                                 top=step_1_bottom + traj_height)

        step_1_segmentedD_box = dict(left=(step_1_left + step_1_right - heatmap_width) / 2,
                                     right=(step_1_left + step_1_right + heatmap_width) / 2,
                                     bottom=step_1_center - heatmap_height / 2,
                                     top=step_1_center + heatmap_height / 2)

        step_2_traj_upper_box = dict(left=step_2_left,
                                     right=step_2_right,
                                     bottom=step_2_upper_top - traj_height,
                                     top=step_2_upper_top)

        step_2_segmentedD_upper_box = dict(left=step_2_right + short_arrow_width,
                                           right=step_2_right + short_arrow_width + heatmap_width,
                                           bottom=step_2_upper_center - heatmap_height / 2,
                                           top=step_2_upper_center + heatmap_height / 2)

        step_2_traj_lower_box = dict(left=step_2_left,
                                     right=step_2_right,
                                     bottom=step_2_lower_top - traj_height,
                                     top=step_2_lower_top,)

        step_2_segmentedD_lower_box = dict(left=step_2_right + short_arrow_width,
                                           right=step_2_right + short_arrow_width + heatmap_width,
                                           bottom=step_2_lower_center - heatmap_height / 2,
                                           top=step_2_lower_center + heatmap_height / 2)

        step_3_traj_upper_box = dict(left=step_3_left,
                                     right=step_3_right,
                                     bottom=step_3_upper_bottom,
                                     top=step_3_upper_top,)

        step_3_traj_lower_box = dict(left=step_3_left,
                                     right=step_3_right,
                                     bottom=step_3_lower_bottom,
                                     top=step_3_lower_top,)

        step_4_traj_box = dict(left=step_4_left,
                               right=step_4_right,
                               bottom=step_4_bottom,
                               top=step_4_top,)

        traj_boxes = [step_1_traj_box_1, step_2_traj_upper_box, step_2_traj_lower_box, step_3_traj_upper_box, step_3_traj_lower_box, step_4_traj_box, step_1_traj_box_2]
        # D_boxes = [step_1_D_box, step_2_D_upper_box, step_2_D_lower_box]
        segmentedD_boxes = [step_1_segmentedD_box, step_2_segmentedD_upper_box, step_2_segmentedD_lower_box]

        # boxes = traj_boxes + D_boxes + segmentedD_boxes
        traj_gridspecs = [gridspec.GridSpec(1, 1, **box) for box in traj_boxes]
        # D_gridspecs = [gridspec.GridSpec(1, 1, **box) for box in D_boxes]
        segmentedD_gridspecs = [gridspec.GridSpec(1, 1, **box) for box in segmentedD_boxes]

        traj_axes = [plt.subplot(gs[0, 0]) for gs in traj_gridspecs]
        # D_axes = [plt.subplot(gs[0, 0]) for gs in D_gridspecs]
        D_axes = None
        segmentedD_axes = [plt.subplot(gs[0, 0]) for gs in segmentedD_gridspecs]

        if plot_arrow:
            y_offset = 0
            def_arrowprops = {
                'arrowstyle': '<-',
                'color': '0.5',
                'shrinkA': 2, 
                'shrinkB': 2,
                'patchA': None, 
                'patchB': None,
            }
            annotprops = {
                'xy': None,
                'xytext': None,
                'xycoords': 'axes fraction',
                'textcoords': 'axes fraction',
            }

            ax = traj_axes[0]
            connectionstyle = 'bar,angle=0'
            annotprops['xy'] = 0.5, 0
            annotprops['xytext'] = 0.5, (step_1_center + heatmap_height / 2 - step_1_top + traj_height) / traj_height + y_offset
            arrowprops = {**def_arrowprops, 'connectionstyle': connectionstyle}
            ax.annotate('', **annotprops, arrowprops=arrowprops,)

            annotprops['xy'] = 0.5, (step_1_center - heatmap_height / 2 - step_1_top + traj_height) / traj_height + y_offset
            annotprops['xytext'] = 0.5, (step_1_bottom + traj_height - step_1_top + traj_height) / traj_height + y_offset
            arrowprops = {**def_arrowprops, 'connectionstyle': connectionstyle}
            ax.annotate('', **annotprops, arrowprops=arrowprops,)

            connectionstyle = 'bar,angle=90,fraction=0.18'
            annotprops['xy'] = 1, (step_1_lower_center - step_1_top + traj_height) / traj_height
            annotprops['xytext'] = (arrow_12_width + step_1_width) / step_1_width, 0.5 + (step_2_upper_top - step_1_top) / traj_height
            arrowprops = {**def_arrowprops, 'connectionstyle': connectionstyle}
            ax.annotate('', **annotprops, arrowprops=arrowprops,)

            annotprops['xytext'] = (arrow_12_width + step_1_width) / step_1_width, (step_2_lower_top - traj_height / 2 - (step_1_top - traj_height)) / traj_height
            connectionstyle = 'bar,angle=90,fraction=-0.4'
            arrowprops = {**def_arrowprops, 'connectionstyle': connectionstyle}
            ax.annotate('', **annotprops, arrowprops=arrowprops,)

            connectionstyle = 'arc3,rad=0'
            ax = traj_axes[1]
            annotprops['xy'] = 1, (step_2_upper_center - (step_2_upper_top - traj_height)) / traj_height
            annotprops['xytext'] = (short_arrow_width + step_2_width) / step_2_width, (step_2_upper_center - (step_2_upper_top - traj_height)) / traj_height
            arrowprops = {**def_arrowprops, 'connectionstyle': connectionstyle}
            ax.annotate('', **annotprops, arrowprops=arrowprops,)

            annotprops['xy'] = (short_arrow_width + heatmap_width + step_2_width) / step_2_width, (step_2_upper_center - (step_2_upper_top - traj_height)) / traj_height
            annotprops['xytext'] = (2 * short_arrow_width + heatmap_width + step_2_width) / step_2_width, (step_2_upper_center - (step_2_upper_top - traj_height)) / traj_height
            arrowprops = {**def_arrowprops, 'connectionstyle': connectionstyle}
            ax.annotate('', **annotprops, arrowprops=arrowprops,)


            ax = traj_axes[2]
            annotprops['xy'] = 1, (step_2_lower_center - (step_2_lower_top - traj_height)) / traj_height
            annotprops['xytext'] = (short_arrow_width + step_2_width) / step_2_width, (step_2_lower_center - (step_2_lower_top - traj_height)) / traj_height
            arrowprops = {**def_arrowprops, 'connectionstyle': connectionstyle}
            ax.annotate('', **annotprops, arrowprops=arrowprops,)

            annotprops['xy'] = (short_arrow_width + heatmap_width + step_2_width) / step_2_width, (step_2_lower_center - (step_2_lower_top - traj_height)) / traj_height
            annotprops['xytext'] = (2 * short_arrow_width + heatmap_width + step_2_width) / step_2_width, (step_2_lower_center - (step_2_lower_top - traj_height)) / traj_height
            arrowprops = {**def_arrowprops, 'connectionstyle': connectionstyle}
            ax.annotate('', **annotprops, arrowprops=arrowprops,)

            ax = traj_axes[3]
            annotprops['xy'] = 1, 0.5
            annotprops['xytext'] = (arrow_34_width + step_3_width) / step_3_width, (step_4_center - step_3_upper_bottom) / traj_height
            connectionstyle = 'bar,angle=90,fraction=-0.29'
            arrowprops = {**def_arrowprops, 'connectionstyle': connectionstyle}
            ax.annotate('', **annotprops, arrowprops=arrowprops,)

            ax = traj_axes[4]
            annotprops['xy'] = 1, 0.5
            annotprops['xytext'] = (arrow_34_width + step_3_width) / step_3_width, (step_4_center - step_3_lower_bottom) / traj_height
            connectionstyle = 'bar,angle=90,fraction=0.29'
            arrowprops = {**def_arrowprops, 'connectionstyle': connectionstyle}
            ax.annotate('', **annotprops, arrowprops=arrowprops,)

    return fig, traj_axes, D_axes, segmentedD_axes


def get_distance(xy_1, xy_2, ax_ratio=1):
    """Compute Euclidean distance between two coordinates. First convert x-coordinates to y-scale. The result is expressed in y-scale."""
    dx = (xy_2[0] - xy_1[0]) * ax_ratio
    dy = xy_2[1] - xy_1[1]
    return math.sqrt(dx ** 2 + dy ** 2)


def get_fraction_to_connect_at_mid(xy_1, xy_2, angle, ax_ratio=1):
    x1, y1 = xy_1
    x2, y2 = xy_2
    dd = get_distance(xy_1, xy_2, ax_ratio=ax_ratio)
    theta1 = math.atan2(y2 - y1, x2 - x1)
    theta0 = np.deg2rad(angle)
    dtheta = theta1 - theta0
    dl = dd * math.sin(dtheta)
    armB = - dl
    f = 0.5 * (y1 - y2)
    fraction = (f - max(armB, 0)) / dd
    print('fraction * dd = ', fraction * dd)
    return fraction


def plot_figure_method_overview(reconstruction, plot_style=1, plot_single_column=False, plot_arrow=True, plot_arrow_from_traj_center=False, save_file=None):

    if plot_style == 1:
        fig, traj_axes, D_axes, segmentedD_axes = get_axes_for_method_overview(plot_single_column=plot_single_column, plot_arrow=plot_arrow, plot_arrow_from_traj_center=plot_arrow_from_traj_center)
    elif plot_style == 2:
        fig, traj_axes, D_axes, segmentedD_axes = get_axes_for_method_overview_2(plot_single_column=plot_single_column, plot_arrow=plot_arrow, reconstruction=reconstruction)

    # Plain trajectories before inferring clonal structure
    traj_ylabel = 'Frequency'
    traj_list = [reconstruction.traj] + [period.traj for period in reconstruction.periods]
    times_list = [reconstruction.times] + [period.times for period in reconstruction.periods]
    for i, ax in enumerate(traj_axes[:3]):
        plt.sca(ax)
        traj, times = traj_list[i], times_list[i]
        AP.plotTraj(traj, times=times, colors=['grey'] * len(traj[0]), alpha=0.6, linewidth=SIZELINE, fontsize=SIZELABEL, title='', plotFigure=False, plotShow=False)
        xticks = [times[0], times[-1]]
        if i == 0:
            set_ticks_labels_axes(xlim=None, ylim=None, xticks=xticks, xticklabels=xticks, ylabel=traj_ylabel)
        else:
            set_ticks_labels_axes(xlim=None, ylim=None, xticks=xticks, xticklabels=xticks, ylabel=None, yticks=[], yticklabels=[])
        # plt.text(x=sublabel_x, y=sublabel_y, s=SUBLABELS[0], transform=ax.transAxes, **DEF_SUBLABELPROPS)

    # Colored trajectories
    period_list = reconstruction.periods
    clade_index_offset = 0
    ancestor_clade_index = -1
    for i, ax in enumerate(traj_axes[3:5]):
        plt.sca(ax)
        period = period_list[i]
        times = period.times
        AP.plotCladeAndMutFreq_overview(period, clade_index_offset=clade_index_offset, ancestor_clade_index=ancestor_clade_index, fontsize=SIZELABEL, legendsize=SIZELEGEND, title='', plotFigure=False, plotShow=False)
        ancestor_clade_index = clade_index_offset + np.argmax(period.cladeFreq[-1])
        xticks = [times[0], times[-1]]
        clade_index_offset += period.numClades
        set_ticks_labels_axes(xlim=None, ylim=None, xticks=xticks, xticklabels=xticks, ylabel=None, yticks=[], yticklabels=[])

    plt.sca(traj_axes[5])
    times = reconstruction.times
    xticks = [times[0], times[-1]]
    AP.plotCladeAndMutFreq_overview(reconstruction, clade_index_offset=0, ancestor_clade_index=-1, fontsize=SIZELABEL, legendsize=SIZELEGEND, title='', plotFigure=False, plotShow=False)
    set_ticks_labels_axes(xlim=None, ylim=None, xticks=xticks, xticklabels=xticks, ylabel=None, yticks=[], yticklabels=[])

    # Indicate detection of fixation event here
    if plot_style == 2 and not plot_single_column:
        plt.sca(traj_axes[6])
        traj, times = traj_list[0], times_list[0]
        AP.plotTraj(traj, times=times, colors=['grey'] * len(traj[0]), alpha=0.6, linewidth=SIZELINE, fontsize=SIZELABEL, title='', plotFigure=False, plotShow=False)
        xticks = [times[0], times[-1]]
        set_ticks_labels_axes(xlim=None, ylim=None, xticks=xticks, xticklabels=xticks, ylabel=traj_ylabel)
        t_fixed = reconstruction.periodBoundaries[0][1]
        y_lower, y_upper = -0.2, 1.2
        plt.plot((t_fixed, t_fixed), (y_lower, y_upper), linestyle='dashed', linewidth=SIZELINE * 1.5, color='red')
        plt.text(t_fixed - 150, y_lower, 'Fixation', color='red', fontsize=SIZELABEL)

    period_list = [reconstruction] + reconstruction.periods

    if D_axes is not None:
        vmin, vmax = None, None
        for i, ax in enumerate(D_axes):
            plt.sca(ax)
            period = period_list[i]
            AP.plotIntDxdx(period.intWeightedDxdx, plot_cbar=False, vmin=vmin, vmax=vmax, plot_xticks=False, plotFigure=False, plotShow=False)

    vmin, vmax = None, None
    for i, ax in enumerate(segmentedD_axes):
        plt.sca(ax)
        period = period_list[i]
        AP.plotSegmentedIntDxdx(period, plot_cbar=False, vmin=vmin, vmax=vmax, plot_xticks=False, plot_yticks=False, plotFigure=False, plotShow=False)

    plt.show()
    if save_file:
        fig.savefig(save_file, facecolor=fig.get_facecolor(), **DEF_FIGPROPS)


def save_subfigure_for_method_overview(reconstruction, clade_colors=None, save_file_prefix=None, postfix='.pdf', def_saveprops={}, linewidth=2*SIZELINE, cladeFreqLinestyle='solid', alleleFreqAlpha=0.6, ylim=(0.001, 0.999)):

    # cladeFreqLinestyle=(0, (3, 3))

    if clade_colors is None:
        clade_colors = sns.husl_palette(reconstruction.numClades)

    traj_width = 2
    traj_height = traj_width * 0.35
    heatmap_height = traj_height
    period_traj_width = traj_width * 0.6

    # Plain trajectories before inferring clonal structure
    traj_list = [reconstruction.traj] + [period.traj for period in reconstruction.periods]
    times_list = [reconstruction.times] + [period.times for period in reconstruction.periods]
    for i, (traj, times) in enumerate(zip(traj_list, times_list)):
        if i == 0:
            figsize = (traj_width, traj_height)
        else:
            figsize = (period_traj_width, traj_height)
        fig = AP.plotTraj(traj, times=times, ylim=ylim, colors=['grey'] * len(traj[0]), alpha=alleleFreqAlpha, linewidth=linewidth, fontsize=SIZELABEL, title='', plotFigure=True, figsize=figsize, returnFig=True, plotShow=False)
        ax = plt.gca()
        set_ticks_labels_axes(xlim=None, ylim=None, xticks=[], xticklabels=[], ylabel=None, yticks=[], yticklabels=[])
        plt.subplots_adjust(0, 0, 1, 1)
        plt.axis('off')
        if save_file_prefix is not None:
            fig.savefig(f'{save_file_prefix}-traj-{i}' + postfix, facecolor=fig.get_facecolor(), **DEF_FIGPROPS, **def_saveprops)

    # Colored trajectories
    clade_index_offset = 0
    ancestor_clade_index = -1
    for i, period in enumerate(reconstruction.periods):
        fig = AP.plotCladeAndMutFreq_overview(period, colors=clade_colors, ylim=ylim, clade_index_offset=clade_index_offset, ancestor_clade_index=ancestor_clade_index, fontsize=SIZELABEL, cladeFreqLinestyle=cladeFreqLinestyle, alleleFreqAlpha=alleleFreqAlpha, linewidth=linewidth, plotLegend=False, legendsize=SIZELEGEND, title='', plotFigure=True, figsize=(period_traj_width, traj_height), returnFig=True, plotShow=False)
        ax = plt.gca()
        ancestor_clade_index = clade_index_offset + np.argmax(period.cladeFreq[-1])
        clade_index_offset += period.numClades
        set_ticks_labels_axes(xlim=None, ylim=None, xticks=[], xticklabels=[], ylabel=None, yticks=[], yticklabels=[])
        plt.subplots_adjust(0, 0, 1, 1)
        plt.axis('off')
        if save_file_prefix is not None:
            fig.savefig(f'{save_file_prefix}-traj-{i + len(traj_list)}' + postfix, facecolor=fig.get_facecolor(), **DEF_FIGPROPS, **def_saveprops)

    fig = AP.plotCladeAndMutFreq_overview(reconstruction, colors=clade_colors, ylim=ylim, clade_index_offset=0, ancestor_clade_index=-1, fontsize=SIZELABEL, cladeFreqLinestyle=cladeFreqLinestyle, alleleFreqAlpha=alleleFreqAlpha, linewidth=linewidth, plotLegend=False, legendsize=SIZELEGEND, title='', plotFigure=True, figsize=(traj_width, traj_height), returnFig=True, plotShow=False)
    ax = plt.gca()
    set_ticks_labels_axes(xlim=None, ylim=None, xticks=[], xticklabels=[], ylabel=None, yticks=[], yticklabels=[])
    plt.subplots_adjust(0, 0, 1, 1)
    plt.axis('off')
    if save_file_prefix is not None:
        fig.savefig(f'{save_file_prefix}-traj-{len(reconstruction.periods) + len(traj_list)}' + postfix, facecolor=fig.get_facecolor(), **DEF_FIGPROPS, **def_saveprops)


    # Legends
    fig = plt.figure(figsize=(period_traj_width, traj_height))
    lines = [matplotlib.lines.Line2D([0], [0], color=GREY_COLOR_HEX, linestyle=cladeFreqLinestyle, linewidth=linewidth), 
             matplotlib.lines.Line2D([0], [0], color=GREY_COLOR_HEX, linewidth=linewidth, alpha=alleleFreqAlpha)]
    ax = plt.gca()
    ax.legend(lines, ['Clade frequency', 'Allele frequency'], fontsize=SIZELEGEND, frameon=False)
    plt.subplots_adjust(0, 0, 1, 1)
    plt.axis('off')
    if save_file_prefix is not None:
        fig.savefig(f'{save_file_prefix}-legend' + postfix, facecolor=fig.get_facecolor(), **DEF_FIGPROPS, **def_saveprops)

    # Indicate detection of fixation event here
    traj, times = traj_list[0], times_list[0]
    fig = AP.plotTraj(traj, times=times, ylim=ylim, colors=[GREY_COLOR_HEX] * len(traj[0]), alpha=alleleFreqAlpha, linewidth=linewidth, fontsize=SIZELABEL, title='', plotFigure=True, figsize=(traj_width, traj_height), returnFig=True, plotShow=False)
    ax = plt.gca()
    set_ticks_labels_axes(xlim=None, ylim=None, xticks=[], xticklabels=[], ylabel=None)
    plt.subplots_adjust(0, 0, 1, 1)
    plt.axis('off')
    t_fixed = reconstruction.periodBoundaries[0][1]
    y_lower, y_upper = -0.2, 1.2
    plt.plot((t_fixed, t_fixed), (y_lower, y_upper), linestyle=(0, (5, 1)), linewidth=linewidth * 1.5, color='#4472c4', alpha=1)
    plt.text(t_fixed - 150, y_lower, 'Fixation', color='red', fontsize=SIZELABEL)
    if save_file_prefix is not None:
        fig.savefig(f'{save_file_prefix}-traj-fixation' + postfix, facecolor=fig.get_facecolor(), **DEF_FIGPROPS, **def_saveprops)


    # Segmented int dxdx matrix
    scale_list = [0.5, 0.5, 0.5]
    for i, period in enumerate([reconstruction] + reconstruction.periods):
        # scale = scale_list[i]
        # segmentedIntDxdx, _ = period.segmentedIntDxdx
        # segmentedIntDxdx = normalize_segmentedIntDxdx(segmentedIntDxdx)
        # offDiagonalTerms = EC.get_off_diagonal_terms(segmentedIntDxdx)
        vmin, vmax = None, None
        fig = AP.plotSegmentedIntDxdx(period, normalize=True, segment_line_color='#7f7f7f', alpha=1, plot_cbar=False, vmin=vmin, vmax=vmax, plot_xticks=False, plot_yticks=False, plotFigure=True, figsize=(heatmap_height, heatmap_height), returnFig=True, plotShow=False)
        ax = plt.gca()
        plt.subplots_adjust(0, 0, 1, 1)
        plt.axis('off')
        if save_file_prefix is not None:
            fig.savefig(f'{save_file_prefix}-segmenetdD-{i}' + postfix, facecolor=fig.get_facecolor(), **DEF_FIGPROPS, **def_saveprops)

    plt.close('all')


def get_genotype_info_from_simulation(simulation, reconstruction):

    times = simulation['times']
    T, L, N, genotypes, genotype_frequencies, pop = parse_info_from_simulation(simulation)
    genotype_index_to_clade, clade_to_genotype_indices = map_genotype_index_to_clade(genotypes, reconstruction)
    return genotypes, genotype_frequencies, genotype_index_to_clade, clade_to_genotype_indices


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
    # ylabels = ['Mutation\nfrequency', 'Haplotype\nproportion', 'Haplotype frequency and\ninferred clade frequency']
    ylabels = ['Mutation\nfrequency', 'Haplotype\nproportion', 'Haplotype\nand clade\nfrequency']
    # ylabel = 'Frequency'
    xlabel = 'Generation'
    size_annotation = SIZEANNOTATION_SINGLE if plot_single_column else SIZEANNOTATION
    sublabel_x, sublabel_y = -0.186, 1.05

    plt.sca(fig.add_subplot(gs_base[0,:]))
    traj = simulation['traj']
    ax = plt.gca()
    AP.plotTraj(traj, linewidth=SIZELINE, colors=allele_colors, alpha=alpha, plotFigure=False, plotShow=False, title=None)
    set_ticks_labels_axes(xlim=xlim, ylim=ylim, xticks=[], xticklabels=[], ylabel=ylabels[0])
    plt.text(x=sublabel_x, y=sublabel_y, s=SUBLABELS[0], transform=ax.transAxes, **DEF_SUBLABELPROPS)

    bottom_axes = [fig.add_subplot(gs_base[1, :]), fig.add_subplot(gs_base[2, :])]

    plt.sca(bottom_axes[1 if annotate_together else 0])
    traj = simulation['traj']
    ax = plt.gca()
    AP.plotTraj(genotype_frequencies.T, linewidth=SIZELINE, linestyle='dashed', alpha=alpha, colors=genotype_colors, plotFigure=False, plotShow=False, title=None)
    AP.plotTraj(reconstruction.cladeFreqWithAncestor, linewidth=SIZELINE * 2, alpha=alpha, colors=clade_colors, plotFigure=False, plotShow=False, title=None)
    plt.plot([0, 0.0001], [-1, -1], linewidth=SIZELINE, linestyle='dashed', color=GREY_COLOR_RGB, alpha=alpha, label='Haplotype')
    plt.plot([0, 0.0001], [-1, -1], linewidth=SIZELINE * 2, color=GREY_COLOR_RGB, alpha=alpha, label='Clade')
    if annotate_together:
        ax.legend(fontsize=SIZELEGEND, loc='center left', bbox_to_anchor=(-0.01, -0.6), frameon=False)
        plt.text(x=sublabel_x, y=sublabel_y, s=SUBLABELS[2], transform=ax.transAxes, **DEF_SUBLABELPROPS)
    else:
        ax.legend(fontsize=SIZELEGEND, loc='center left', bbox_to_anchor=(-0.01, -0.6), frameon=False)
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

    set_ticks_labels_axes(xlim=xlim, xlabel=xlabel if annotate_together else None, ylim=ylim, xticks=[] if not annotate_together else None, xticklabels=[] if not annotate_together else None, ylabel=ylabels[2 if annotate_together else 1])

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

    set_ticks_labels_axes(xlim=xlim, ylim=ylim_for_stackplot, ylabel=ylabels[1 if annotate_together else 2], xlabel=xlabel if not annotate_together else None, xticks=[] if annotate_together else None, xticklabels=[] if annotate_together else None)

    if plot_single_column:
        plt.subplots_adjust(0.17, 0.17, 0.92, 0.95)
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
    selection_recovered = inference[OUR_METHOD_NAME][2]
    selection_trueCov = inference[TRUE_COV_NAME][2]
    selection_estCov = inference[EST_METHOD_NAME][2]
    selection_SL = inference['SL'][2]
    selection_list = [selection_trueCov, selection_recovered, selection_SL]
    xlabel = 'True selections'
    if compare_with_SL:
        ys_list = [selection_trueCov, selection_recovered, selection_SL]
        xs_list = [selection_true, selection_true, selection_true]
    else:
        ys_list = [selection_trueCov, selection_recovered, selection_recovered]
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
            fitness_recovered = RC.computeFitnessOfGenotypes(genotypes, selection_recovered)
            plot_comparison(fitness_true, fitness_recovered, xlabel, ylabel, label='Recovered' if plot_true_cov else None, alpha=alpha, ylim=ylim, yticks=yticks, xticks=yticks, plot_title=not plot_true_cov)
        else:
            plot_comparison(selection_true, selection_recovered, xlabel, ylabel, label='Recovered' if plot_true_cov else None, alpha=alpha, ylim=ylim, yticks=yticks, xticks=yticks, plot_title=not plot_true_cov)
        if plot_true_cov:
            plt.scatter(selection_true, selection_trueCov, s=SIZEDOT, alpha=alpha, label='True_cov')
            plt.legend(fontsize=SIZELEGEND)
    elif compare_with_true:
        ylabels = ['Selections inferred\nwith true covariance', 'Selections inferred with\nreconstructed covariance', 'Selections inferred when\nignoring LD']
        for i, ys in enumerate(selection_list):
            ylabel = ylabels[i]
            plt.sca(axes[i, 1])
            plot_comparison(selection_true, ys, xlabel, ylabel, alpha=alpha, ylim=ylim, yticks=yticks, xticks=yticks if i == nRow - 1 else [])
    elif plot_genotype_fitness:
        ylabels = ['Fitness inferred\nwith true covariance', 'Fitness inferred with\nreconstructed covariance', 'Fitness inferred when\nignoring LD']
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


def plot_figure_performance_Evoracle_example(simulation, reconstruction, evaluation, results_evoracle, nRow=3, nCol=2, plot_genotype_fitness=True, threshold=0, ylim=(0.95, 1.25), yticks=(1.0, 1.1, 1.2), alpha=0.6, title_pad=4, save_file=None):

    T, L, N, genotypes, genotype_frequencies, pop = parse_info_from_simulation(simulation)

    w = SINGLE_COLUMN
    goldh = w * 1 if nRow == 2 else w * 1.3
    fig, axes = plt.subplots(nRow, nCol, figsize=(w, goldh))
    sublabel_x, sublabel_y = -0.25, 1.08

    if nRow == 2:
        heatmaps = [axes[0, 0], axes[0, 1], axes[1, 0]]
    else:
        heatmaps = [axes[0, 0], axes[1, 0], axes[2, 0]]
    cov_list = [reconstruction.intCov, results_evoracle['int_cov'], results_evoracle['int_cov'] - reconstruction.intCov]
    # title_list = [r"True covariance, $C$", r"Recovered covariance, $\hat{C}$", "Error of recovered\ncovariance, $\hat{C}-C$"]
    title_list = [r"True covariance, $C$", r"Covariance inferred by Evoracle, $\hat{C}$", "Error matrix, $\hat{C}-C$"]
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
    selection_trueCov = inference[TRUE_COV_NAME][2]
    selection_evoracle = results_evoracle['selection']
    # selection_recovered = inference[OUR_METHOD_NAME][2]
    # selection_estCov = inference[EST_METHOD_NAME][2]
    selection_SL = inference['SL'][2]
    selection_list = [selection_trueCov, selection_evoracle, selection_SL]

    if plot_genotype_fitness:
        T, L = simulation['traj'].shape
        N = np.sum(simulation['nVec'][0])
        genotypes, _, _ = AP.getDominantGenotypeTrajectories(simulation, threshold=threshold, T=T, totalPopulation=N)
        fitness_true = RC.computeFitnessOfGenotypes(genotypes, selection_true)
        xlabel, ylabel = 'True genotype fitness', 'Inferred genotype fitness'
        fitness_list = [RC.computeFitnessOfGenotypes(genotypes, selection) for selection in selection_list]
        fitness_trueCov, fitness_recovered, fitness_SL = tuple(fitness_list)
        ys_fitness_list = fitness_list
        xs_fitness_list = [fitness_true] * 3
        fitness_values = np.concatenate([fitness_true] + fitness_list)
        if ylim is None:
            ylim = [np.min(fitness_values) - 0.01, np.max(fitness_values) + 0.01]

    if plot_genotype_fitness:
        ylabels = ['Fitness inferred\nwith true covariance', 'Fitness inferred with\ninferred covariance', 'Fitness inferred when\nignoring LD']
        xlabel = 'True fitness'
        for i, (xs, ys) in enumerate(zip(xs_fitness_list, ys_fitness_list)):
            ylabel = ylabels[i]
            plt.sca(axes[i, 1])
            ax = plt.gca()
            plot_comparison(xs, ys, xlabel if i == nRow - 1 else None, ylabel, alpha=alpha, ylim=ylim, yticks=yticks, xticks=yticks, xticklabels=yticks if i == nRow - 1 else [], plot_title=False)
            plt.text(x=0.05, y=0.9, s="Spearman's " + r'$\rho$' + ' = %.2f' % stats.spearmanr(xs, ys)[0], transform=ax.transAxes, fontsize=SIZELABEL)
            plt.text(x=sublabel_x, y=sublabel_y, s=SUBLABELS[3 + i], transform=ax.transAxes, **DEF_SUBLABELPROPS)

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


def plot_figure_performance_on_simulated_data_helper(ax, yspine_position=0.05, xspine_position=-0.05):

    ax.tick_params(**DEF_TICKPROPS)
    plt.setp(ax.spines.values(), **DEF_AXPROPS)
    # ax.spines['left'].set_position(('data', yspine_position))
    # ax.spines['bottom'].set_position(('data', xspine_position))
    ax.spines['left'].set_position(('axes', yspine_position))
    ax.spines['bottom'].set_position(('axes', xspine_position))


def plot_figure_performance_on_simulated_data(MAE_cov, Spearmanr_cov, MAE_selection, Spearmanr_selection, MAE_fitness=None, Spearmanr_fitness=None, method_list=METHODS, two_columns=False, plot_legend=False, use_pearsonr=False, evaluate_fitness=False, annot=False, ymax_MAE_covariance=4, ymax_MAE_fitness=0.08, ymin_spearmanr_fitness=0, save_file=None):
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
    box_left  = 0.21
    box_right = 0.995 if two_columns else 0.94
    wspace    = 0.5

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
        ylim_list = [[0, ymax_MAE_covariance * (4.1 / 4)], [0, 1.1], [0, 0.041], [0, 1.1], [0, ymax_MAE_fitness * (0.082 / 0.08)], [ymin_spearmanr_fitness, 1.1]]
        yticks_list = [[0, int(ymax_MAE_covariance / 2) if ymax_MAE_covariance > 1 else ymax_MAE_covariance / 2, ymax_MAE_covariance], [0, 0.5, 1], [0, 0.02, 0.04], [0, 0.5, 1], [0, ymax_MAE_fitness / 2, ymax_MAE_fitness], [0, 0.5, 1]]
        yticklabels_list = [['0', int(ymax_MAE_covariance / 2) if ymax_MAE_covariance > 1 else ymax_MAE_covariance / 2, r'$\geq$' + f'{ymax_MAE_covariance}'], ['0', '0.5', '1'], ['0', '0.02', r'$\geq$' + '0.04'], ['0', '0.5', '1'], ['0', str(ymax_MAE_fitness / 2), r'$\geq$' + str(ymax_MAE_fitness)], ['0', '0.5', '1']]
        ceil_list = [ymax_MAE_covariance, None, 0.04, None, ymax_MAE_fitness, None]
        floor_list = [None, None, None, None, None, None]
    else:
        ylim_list = [[0, 4.4], [0, 1.1], [0, 0.088] if evaluate_fitness else [0, 0.044], [0, 1.1]]
        yticks_list = [[0, 2, 4], [0, 0.5, 1], [0, 0.04, 0.08] if evaluate_fitness else [0, 0.02, 0.04], [0, 0.5, 1]]
        yticklabels_list = [['0', '2', r'$\geq 4$'], ['0', '0.5', '1'], ['0', '0.04', '0.08'] if evaluate_fitness else ['0', '0.02', '0.04'], ['0', '0.5', '1']]
        ceil_list = [4, None, 0.08 if evaluate_fitness else 0.04, None]
        floor_list = [None, None, None, None]
    if MAE_fitness is not None and Spearmanr_fitness is not None:
        if two_columns:
            ylabel_list = ['MAE of recovered\ncovariances', LABEL_SPEARMANR_COVARIANCE_FOUR_2, 'MAE of inferred\nselection coefficients', LABEL_SPEARMANR_FOUR_2, LABEL_MAE_FITNESS, LABEL_SPEARMANR_FITNESS_FOUR_2]
        else:
            ylabel_list = ['MAE of recovered\ncovariances', LABEL_SPEARMANR_COVARIANCE_FOUR, 'MAE of inferred\nselection coefficients', LABEL_SPEARMANR_FOUR, LABEL_MAE_FITNESS, LABEL_SPEARMANR_FITNESS_FOUR]
    else:
        if use_pearsonr:
            if evaluate_fitness:
                ylabel_list = ['MAE of recovered\ncovariances', LABEL_PEARSONR_COVARIANCE_FOUR, LABEL_MAE_FITNESS, LABEL_PEARSONR_FITNESS_FOUR]
            else:
                ylabel_list = ['MAE of recovered\ncovariances', LABEL_PEARSONR_COVARIANCE_FOUR, 'MAE of inferred\nselection coefficients', LABEL_PEARSONR_FOUR]
        else:
            if evaluate_fitness:
                ylabel_list = ['MAE of recovered\ncovariances', LABEL_SPEARMANR_COVARIANCE_FOUR, LABEL_MAE_FITNESS, LABEL_SPEARMANR_FITNESS_FOUR]
            else:
                ylabel_list = ['MAE of recovered\ncovariances', LABEL_SPEARMANR_COVARIANCE_FOUR, 'MAE of inferred\nselection coefficients', LABEL_SPEARMANR_FOUR]

    xs = np.arange(0, len(method_list))
    xlim = [-1, len(method_list) + 0.4] if two_columns else [-0.8, len(method_list) + 0.4]
    sublabel_x, sublabel_y = -0.6, 1.15

    ## set colors and methods list

    fc        = '#ff6666'  #'#EB4025'
    ffc       = '#ff6666'  #'#EB4025'
    hc        = '#FFB511'
    nc        = '#E8E8E8'
    hfc       = '#ffcd5e'
    nfc       = '#f0f0f0'
    methods   = method_list
    xticklabels = methods
    # xticklabels = ['0', '1', '2', '3', '4' ] if two_columns else METHODS
    # colorlist   = [   fc,    hc,    nc,      nc,      nc,         nc,      nc,    nc]
    # fclist      = [  ffc,   hfc,   nfc,     nfc,     nfc,        nfc,     nfc,   nfc]
    colorlist = [fc, hc] + [nc] * (len(methods) - 2)
    fclist = [ffc, hfc] + [nfc] * (len(methods) - 2)
    eclist = [BKCOLOR] * len(methods)

    hist_props = dict(lw=SIZELINE/2, width=0.7, align='center', orientation='vertical',
                      edgecolor=[BKCOLOR for i in range(len(methods))])

    for row, metrics in enumerate(metrics_list):
        ylim = ylim_list[row]
        yticks, yticklabels, ylabel = yticks_list[row], yticklabels_list[row], ylabel_list[row]
        floor, ceil = floor_list[row], ceil_list[row]

        ys = [metrics[method] for method in method_list]
        max_length = np.max([len(_) for _ in ys])
        for ys_ in ys:
            if len(ys_) < max_length:
                ys_ += [np.mean(ys_)] * (max_length - len(ys_))
        y_avgs = np.array([np.mean(_) for _ in ys])
        ax = axes[row]

        na_fontsize = SIZELABEL
        na_text_offset = -0.46

        if row == 1:
            scatter_indices = np.array([_ for _ in range(len(ys)) if method_list[_] not in [TRUE_COV_NAME, 'SL']])  # Spearmanr of covariances does not apply to the SL/MPL method

            na_xy_list = [(0, 0.15)]
            if 'SL' in method_list:
                na_xy_list.append((len(method_list) - 1, 0.15))
            plt.sca(ax)
            for na_x, na_y in na_xy_list:
                plt.plot([na_x, na_x], [0, na_y - 0.05], linewidth=SIZELINE, color=BKCOLOR)
                plt.text(na_x + na_text_offset, na_y, 'NA', fontsize=na_fontsize)

            y_avgs[0] = None
            if 'SL' in method_list:
                y_avgs[-1] = None

        elif row == 0:
            scatter_indices = np.array([_ for _ in range(len(ys)) if method_list[_] not in [TRUE_COV_NAME, 'SL']])

            na_xy_list = [(0, 0.6 * (ymax_MAE_covariance / 4))]
            if 'SL' in method_list:
                na_xy_list.append((len(method_list) - 1, 0.6))
            plt.sca(ax)
            for na_x, na_y in na_xy_list:
                plt.plot([na_x, na_x], [0, na_y - 0.2 * (ymax_MAE_covariance / 4)], linewidth=SIZELINE, color=BKCOLOR)
                plt.text(na_x + na_text_offset, na_y, 'NA', fontsize=na_fontsize)
            plt.sca(ax)

            y_avgs[0] = None
            if 'SL' in method_list:
                y_avgs[-1] = None

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
        # if annot and (row == 3 or row == 2):
        if annot:
            for x, y in zip(xs, y_avgs):
                plt.sca(ax)
                # annotation = '%.2f' % y if row == 3 else ('%.3f' % y)
                annotation = '%.3f' % y
                plt.text(x - 0.3, min(y * 1.2, ceil * 1.2) if ceil is not None else y * 1.2, annotation, fontsize=SIZELABEL / 1.5)

        del pprops['colors']

        ys = np.array(ys)
        if floor is not None:
            ys[ys <= floor] = floor
        if ceil is not None:
            ys[ys >= ceil] = ceil

        # for ys_ in ys:
        #     if floor is not None:
        #         for i, y_ in enumerate(ys_):
        #             if y_ <= floor:
        #                 ys_[i] = floor
        #     if ceil is not None:
        #         for i, y_ in enumerate(ys_):
        #             if y_ >= ceil:
        #                 ys_[i] = ceil

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


def plot_scatter_and_line(xs, ys, color=None, s=SIZEDOT, label=None, edgecolors=EDGECOLORS, marker='.', alpha=1, linewidth=SIZELINE):
    plt.scatter(xs, ys, s=s, marker=marker, color=color, edgecolors=edgecolors, label=label, alpha=alpha)
    plt.plot(xs, ys, color=color, linewidth=linewidth, alpha=alpha)


def plot_figure_performance_on_simulated_data_with_temporal_subsampling(MAE_cov_list, Spearmanr_cov_list, MAE_selection_list, Spearmanr_selection_list, MAE_fitness_list, Spearmanr_fitness_list, sampling_intervals, method_list=METHODS, plot_legend=False, use_pearsonr=False, evaluate_fitness=False, annot=False, ymax_MAE_fitness=0.1, ymin_spearmanr_fitness=0.89, x_log_scale=True, save_file=None):
    """
    Comparisons versus existing methods of covariance recovery and selection inference.
    """

    w       = DOUBLE_COLUMN
    hshrink = 1.5
    goldh   = 0.60 * w * hshrink
    fig     = plt.figure(figsize=(w, goldh))

    box_top   = 0.98
    box_left  = 0.11
    box_right = 0.995
    wspace    = 0.35

    ddy = 0.1 / hshrink
    dy = 0.39 / hshrink  # Adjusts height of each subplot, & height of white space below the subplots.

    metrics_lists = [MAE_cov_list, Spearmanr_cov_list, MAE_selection_list, Spearmanr_selection_list, MAE_fitness_list, Spearmanr_fitness_list]
    ylim_list = [[0, 4.1], [0, 1.1], [0, 0.041], [0.25, 0.75], [0, ymax_MAE_fitness * (0.082 / 0.08)], [ymin_spearmanr_fitness, 1.01]]
    yticks_list = [[0, 2, 4], [0, 0.5, 1], [0, 0.02, 0.04], [0.3, 0.5, 0.7], [0, ymax_MAE_fitness / 2, ymax_MAE_fitness], [0, 0.5, 1] if ymin_spearmanr_fitness == 0 else [0.9, 0.95, 1.0]]
    yticklabels_list = [['0', '2', r'$\geq$' + '4'], ['0', '0.5', '1'], ['0', '0.02', '0.04'], ['0.25', '0.5', '0.75'], ['0', str(ymax_MAE_fitness / 2), str(ymax_MAE_fitness)], [str(_) for _ in yticks_list[-1]]]
    ylabel_list = ['MAE of recovered\ncovariances', LABEL_SPEARMANR_COVARIANCE_FOUR_2, 'MAE of inferred\nselection coefficients', LABEL_SPEARMANR_FOUR_2, LABEL_MAE_FITNESS, LABEL_SPEARMANR_FITNESS_FOUR_2]


    sublabel_x, sublabel_y = -0.1, 1.02
    xlabel = 'Time intervals between sampling\n' + DELTA_G + ' (generation)'
    if x_log_scale:
        xs = sampling_intervals
        xlim = [sampling_intervals[0] / 1.5, sampling_intervals[-1] * 1.5]
        xspine_position, yspine_position =  0, 0
    else:
        xs = range(len(sampling_intervals))
        xlim = [-0.5, len(sampling_intervals) - 0.5]
        xspine_position, yspine_position =  -0.05, 0.05

    boxes = [dict(left=box_left,
                  right=box_right,
                  bottom=box_top-((i+1)*dy)-(i*ddy),
                  top=box_top-(i*dy)-(i*ddy)) for i in range(len(metrics_lists)//2)]
    gridspecs = [gridspec.GridSpec(1, 2, wspace=wspace, **box) for box in boxes]
    axes = []
    for _ in gridspecs:
        axes += [plt.subplot(_[0, 1]), plt.subplot(_[0, 0])]

    for i, (metrics_list, ax, ylim, yticks, yticklabels, ylabel) in enumerate(zip(metrics_lists, axes, ylim_list, yticks_list, yticklabels_list, ylabel_list)):
        plt.sca(ax)
        for j, method in enumerate(method_list):
            if i in [0, 1] and method in [TRUE_COV_NAME, 'SL']:
                continue
            ys = [metrics[method] for metrics, x in zip(metrics_list, xs)]
            plot_scatter_and_line(xs, ys, color=METHOD_COLORS[j], label=method, marker=METHOD_MARKERS[j], s=SIZEDOT)
            # plt.scatter(xs, ys, color=METHOD_COLORS[j], label=method, marker=METHOD_MARKERS[j], s=SMALLSIZEDOT)

        if x_log_scale:
            plt.xscale('log')
        plt.ylabel(ylabel, fontsize=SIZELABEL)
        if i >= 4:
            plt.xlabel(xlabel, fontsize=SIZELABEL)
        if i == 2:
            plt.legend(fontsize=SIZELEGEND, ncol=2)
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.xticks(ticks=xs, labels=xs)
        plt.yticks(ticks=yticks, labels=yticklabels, rotation=0)
        plot_figure_performance_on_simulated_data_helper(ax, xspine_position=xspine_position, yspine_position=yspine_position)
        if i % 2 == 1:
            plt.text(x=sublabel_x, y=sublabel_y, s=SUBLABELS[i // 2], transform=ax.transAxes, **DEF_SUBLABELPROPS)

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


def get_axes_for_reconstruction_example_LTEE(figsize=None, small_cov_box=False):
    ratio   = 0.36  # 0.33
    w       = DOUBLE_COLUMN
    h       = ratio * w
    if figsize is not None:
        fig = plt.figure(figsize=figsize)
    else:
        fig = plt.figure(figsize=(w, h))

    divider_width = 0.1
    box_left = 0.065
    # box_bottom = 0.14
    box_top = 0.9  # 0.95
    ddy = 0.07  # Adjust hspace
    dy = 0.35  # 0.37  # Adjusts height of each subplot. Also adjusts height of white space below the subplots.
    # Height of heatmap = cov_top - cov_bottom
    # Width of heatmap = (cov_top - cov_bottom) * (h / w)

    box_bottom = box_top - (2*dy)-(1*ddy)

    cov_right = 0.98
    cov_bottom = box_bottom
    cov_top = box_top

    cbar_width = 0.1
    cov_width = ratio * (cov_top - cov_bottom) + cbar_width
    box_middle = cov_right - divider_width - cov_width
    cov_left = box_middle + divider_width

    if small_cov_box:
        cov_right = 0.95
        cov_bottom = box_bottom + 0.15
        cov_top = box_top - 0.15
        cov_width = ratio * (cov_top - cov_bottom) + cbar_width
        box_middle_ = cov_right - divider_width - cov_width
        cov_left = box_middle_ + divider_width

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


def get_axes_for_reconstruction_example_LTEE_p6(figsize=None, small_cov_box=False):
    ratio   = 1
    w       = DOUBLE_COLUMN
    h       = ratio * w
    if figsize is not None:
        fig = plt.figure(figsize=figsize)
    else:
        fig = plt.figure(figsize=(w, h))

    divider_height = 0.08
    box_left = 0.2
    box_right = 0.8
    box_bottom = 0.1
    box_top = 0.95
    ddy = 0.07 / (ratio / 0.33)  # Adjust hspace
    dy = 0.37 / (ratio / 0.33)  # Adjusts height of each subplot. Also adjusts height of white space below the subplots.
    # Height of heatmap = cov_top - cov_bottom
    # Width of heatmap = (cov_top - cov_bottom) * (h / w)

    cov_top = box_top-(2*dy)-(1*ddy) - divider_height
    cov_bottom = box_bottom
    cov_left = box_left - 0.05
    cov_right = box_right + 0.05

    traj_boxes = [dict(left=box_left,
                  right=box_right,
                  bottom=box_top-((i+1)*dy)-(i*ddy),
                  top=box_top-(i*dy)-(i*ddy)) for i in range(2)]

    cov_box = dict(left=cov_left,
                   right=cov_right,
                   bottom=cov_bottom,
                   top=cov_top)

    boxes = traj_boxes + [cov_box]
    print(boxes)
    gridspecs = [gridspec.GridSpec(1, 1, **boxes[i]) for i in range(len(boxes))]
    axes = [plt.subplot(gridspecs[i][0, 0]) for i in range(len(gridspecs))]
    return fig, axes



def plot_figure_reconstruction_example_LTEE(pop, reconstruction, data, directory=LH.CLUSTERIZATION_OUTPUT_DIR, flattened=False, alpha=0.4, plotIntpl=True, ylim=(-0.03, 1.03), linewidth=SIZELINE, bbox_to_anchor_legend=(-0.005, 0.5), annotation_line_size=SIZEANNOTATION, legend_frameon=False, plot_legend_out_on_the_right=True, plot_dxdx_by_clades=False, plot_cov_by_clades=False, xtick_distance=40, cbar_shrink=1, save_file=None, figsize=None):

    if pop == 'p6':
        fig, axes = get_axes_for_reconstruction_example_LTEE_p6(figsize=figsize)
    else:
        fig, axes = get_axes_for_reconstruction_example_LTEE(figsize=figsize)
    xlabel, ylabel = 'Generation', 'Frequency'
    xticks = range(0, 60001, 10000)
    # tStart, tEnd = reconstruction.times[0], reconstruction.times[-1]
    tStart, tEnd = 0, 60500
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
    AP.plotTotalCladeFreq(reconstruction.cladeFreqWithAncestor, cladeMuts=reconstruction.cladeMuts, otherMuts=reconstruction.otherMuts, colorAncestorFixedRed=True, traj=reconstruction.traj, times=reconstruction.times, colors=clade_colors, alpha=alpha, plotFigure=False, plotLegend=False, plotClade=False, plotShow=False, fontsize=SIZELABEL, linewidth=SIZELINE, legendsize=SIZELEGEND)
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
    segmentedIntDxdx, groups = LH.load_dxdx_for_LTEE(pop, directory=directory)
    if plot_dxdx_by_clades:
        intDxdx = RC.undoSegmentMatrix(segmentedIntDxdx, groups)
        clades = [reconstruction.otherMuts] + reconstruction.cladeMuts
        segmentedIntDxdx, clades_sorted = RC.segmentMatrix(intDxdx, clades)
        plot_dxdx_heatmap(segmentedIntDxdx, clades, as_subplot=True, xtick_distance=xtick_distance, cbar_shrink=cbar_shrink, ylabel_prefix="Clade", plot_ylabel=False, plot_top_axis=True)
    elif plot_cov_by_clades:
        clades = [reconstruction.otherMuts] + reconstruction.cladeMuts
        segmentedIntCov, clades_sorted = RC.segmentMatrix(reconstruction.recoveredIntCov, clades)
        plot_dxdx_heatmap(segmentedIntCov, clades, as_subplot=True, xtick_distance=xtick_distance, cbar_shrink=cbar_shrink, ylabel_prefix="Clade", plot_ylabel=False, plot_top_axis=True)
    else:  # plot_dxdx_by_groups
        plot_dxdx_heatmap(segmentedIntDxdx, groups, as_subplot=True, xtick_distance=xtick_distance, cbar_shrink=cbar_shrink)
    plt.setp(ax.spines.values(), **DEF_AXPROPS)
    if pop == 'p6':
        plt.text(x=sublabel_x, y=sublabel_y, s=SUBLABELS[2], transform=ax.transAxes, **DEF_SUBLABELPROPS)
    else:
        plt.text(x=-0.07, y=0.97, s=SUBLABELS[2], transform=ax.transAxes, **DEF_SUBLABELPROPS)

    # if legend_frameon:
    #     plt.subplots_adjust(0.06, 0.15, 0.83, 0.96)
    # else:
    #     plt.subplots_adjust(0.06, 0.15, 0.99, 0.96)
    # plt.subplots_adjust(0.06, 0.15, 0.99, 0.96)
    plt.show()
    if save_file:
        fig.savefig(save_file, facecolor=fig.get_facecolor(), **DEF_FIGPROPS)



def plot_figure_reconstruction_example_LTEE_two_periods(pop, rec1, rec2, data, rec3=None, remove_clade_muts_before_period_start=False, directory=LH.CLUSTERIZATION_OUTPUT_DIR, flattened=False, alpha=0.15, alpha_shared_muts=0.3, plotIntpl=True, ylim=(-0.03, 1.03), linewidth=SIZELINE, linewidth_shared_muts=SIZELINE, bbox_to_anchor_legend=(-0.005, 0.5), annotation_line_size=SIZEANNOTATION, color_shared_muts=True, legend_frameon=False, plot_legend_out_on_the_right=True, verbose=False, save_file=None, figsize=None):

    rec1, rec2 = copy.deepcopy(rec1), copy.deepcopy(rec2)
    if rec3 is not None:
        rec3 = copy.deepcopy(rec3)

    fig, axes = get_axes_for_reconstruction_example_LTEE(figsize=figsize)
    xlabel, ylabel = 'Generation', 'Frequency'
    xticks = range(0, 60001, 10000)
    tStart, tEnd = 0, 60500
    clade_colors = {
        0: LTEE_EXTINCT_COLOR,
        1: LTEE_MAJOR_FIXED_COLOR,  # Blue
        2: LTEE_MINOR_FIXED_COLOR,  # Orange
        3: '#64af31',  # Green
        4: '#cd7af5',  # Purple
        5: 'sienna', 
        6: 'navy',
        7: 'teal',
        8: 'aqua',
        9: 'darkgreen',
        10: 'goldenrod',
    }
    if pop in EXCHANGE_FIRST_TWO_CLADES_LTEE and EXCHANGE_FIRST_TWO_CLADES_LTEE[pop]:
        clade_colors[0], clade_colors[1] = clade_colors[1], clade_colors[0]

    plt.sca(axes[0])
    ax = plt.gca()
    traj = data[pop]['traj']
    times = LH.TIMES_INTPL


    # For latter periods, remove cladeMuts if it emerge before its period
    def getEmergenceTimes(traj, times, thPresent=0.5):
        """
        Computes emergence time of all mutations.
        """
        T, L = traj.shape
        emergenceTimes = np.full(L, -1, dtype=int)
        for l in range(L):
            for t in range(T):
                if traj[t, l] > thPresent:
                    emergenceTimes[l] = times[t]
                    break
        return emergenceTimes

    def removeCladeMutsBeforePeriodStart(rec, period_no, verbose=False):
        for i, clade in enumerate(rec.cladeMuts):
            mut_index_to_keep = []
            for j, l in enumerate(clade):
                if emergenceTimes[l] < rec.times[0]:
                    rec.otherMuts.append(l)
                    if verbose:
                        print(f'Clade {i + 1} in reconstruction of period {period_no} contains mutation {l} which emerges at {emergenceTimes[l]}, earlier than the period starting time {rec.times[0]}; Now excluded.')
                else:
                    mut_index_to_keep.append(j)
            rec.cladeMuts[i] = [clade[j] for j in mut_index_to_keep]

    def countCladeMuts(rec, period_no):
        print(f"Period {period_no}", end='\t')
        for i, clade in enumerate(rec.cladeMuts):
            print(f"{len(clade)}", end='\t')
        print()


    def removeCladeWithFewMuts(rec, period_no, thNum=30):
        clade_index_to_keep = []
        for i, clade in enumerate(rec.cladeMuts):
            if len(clade) < thNum:
                for l in clade:
                    rec.otherMuts.append(l)
            else:
                clade_index_to_keep.append(i)
        rec.cladeMuts = [rec.cladeMuts[i] for i in clade_index_to_keep]
        rec.numClades = len(rec.cladeMuts)


    if remove_clade_muts_before_period_start:
        if verbose:
            print(f"Before removing mutations that emerge too early...")
            countCladeMuts(rec1, 1)
            countCladeMuts(rec2, 2)
            if rec3 is not None:
                countCladeMuts(rec3, 3)
            # AP.plotTotalCladeFreq(rec3.cladeFreqWithAncestor, traj=traj, times=times, plotClade=False, cladeMuts=rec3.cladeMuts, otherMuts=rec3.otherMuts)
        emergenceTimes = getEmergenceTimes(traj, times)
        removeCladeMutsBeforePeriodStart(rec2, 2, verbose=verbose)
        if rec3 is not None:
            removeCladeMutsBeforePeriodStart(rec3, 3, verbose=verbose)

        if verbose:
            print(f"After removing mutations that emerge too early...")
            countCladeMuts(rec1, 1)
            countCladeMuts(rec2, 2)
            if rec3 is not None:
                countCladeMuts(rec3, 3)

        removeCladeWithFewMuts(rec2, 2)
        removeCladeWithFewMuts(rec3, 3)

        if verbose:
            print(f"After removing clades with too few mutations...")
            countCladeMuts(rec1, 1)
            countCladeMuts(rec2, 2)
            if rec3 is not None:
                countCladeMuts(rec3, 3)
            # AP.plotTotalCladeFreq(rec3.cladeFreqWithAncestor, traj=traj, times=times, plotClade=False, cladeMuts=rec3.cladeMuts, otherMuts=rec3.otherMuts)


    identity_to_muts = compare_cladeMuts(rec1, rec2)
    if rec3 is not None:
        identity_to_muts_2 = compare_cladeMuts(rec2, rec3)

    cladeMuts = []
    for k in range(rec1.numClades):
        cladeMuts.append(identity_to_muts[k][-1])  # -1 is other
    for k in range(rec2.numClades):
        cladeMuts.append(identity_to_muts[-1][k])
    if rec3 is not None:
        for k in range(rec3.numClades):
            cladeMuts.append(identity_to_muts_2[-1][k])

    sharedMuts = []
    for k1 in range(rec1.numClades):
        for k2 in range(rec2.numClades):
            sharedMuts += identity_to_muts[k1][k2]

    otherMuts = identity_to_muts[-1][-1]
    if rec3 is not None:
        otherMuts = [_ for _ in otherMuts if _ in identity_to_muts_2[-1][-1]]

    if plot_legend_out_on_the_right:
        if len(cladeMuts) >= 4:
            bbox_to_anchor_legend = (1, 0.45)
        else:
            bbox_to_anchor_legend = (1, 0.5)
        xlim = (tStart - 500, tEnd + 500)
        sublabel_x, sublabel_y = -0.112, 0.957
    else:
        xlim = (tStart - 2500, tEnd + 500)
        sublabel_x, sublabel_y = -0.05, 1

    for l in otherMuts:
        if traj[-1, l] <= rec1.thExtinct:
            plt.plot(times, traj[:, l], color=clade_colors[0], linewidth=0.5 * linewidth, alpha=alpha)
        else:
            plt.plot(times, traj[:, l], color=LTEE_ANCESTOR_FIXED_COLOR, linewidth=0.5 * linewidth, alpha=alpha)
    if color_shared_muts:
        for c in range(1, len(cladeMuts) + 1):
            for l in cladeMuts[c - 1]:
                plt.plot(times, traj[:, l], color=clade_colors[c], linewidth=0.5 * linewidth, alpha=alpha)
        for l in sharedMuts:
            plt.plot(times, traj[:, l], color=LTEE_INFERRED_SHARED_COLOR, linewidth=linewidth_shared_muts, alpha=alpha_shared_muts, zorder=10)
    else:
        for k in range(1, rec1.numClades + 1):
            for l in rec1.cladeMuts[k - 1]:
                plt.plot(times, traj[:, l], color=clade_colors[k], linewidth=0.5 * linewidth, alpha=alpha)
        for k in range(rec2.numClades):
            for l in rec2.cladeMuts[k]:
                plt.plot(times, traj[:, l], color=clade_colors[k + rec1.numClades + 1], linewidth=0.5 * linewidth, alpha=alpha)
        if rec3 is not None:
            for k in range(rec3.numClades):
                for l in rec3.cladeMuts[k]:
                    plt.plot(times, traj[:, l], color=clade_colors[k + rec1.numClades + rec2.numClades + 1], linewidth=0.5 * linewidth, alpha=alpha)

    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.xticks(ticks=xticks, labels=[])

    labels = [f'Clade {c+1}' for c in range(len(cladeMuts))]
    handles = [matplotlib.lines.Line2D([], [], color=LTEE_ANCESTOR_FIXED_COLOR, label='Ancestor')]
    handles += [
        matplotlib.lines.Line2D([], [], color=LTEE_MAJOR_FIXED_COLOR, label=labels[0]),
        matplotlib.lines.Line2D([], [], color=LTEE_MINOR_FIXED_COLOR, label=labels[1])]
    handles += [matplotlib.lines.Line2D([], [], color=clade_colors[c + 1], label=labels[c]) for c in range(2, len(cladeMuts))]
    if color_shared_muts:
        handles += [matplotlib.lines.Line2D([], [], color=LTEE_INFERRED_SHARED_COLOR, label='Shared acr-\noss clades')]
    handles += [matplotlib.lines.Line2D([], [], color=LTEE_EXTINCT_COLOR, label='Extinct')]

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
    segmentedIntDxdx, groups = LH.load_dxdx_for_LTEE(pop, directory=directory)
    for l in range(len(segmentedIntDxdx)):
        segmentedIntDxdx[l, l] = 0
    site_to_index = {}
    index = 0
    for group in groups:
        for l in group:
            site_to_index[l] = index
            index += 1
    groups_new = [[] for k in range(len(cladeMuts) + 1)]
    for l in otherMuts + sharedMuts:
        groups_new[0].append(site_to_index[l])
    for k, muts in enumerate(cladeMuts):
        for l in muts:
            groups_new[k + 1].append(site_to_index[l])

    segmentedIntDxdx, groups_new = RC.segmentMatrix(segmentedIntDxdx, groups_new)
    plot_dxdx_heatmap(segmentedIntDxdx, groups_new, as_subplot=True)
    plt.setp(ax.spines.values(), **DEF_AXPROPS)
    plt.text(x=-0.07, y=0.97, s=SUBLABELS[2], transform=ax.transAxes, **DEF_SUBLABELPROPS)

    # if legend_frameon:
    #     plt.subplots_adjust(0.06, 0.15, 0.83, 0.96)
    # else:
    #     plt.subplots_adjust(0.06, 0.15, 0.99, 0.96)
    plt.show()
    if save_file:
        fig.savefig(save_file, facecolor=fig.get_facecolor(), **DEF_FIGPROPS)


def plot_identity_two_periods(pop, rec1, rec2, data, square=True, cmap=CMAP, reduce_vmax=False, plot_xticks=True, plot_yticks=True, xticklabels_rotation=0, yticklabels_rotation=0):

    ax = plt.gca()

    pop_to_vmax = {
        'm1': 1000, 'm2': 1000, 'm4': 600, 'm5': 90, 'm6': 60, 'p1': 100, 'p3': 500, 'p5': 60
    }

    num_clades_LTEE = 2
    clade_LTEE_to_identity_index = get_mapping_clade_LTEE_to_identity_index()

    identity_to_muts = compare_cladeMuts(rec1, rec2)

    cladeMuts = []
    for k in range(rec1.numClades):
        cladeMuts.append(identity_to_muts[k][-1])
    for k in range(rec2.numClades):
        cladeMuts.append(identity_to_muts[-1][k])

    sharedMuts = []
    for k1 in range(rec1.numClades):
        for k2 in range(rec2.numClades):
            sharedMuts += identity_to_muts[k1][k2]

    otherMuts = identity_to_muts[-1][-1]

    num_clades = len(cladeMuts)

    yticks = np.arange(0, num_clades + 1) + 0.5
    yticklabels_complete = [f'Clade {k}' for k in range(1, num_clades + 1)] + ['Other']
    xticks = np.arange(0, num_clades_LTEE + 1) + 0.5
    # xticklabels = ['Major clade', 'Minor clade', 'Ancestral, Basal & Extinct']
    xticklabels = ['Major', 'Minor', 'Other']
    identity_counts = np.zeros((num_clades + 1, num_clades_LTEE + 1))
    for k, muts in enumerate(cladeMuts + [otherMuts + sharedMuts]):
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


def plot_identity_three_periods(pop, rec1, rec2, rec3, data, square=True, cmap=CMAP, reduce_vmax=False, plot_xticks=True, plot_yticks=True, xticklabels_rotation=0, yticklabels_rotation=0, remove_clade_muts_before_period_start=True, verbose=False):

    rec1, rec2, rec3 = copy.deepcopy(rec1), copy.deepcopy(rec2), copy.deepcopy(rec3)
    ax = plt.gca()
    traj = data[pop]['traj']
    times = LH.TIMES_INTPL

    pop_to_vmax = {
        'm1': 1000, 'm2': 1000, 'm4': 600, 'm5': 90, 'm6': 60, 'p1': 100, 'p3': 500, 'p5': 60
    }

    num_clades_LTEE = 2
    clade_LTEE_to_identity_index = get_mapping_clade_LTEE_to_identity_index()

    # For latter periods, remove cladeMuts if it emerge before its period
    def getEmergenceTimes(traj, times, thPresent=0.5):
        """
        Computes emergence time of all mutations.
        """
        T, L = traj.shape
        emergenceTimes = np.full(L, -1, dtype=int)
        for l in range(L):
            for t in range(T):
                if traj[t, l] > thPresent:
                    emergenceTimes[l] = times[t]
                    break
        return emergenceTimes

    def removeCladeMutsBeforePeriodStart(rec, period_no, verbose=False):
        for i, clade in enumerate(rec.cladeMuts):
            mut_index_to_keep = []
            for j, l in enumerate(clade):
                if emergenceTimes[l] < rec.times[0]:
                    rec.otherMuts.append(l)
                    if verbose:
                        print(f'Clade {i + 1} in reconstruction of period {period_no} contains mutation {l} which emerges at {emergenceTimes[l]}, earlier than the period starting time {rec.times[0]}; Now excluded.')
                else:
                    mut_index_to_keep.append(j)
            rec.cladeMuts[i] = [clade[j] for j in mut_index_to_keep]

    def countCladeMuts(rec, period_no):
        print(f"Period {period_no}", end='\t')
        for i, clade in enumerate(rec.cladeMuts):
            print(f"{len(clade)}", end='\t')
        print()


    def removeCladeWithFewMuts(rec, period_no, thNum=30):
        clade_index_to_keep = []
        for i, clade in enumerate(rec.cladeMuts):
            if len(clade) < thNum:
                for l in clade:
                    rec.otherMuts.append(l)
            else:
                clade_index_to_keep.append(i)
        rec.cladeMuts = [rec.cladeMuts[i] for i in clade_index_to_keep]
        rec.numClades = len(rec.cladeMuts)

    if remove_clade_muts_before_period_start:
        if verbose:
            print(f"Before removing mutations that emerge too early...")
            countCladeMuts(rec1, 1)
            countCladeMuts(rec2, 2)
            if rec3 is not None:
                countCladeMuts(rec3, 3)
            # AP.plotTotalCladeFreq(rec3.cladeFreqWithAncestor, traj=traj, times=times, plotClade=False, cladeMuts=rec3.cladeMuts, otherMuts=rec3.otherMuts)
        emergenceTimes = getEmergenceTimes(traj, times)
        removeCladeMutsBeforePeriodStart(rec2, 2, verbose=verbose)
        if rec3 is not None:
            removeCladeMutsBeforePeriodStart(rec3, 3, verbose=verbose)

        if verbose:
            print(f"After removing mutations that emerge too early...")
            countCladeMuts(rec1, 1)
            countCladeMuts(rec2, 2)
            if rec3 is not None:
                countCladeMuts(rec3, 3)

        removeCladeWithFewMuts(rec2, 2)
        removeCladeWithFewMuts(rec3, 3)

        if verbose:
            print(f"After removing clades with too few mutations...")
            countCladeMuts(rec1, 1)
            countCladeMuts(rec2, 2)
            if rec3 is not None:
                countCladeMuts(rec3, 3)
            # AP.plotTotalCladeFreq(rec3.cladeFreqWithAncestor, traj=traj, times=times, plotClade=False, cladeMuts=rec3.cladeMuts, otherMuts=rec3.otherMuts)


    identity_to_muts = compare_cladeMuts(rec1, rec2)
    if rec3 is not None:
        identity_to_muts_2 = compare_cladeMuts(rec2, rec3)

    cladeMuts = []
    for k in range(rec1.numClades):
        cladeMuts.append(identity_to_muts[k][-1])  # -1 is other
    for k in range(rec2.numClades):
        cladeMuts.append(identity_to_muts[-1][k])
    if rec3 is not None:
        for k in range(rec3.numClades):
            cladeMuts.append(identity_to_muts_2[-1][k])

    sharedMuts = []
    for k1 in range(rec1.numClades):
        for k2 in range(rec2.numClades):
            sharedMuts += identity_to_muts[k1][k2]

    otherMuts = identity_to_muts[-1][-1]
    if rec3 is not None:
        otherMuts = [_ for _ in otherMuts if _ in identity_to_muts_2[-1][-1]]

    num_clades = len(cladeMuts)

    yticks = np.arange(0, num_clades + 1) + 0.5
    yticklabels_complete = [f'Clade {k}' for k in range(1, num_clades + 1)] + ['Other']
    xticks = np.arange(0, num_clades_LTEE + 1) + 0.5
    # xticklabels = ['Major clade', 'Minor clade', 'Ancestral, Basal & Extinct']
    xticklabels = ['Major', 'Minor', 'Other']
    identity_counts = np.zeros((num_clades + 1, num_clades_LTEE + 1))
    for k, muts in enumerate(cladeMuts + [otherMuts + sharedMuts]):
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


def plot_figure_reconstruction_example_LTEE_two_periods_alternative(pop, rec1, rec2, data, directory=LH.CLUSTERIZATION_OUTPUT_DIR, flattened=False, alpha=0.15, alpha_shared_muts=0.3, plotIntpl=True, ylim=(-0.03, 1.03), linewidth=SIZELINE, linewidth_shared_muts=SIZELINE, bbox_to_anchor_legend=(-0.005, 0.5), annotation_line_size=SIZEANNOTATION, legend_frameon=False, plot_legend_out_on_the_right=True, save_file=None, figsize=None):

    fig, axes = get_axes_for_reconstruction_example_LTEE(figsize=figsize, small_cov_box=True)
    xlabel, ylabel = 'Generation', 'Frequency'
    xticks = range(0, 60001, 10000)
    tStart, tEnd = 0, 60500
    if pop in EXCHANGE_FIRST_TWO_CLADES_LTEE and EXCHANGE_FIRST_TWO_CLADES_LTEE[pop]:
        clade_colors = {
            0: LTEE_EXTINCT_COLOR,  # Grey
            1: LTEE_MINOR_FIXED_COLOR,  # Blue
            2: LTEE_MAJOR_FIXED_COLOR,  # Orange
            3: '#64af31',  # Green
            4: '#cd7af5',  # Purple
            5: 'red',
        }
    else:
        clade_colors = {
            0: LTEE_EXTINCT_COLOR,
            1: LTEE_MAJOR_FIXED_COLOR,
            2: LTEE_MINOR_FIXED_COLOR,
            3: '#64af31',
            4: '#cd7af5',
            5: 'red',
        }

    plt.sca(axes[0])
    ax = plt.gca()
    traj = data[pop]['traj']
    times = LH.TIMES_INTPL
    identity_to_muts = compare_cladeMuts(rec1, rec2)

    cladeMuts = []
    for k in range(rec1.numClades):
        cladeMuts.append(identity_to_muts[k][-1])
    for k in range(rec2.numClades):
        cladeMuts.append(identity_to_muts[-1][k])

    sharedMuts = []
    for k1 in range(rec1.numClades):
        for k2 in range(rec2.numClades):
            sharedMuts += identity_to_muts[k1][k2]

    otherMuts = identity_to_muts[-1][-1]

    if plot_legend_out_on_the_right:
        if len(cladeMuts) >= 4:
            bbox_to_anchor_legend = (1, 0.45)
        else:
            bbox_to_anchor_legend = (1, 0.5)
        xlim = (tStart - 500, tEnd + 500)
        sublabel_x, sublabel_y = -0.112, 0.957
    else:
        xlim = (tStart - 2500, tEnd + 500)
        sublabel_x, sublabel_y = -0.05, 1

    for l in otherMuts:
        if traj[-1, l] <= rec1.thExtinct:
            plt.plot(times, traj[:, l], color=clade_colors[0], linewidth=0.5 * linewidth, alpha=alpha)
        else:
            plt.plot(times, traj[:, l], color=LTEE_ANCESTOR_FIXED_COLOR, linewidth=0.5 * linewidth, alpha=alpha)
    for c in range(1, len(cladeMuts) + 1):
        for l in cladeMuts[c - 1]:
            plt.plot(times, traj[:, l], color=clade_colors[c], linewidth=0.5 * linewidth, alpha=alpha)
    for l in sharedMuts:
        plt.plot(times, traj[:, l], color=LTEE_INFERRED_SHARED_COLOR, linewidth=linewidth_shared_muts, alpha=alpha_shared_muts, zorder=10)

    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.xticks(ticks=xticks, labels=[])

    labels = [f'Clade {c+1}' for c in range(len(cladeMuts))]
    handles = [matplotlib.lines.Line2D([], [], color=LTEE_ANCESTOR_FIXED_COLOR, label='Ancestor')] + [matplotlib.lines.Line2D([], [], color=LTEE_MAJOR_FIXED_COLOR, label=labels[0]),
     matplotlib.lines.Line2D([], [], color=LTEE_MINOR_FIXED_COLOR, label=labels[1])] + [matplotlib.lines.Line2D([], [], color=clade_colors[c + 1], label=labels[c]) for c in range(2, len(cladeMuts))] + [matplotlib.lines.Line2D([], [], color=LTEE_INFERRED_SHARED_COLOR, label='Shared acr-\noss clades')] + [matplotlib.lines.Line2D([], [], color=LTEE_EXTINCT_COLOR, label='Extinct')]
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
    plot_identity_comparison(rec1, rec2, identity_to_muts=identity_to_muts)
    plt.text(x=-0.53, y=1.07, s=SUBLABELS[2], transform=ax.transAxes, **DEF_SUBLABELPROPS)

    # if legend_frameon:
    #     plt.subplots_adjust(0.06, 0.15, 0.83, 0.96)
    # else:
    #     plt.subplots_adjust(0.06, 0.15, 0.99, 0.96)
    plt.show()
    if save_file:
        fig.savefig(save_file, facecolor=fig.get_facecolor(), **DEF_FIGPROPS)


def compare_cladeMuts(rec1, rec2):
    identity_counts = np.zeros((rec1.numClades + 1, rec2.numClades + 1))
    identity_to_muts = [[[] for k2 in range(rec2.numClades + 1)] for k1 in range(rec1.numClades + 1)]
    rec2_mut_to_clade = {}
    for k, muts in enumerate(rec2.cladeMuts + [rec2.otherMuts]):
        for mut in muts:
            rec2_mut_to_clade[mut] = k
    for k1, muts in enumerate(rec1.cladeMuts + [rec1.otherMuts]):
        for mut in muts:
            k2 = rec2_mut_to_clade[mut]
            identity_counts[k1, k2] += 1
            identity_to_muts[k1][k2].append(mut)
    return identity_to_muts


def plot_identity_comparison(rec1, rec2, identity_to_muts=None, square=True, 
                      xticklabels_rotation=0, yticklabels_rotation=0):

    if identity_to_muts is None:
        identity_to_muts = compare_cladeMuts(rec1, rec2)

    identity_counts = np.zeros((rec1.numClades + 1, rec2.numClades + 1))
    for k1, muts1 in enumerate(rec1.cladeMuts + [rec1.otherMuts]):
        for k2, muts2 in enumerate(rec2.cladeMuts + [rec2.otherMuts]):
            identity_counts[k1, k2] = len(identity_to_muts[k1][k2])

    ax = plt.gca()
    sns.heatmap(identity_counts, center=0, vmin=0, vmax=None, 
                cmap=CMAP, square=square, cbar=False, annot=True, fmt='.0f', 
                annot_kws={"size": SIZEANNOTATION_HEATMAP})
    yticks = np.arange(0, rec1.numClades + 1) + 0.5
    yticklabels = [f'Clade {k}' for k in range(1, rec1.numClades + 1)] + ['Other']
    xticks = np.arange(0, rec2.numClades + 1) + 0.5
    xticklabels = [f'Clade {k}' for k in range(rec1.numClades + 1, rec1.numClades + rec2.numClades + 1)] + ['Other']
    plt.xticks(ticks=xticks, labels=xticklabels, fontsize=SIZELABEL, rotation=xticklabels_rotation)
    plt.yticks(ticks=yticks, labels=yticklabels, fontsize=SIZELABEL, rotation=yticklabels_rotation)
    ax.tick_params(**DEF_TICKPROPS_HEATMAP)
    plt.setp(ax.spines.values(), **DEF_AXPROPS)
    plt.xlabel('Clades inferred in second period', fontsize=SIZELABEL)
    plt.ylabel('Clades inferred in first period', fontsize=SIZELABEL)


def plot_dxdx_heatmap(dxdx, groups, as_subplot=False, ylabel_x=-0.21, alpha=0.5, grid_line_offset=0, cbar_shrink=1, cbar_label="Value in " + r"$D$" + " matrix", xtick_distance=40, figsize=(4, 3), rasterized=True, ylabel_prefix="Group", plot_ylabel=False, plot_top_axis=False, save_file=None):

    L = len(dxdx)
    if L > 8000:
        xtick_distance = 2000
    elif L > 6000:
        xtick_distance = 1000
    elif L > 3000:
        xtick_distance = 800
    elif L > 1000:
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
    cbar.set_label(cbar_label, fontsize=SIZELABEL)
    xticklabels = np.arange(0, L + xtick_distance // 2, xtick_distance)
    xticks = [l for l in xticklabels]
    ticks, ylabels, group_sizes = get_ticks_and_labels_for_clusterization(groups, name=ylabel_prefix, note_size=True)
    # plot_ticks_and_labels_for_clusterization(ticks, ylabels, group_sizes, ylabel_x=ylabel_x)
    ax.hlines([_ + grid_line_offset for _ in ticks], *ax.get_xlim(), color=GREY_COLOR_HEX, alpha=alpha, linewidth=SIZELINE * 1.2)
    ax.vlines([_ + grid_line_offset for _ in ticks], *ax.get_ylim(), color=GREY_COLOR_HEX, alpha=alpha, linewidth=SIZELINE * 1.2)
    if plot_ylabel:
        yticks, yticklabels = ticks, ylabels
    else:
        yticks, yticklabels = [], []
    set_ticks_labels_axes(yticks=yticks, yticklabels=yticklabels, xticks=xticks, xticklabels=xticklabels)
    plt.xlabel('Locus index', fontsize=SIZELABEL)

    if plot_top_axis:
        ax2 = ax.twiny()  # ax2 is responsible for "top" axis
        ticks = [0] + ticks
        print(ticks)
        ticks = [(ticks[i] + ticks[i + 1]) / 2 for i in range(len(ticks) - 1)]
        print(ticks)
        ylabels = [_.replace(" (", "\n(") for _ in ylabels]
        if L > 10000:  # pop p6
            ticks[1] -= 700
            ticks[2] -= 500
        else:
            min_tick_dist = L * 0.175
            for i in range(len(ticks) - 2, 0, -1):
                if ticks[i + 1] - ticks[i] < min_tick_dist:
                    ticks[i] = ticks[i + 1] - min_tick_dist
        print(ticks)
        plt.sca(ax2)
        plt.xticks(ticks=ticks, labels=ylabels, fontsize=SIZELABEL, rotation=0)
        plt.xlim(0, L)
        ax2.tick_params(**DEF_TICKPROPS_COV_TOP)
        plt.setp(ax.spines.values(), **DEF_AXPROPS)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.spines['bottom'].set_visible(False)
        ax2.spines['left'].set_visible(False)
        # ax2.axis["right"].major_ticklabels.set_visible(False)
        # ax2.axis["left"].major_ticklabels.set_visible(False)
        # ax2.axis["bottom"].major_ticklabels.set_visible(False)
        # ax2.axis["top"].major_ticklabels.set_visible(True)

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


def plot_figure_identities_LTEE(populations, reconstructions, data, custom_arrangement=True, mark_nonclonal_with_rectangle=False, plot_ylabel_for_all_second_row=True, mark_nonclonal_with_horizontal_text=True, mark_nonclonal_with_vertical_text=False, nRow=2, nCol=4, wspace=0.3, hspace=0.3, square=False, reduce_vmax=False, ticklabels_rotation=0, title_prefix='', plot_single_column=False, plot_sublabel=True, save_file=None, rec1_p3=None, rec2_p3=None, rec1_m3=None, rec2_m3=None, rec3_m3=None):
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
            plt.title(f"{title_prefix}{pop}", fontsize=SIZELABEL)
            if i == 0:
                if plot_sublabel:
                    plt.text(x=sublabel_x, y=sublabel_y, s=SUBLABELS[0], transform=ax.transAxes, **DEF_SUBLABELPROPS)

        for i, pop in enumerate(populations_3_clades):
            ax = plt.subplot(srl_gs[0, i])
            if pop == 'p3':
                plot_identity_two_periods(pop, rec1_p3, rec2_p3, data, square=square, reduce_vmax=reduce_vmax, plot_xticks=True, plot_yticks=True if plot_ylabel_for_all_second_row else (i == 0), xticklabels_rotation=ticklabels_rotation, yticklabels_rotation=ticklabels_rotation)
            else:
                plot_identity_helper(pop, reconstructions[pop], data, square=square, reduce_vmax=reduce_vmax, plot_xticks=True, plot_yticks=True if plot_ylabel_for_all_second_row else (i == 0), xticklabels_rotation=ticklabels_rotation, yticklabels_rotation=ticklabels_rotation)
            plt.title(f"{title_prefix}{pop}", fontsize=SIZELABEL)
            if i == 0:
                if plot_sublabel:
                    plt.text(x=sublabel_x, y=sublabel_y, s=SUBLABELS[1], transform=ax.transAxes, **DEF_SUBLABELPROPS)

        pop = special_pop
        ax = plt.subplot(srm_gs[0, 0])
        plot_identity_helper(pop, reconstructions[pop], data, square=square, cmap=CMAP_NONCLONAL, reduce_vmax=reduce_vmax, plot_xticks=True, plot_yticks=True, xticklabels_rotation=ticklabels_rotation, yticklabels_rotation=ticklabels_rotation)
        plt.title(f"{title_prefix}{pop}", fontsize=SIZELABEL)
        if plot_sublabel:
            plt.text(x=sublabel_x, y=sublabel_y, s=SUBLABELS[2], transform=ax.transAxes, **DEF_SUBLABELPROPS)

        for i, pop in enumerate(populations_4_clades):
            ax = plt.subplot(srr_gs[0, i])
            if pop == 'm3':
                plot_identity_three_periods(pop, rec1_m3, rec2_m3, rec3_m3, data, square=square, cmap=CMAP_NONCLONAL, reduce_vmax=reduce_vmax, plot_xticks=True, plot_yticks=True if plot_ylabel_for_all_second_row else (i == 0), xticklabels_rotation=ticklabels_rotation, yticklabels_rotation=ticklabels_rotation)
            else:
                plot_identity_helper(pop, reconstructions[pop], data, square=square, cmap=CMAP_NONCLONAL, reduce_vmax=reduce_vmax, plot_xticks=True, plot_yticks=True if plot_ylabel_for_all_second_row else (i == 0), xticklabels_rotation=ticklabels_rotation, yticklabels_rotation=ticklabels_rotation)
            plt.title(f"{title_prefix}{pop}", fontsize=SIZELABEL)

        if mark_nonclonal_with_rectangle:
            rect = matplotlib.patches.Rectangle((-9.8, -1.3), 13, 7.5, clip_on=False, linewidth=0.6, edgecolor=GREY_COLOR_HEX, facecolor='none')
            ax.add_patch(rect)

        # bbox_color = '#E5CDE9'
        bbox_color = '#C7DBD0'
        bbox_alpha = 0.8

        if mark_nonclonal_with_horizontal_text:
            t = plt.text(-2.25, -0.46, 'No clonal structure inferred previously', transform=ax.transAxes, fontsize=SIZELABEL, color='black')
            # bbox = matplotlib.patches.FancyBboxPatch((-9.55, 6.7), 12.25, 0.8, clip_on=False, linewidth=0.6, edgecolor=bbox_color, facecolor=bbox_color, alpha=bbox_alpha)
            bbox = matplotlib.patches.FancyBboxPatch((-9.55, 5.5), 12.25, 0.42, clip_on=False, linewidth=0.6, edgecolor=bbox_color, facecolor=bbox_color, alpha=bbox_alpha)
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
            plt.title(f"{title_prefix}{pop}", fontsize=SIZELABEL)
        if nCol == 6 and nRow == 2:
            plt.subplots_adjust(0.06, 0.1, 0.98, 0.9, wspace=wspace, hspace=hspace)
        else:
            plt.subplots_adjust(wspace=wspace, hspace=hspace)
    plt.show()
    if save_file:
        fig.savefig(save_file, facecolor=fig.get_facecolor(), **DEF_FIGPROPS)


def get_axes_for_performance_on_real_data(plot_single_column=True, perf_box_squeeze=0.1):
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
        cov_right = 0.98
        cov_bottom = box_bottom
        cov_top = box_top

        cbar_width = 0.1
        cov_width = ratio * (cov_top - cov_bottom) + cbar_width
        box_middle = cov_right - divider_width - cov_width
        cov_left = box_middle + divider_width

        perf_left = box_left + perf_box_squeeze
        perf_right = box_middle - perf_box_squeeze
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


def plot_cov_heatmap(reconstruction, ylabel_x=-0.21, alpha=0.5, grid_line_offset=0, cbar_shrink=1, xtick_distance=40, cbar_label="Value in recovered integrated covariance matrix"):
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
    cbar.set_label(cbar_label, fontsize=SIZELABEL)

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

    xlim, ylim = [-0.6, len(methods)], [0, 1]
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
    for method, inferred_fitness, x, y in zip(methods, inferred_fitness_list, xs, ys):
        if np.isnan(y):
            na_x, na_y = x - 0.225, 0.2
            plt.plot([x, x], [0.01, na_y - 0.05], linewidth=SIZELINE, color=BKCOLOR)
            plt.text(na_x, na_y, 'NA', fontsize=SIZESUBLABEL)


def plot_clade_annotation(annotXs, annotYs, clade_colors, clade_labels, background_width=10.8, background_width_for_ancestor=12.8, background_height=0.09, background_color='#f2f2f2'):
    ax = plt.gca()
    for x, y, color, label in zip(annotXs, annotYs, clade_colors, clade_labels):
        plt.text(x, y, label, fontsize=SIZEANNOTATION, color=color, zorder=10)
        patch = mpatches.FancyBboxPatch((x, y + 0.01), background_width_for_ancestor if label == 'Ancestor' else background_width, background_height, color=background_color, zorder=5, boxstyle=mpatches.BoxStyle("Round", pad=0.02, rounding_size=0), clip_on=False)
        ax.add_patch(patch)


def plot_figure_performance_on_data_PALTE(traj, reconstruction, measured_fitness_by_pop, inferred_fitness_by_pop_list, times=PALTEanalysis.TIMES, plot_single_column=False, use_pearsonr=False, plot_muller_plot=True, background_width=65, background_width_for_ancestor=75, background_height=0.09, background_color='#f2f2f2', methods=PALTEanalysis.METHODS, alpha=0.5, ylim=(-0.03, 1.03), grid_line_offset=-0.5, save_file=None):

    perf_box_squeeze = 0.1 if len(methods) == 4 else 0.07
    fig, axes = get_axes_for_performance_on_real_data(plot_single_column=plot_single_column, perf_box_squeeze=perf_box_squeeze)

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
        annotXs = [20, 80, 90, 100, 300, 350, 500]
        annotYs = [0.8, 0.15, 0.45, 0.65, 0.1, 0.65, 0.85]
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
    plt.text(x=1.06, y=sublabel_y, s=SUBLABELS[3], transform=axes[0].transAxes, **DEF_SUBLABELPROPS)

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


def plot_figure_performance_on_data_tobramycin(traj, reconstruction, measured_MIC_list, median_inferred_fitness_lists, times=tobramycin_analysis.TIMES_PA, plot_single_column=False, use_pearsonr=False, plot_muller_plot=True, background_width=9, background_height=0.09, background_color='#f2f2f2', methods=tobramycin_analysis.METHODS, alpha=0.5, xticks=np.arange(0, 90, 10), ylim=(-0.03, 1.03), grid_line_offset=0, save_file=None):

    perf_box_squeeze = 0.1 if len(methods) == 4 else 0.07
    fig, axes = get_axes_for_performance_on_real_data(plot_single_column=plot_single_column, perf_box_squeeze=perf_box_squeeze)

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
        # annotXs = [2, 65, 15, 30, 30]
        # annotYs = [0.8, 0.25, 0.45, 0.7, 0.7]
        annotXs = [2, 15, 50, 70, 30]
        annotYs = [0.8, 0.4, 0.55, 0.8, 0.7]
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



def plot_figure_runtime_LTEE(populations, num_alleles_sorted, run_time, methods=METHODS_COMPARE_RUN_TIME, colors=METHODS_COMPARE_RUN_TIME_COLORS, markers=MARKERS_METHODS_COMPARE_RUN_TIME, markersize=1.5*SIZEDOT, fit_linear_regression=True, plot_linear_regression=True, plot_fake_time=True, max_run_time=336, save_file=None):

    ratio = 0.5
    w = DOUBLE_COLUMN
    h = ratio * w
    fig = plt.figure(figsize=(w, h))
    ax = plt.gca()

    xs_all = np.array(num_alleles_sorted)
    xs_mutators = xs_all[6:]
    xlabel = 'Number of alleles'
    ylabel = 'Run time (hour)'
    yticks = [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 336]
    yticklabels = [0.001, 0.01, 0.1, 1, 10, 100, '>336']
    xticks = [100, 500, 1000, 5000, 10000]
    xticklabels = xticks
    ylim = (1e-3, 370)
    xlim = (1e2, 2e4)

    for i, (method, color, marker) in enumerate(zip(methods, colors, markers)):
        indices = [_ for _, time in enumerate(run_time[method]) if time != 'nan']
        ys = np.array(run_time[method])[indices].astype(float)
        xs = xs_all[indices].astype(float)
        plt.scatter(xs, ys, label=method, color=color, marker=marker, s=markersize)
        if fit_linear_regression:
            reg = stats.linregress(np.log10(xs), np.log10(ys))
            predictor = lambda x: x * reg.slope + reg.intercept
            if plot_linear_regression and method == OUR_METHOD_NAME:
                xs = [1.5e2, 1.5e4]
                ys = [10 ** predictor(np.log10(x)) for x in xs]
                print(f'method={method}, slope=%.3f, intercept=%.3f, rvalue=%.3f, pvalue=%.3f' % (reg.slope, reg.intercept, reg.rvalue, reg.pvalue))
                plt.plot(xs, ys, color=color, linewidth=SIZELINE * 1.5, linestyle='dashed', label=r"$y=10^{-5.55}x^{1.62}$" + ", Pearson's " + r'$r=$' + '%.3f'%reg.rvalue)

        if plot_fake_time:
            indices = [_ for _, time in enumerate(run_time[method]) if time == 'nan']
            if indices:
                xs = xs_all[indices].astype(float)
                ys = np.full((len(xs)), max_run_time) 
                # plt.scatter(xs, ys, color=color, marker=marker, s=markersize, facecolors='none')
                plt.scatter(xs, ys, color=color, marker=marker, s=markersize)

    # plt.plot([1.5e2, 1e4], [max_run_time] * 2, color=GREY_COLOR_HEX, linewidth=SIZELINE * 1.5, linestyle='dashed')
    # plt.text(1.1e4, 280, '2 weeks', fontsize=SIZELABEL)

    # plt.plot(xs_mutators, [24] * len(xs_mutators), color=GREY_COLOR_HEX, linewidth=SIZELINE, linestyle='dashed')
    # plt.text(xs_all[6] - 1200, 20, '1 day', fontsize=SIZELABEL)

    plt.xscale('log')
    plt.yscale('log')

    set_ticks_labels_axes(xlim=xlim, ylim=ylim, ylabel=ylabel, yticks=yticks, yticklabels=yticklabels, xlabel=xlabel, xticks=xticks, xticklabels=xticklabels)
    # def_tickprops = {key: value for key, value in DEF_TICKPROPS.items()}
    # def_tickprops['top'] = 'none'
    # plt.gca().tick_params(**def_tickprops)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.legend(loc='lower right', fontsize=SIZELEGEND)

    plt.show()
    if save_file:
        fig.savefig(save_file, facecolor=fig.get_facecolor(), **DEF_FIGPROPS)


def save_subfigure_for_merge_periods(reconstruction, clade_colors=None, save_file_prefix=None, postfix='.pdf', def_saveprops={}, linewidth=2*SIZELINE, cladeFreqLinestyle='solid', alleleFreqAlpha=0.4, ylim=(0.001, 0.999)):


    if clade_colors is None:
        clade_colors = sns.husl_palette(reconstruction.numClades + 2)

    traj_width = 2
    traj_height = traj_width * 0.35
    heatmap_height = traj_height

    # Periods
    clade_index_offset = 0
    ancestor_clade_index = -1
    for i, period in enumerate(reconstruction.periods):
        period_traj_width = traj_width * (period.times[-1] - period.times[0]) / (reconstruction.times[-1] - reconstruction.times[0])
        fig = AP.plotCladeAndMutFreq_overview(period, colors=clade_colors, ylim=ylim, clade_index_offset=clade_index_offset, ancestor_clade_index=ancestor_clade_index, fontsize=SIZELABEL, cladeFreqLinestyle=cladeFreqLinestyle, alleleFreqAlpha=alleleFreqAlpha, linewidth=linewidth, plotLegend=False, legendsize=SIZELEGEND, title='', plotFigure=True, figsize=(period_traj_width, traj_height), returnFig=True, plotShow=False)
        ax = plt.gca()
        ancestor_clade_index = clade_index_offset + np.argmax(period.cladeFreq[-1])
        clade_index_offset += period.numClades
        set_ticks_labels_axes(xlim=None, ylim=None, xticks=[], xticklabels=[], ylabel=None, yticks=[], yticklabels=[])
        plt.subplots_adjust(0, 0, 1, 1)
        plt.axis('off')
        if save_file_prefix is not None:
            fig.savefig(f'{save_file_prefix}-traj-{i}' + postfix, facecolor=fig.get_facecolor(), **DEF_FIGPROPS, **def_saveprops)

    fig = AP.plotCladeAndMutFreq_overview(reconstruction, colors=clade_colors, ylim=ylim, clade_index_offset=0, ancestor_clade_index=-1, fontsize=SIZELABEL, cladeFreqLinestyle=cladeFreqLinestyle, alleleFreqAlpha=alleleFreqAlpha, linewidth=linewidth, plotLegend=False, legendsize=SIZELEGEND, title='', plotFigure=True, figsize=(traj_width, traj_height), returnFig=True, plotShow=False)
    ax = plt.gca()
    set_ticks_labels_axes(xlim=None, ylim=None, xticks=[], xticklabels=[], ylabel=None, yticks=[], yticklabels=[])
    plt.subplots_adjust(0, 0, 1, 1)
    plt.axis('off')
    if save_file_prefix is not None:
        fig.savefig(f'{save_file_prefix}-traj-{len(reconstruction.periods)}' + postfix, facecolor=fig.get_facecolor(), **DEF_FIGPROPS, **def_saveprops)

    # Refined periods
    refined_period_colors_list = [
        clade_colors,
        [clade_colors[0], clade_colors[2], clade_colors[1], clade_colors[3], clade_colors[4]],
    ]
    for i, period in enumerate(reconstruction.refinedPeriods):
        refined_period_colors = refined_period_colors_list[i]
        period_traj_width = traj_width * (period.times[-1] - period.times[0]) / (reconstruction.times[-1] - reconstruction.times[0])
        print(period.cladeFreq.shape, clade_index_offset, len(clade_colors)) 
        fig = AP.plotCladeAndMutFreq_overview(period, colors=refined_period_colors, ylim=ylim, fontsize=SIZELABEL, ancestor_clade_index=-1 if i == 0 else 0, clade_index_offset=0 if i == 0 else 1, cladeFreqLinestyle=cladeFreqLinestyle, alleleFreqAlpha=alleleFreqAlpha, linewidth=linewidth, plotLegend=False, legendsize=SIZELEGEND, title='', plotFigure=True, figsize=(period_traj_width, traj_height), returnFig=True, plotShow=False)
        ax = plt.gca()
        ancestor_clade_index = clade_index_offset + np.argmax(period.cladeFreq[-1])
        clade_index_offset += period.numClades
        set_ticks_labels_axes(xlim=None, ylim=None, xticks=[], xticklabels=[], ylabel=None, yticks=[], yticklabels=[])
        plt.subplots_adjust(0, 0, 1, 1)
        plt.axis('off')
        if save_file_prefix is not None:
            fig.savefig(f'{save_file_prefix}-traj-{i + 1+ len(reconstruction.periods)}' + postfix, facecolor=fig.get_facecolor(), **DEF_FIGPROPS, **def_saveprops)

    plt.close('all')


############################################
#
# Placeholder
#
############################################

