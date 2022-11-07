"""
This Python file contains code for parsing data from Scribner, M. R., Santos-Lopez, A., Marshall, C. W., Deitrick, C. & Cooper, V. S. Parallel Evolution of Tobramycin Resistance across Species and Environments. MBio 11, (2020) in:
    Title of the paper
    Yunxiao Li, and
    John P. Barton (john.barton@ucr.edu)

"""

#############  PACKAGES  #############

import sys

import numpy as np
import math

from copy import deepcopy

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec

import mplot as mp

import seaborn as sns

import pandas as pd

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

LOLIPOP_INPUT_DIR = './data/lolipop/input'
LOLIPOP_OUTPUT_DIR = './data/lolipop/output'
LOLIPOP_PARSED_OUTPUT_DIR = './data/lolipop/parsed_output'

TOBRAMYCIN_DIR = './data/tobramycin_analysis/paeruginosa_WGS/allele_and_muller_plots/'

NUM_REPLICATES = 3
MEDIA = ['Biofilm', 'Planktonic']
PA_ALLELE_FILES = {
    medium: [f'{TOBRAMYCIN_DIR}/{medium}_{rpl + 1}_mullerinput.csv'
                   for rpl in range(NUM_REPLICATES)] for medium in MEDIA
}

TIMES_AB = (np.array([0, 1, 3, 4, 6, 7, 9, 10, 12]) * 6.64).astype(int)
TIMES_PA = (np.array([0, 3, 4, 6, 7, 9, 10, 12]) * 6.64).astype(int)

METHODS = ['recovered', 'Lolipop', 'est_cov', 'SL']

MU = 1e-6
REG = 4  # Adjustable through set_reg(reg)

PA_MUTATIONS_TO_MIC = {
    'Ancestor': 0.5, #
    'fusA1 Q129H': 1.5,
    'fusA1 N592I': 2.0, #
    'fusA1 F596L': 2.0, #
    'fusA1 Q678L': 2.0, #
    'fusA1 R680C': 1.0, #
    'ptsP Δ42bp (deletion of nt 1846–1887/2280)': 1.0, #
    'ptsP Δ14bp (deletion of nt 1296–1309/2280)': 1.5,
    'fusA1 Q678L ptsP R301C': 4.0,
    'fusA1 T456A ptsP V661E': 2.0,
    'fusA1 Q563L ptsP E335*': 4.0,
    'fusA1 R680C ptsP E335*': 4.0, #
    'fusA1 T671A ptsP Δ1bp (deletion of nt 1122/2280)': 4.0, #
    'fusA1 N592I orfN Δ1bp (deletion of nt 148/1017)': 2.0,
}

PA_GENES_PRESENT = ['fusA1 N592I', 'fusA1 F596L', 'fusA1 Q678L', 'fusA1 R680C', 'fusA1 T671A',
                     'ptsP V661E', 'ptsP E335*', 'ptsP coding', 'ptsP coding.2', ]

PA_MUTATIONS_TO_GENES = {
    'Ancestor': [], #
    'fusA1 N592I': ['fusA1 N592I'], #
    'fusA1 F596L': ['fusA1 F596L'], #
    'fusA1 Q678L': ['fusA1 Q678L'], #
    'fusA1 R680C': ['fusA1 R680C'], #
    'ptsP Δ42bp (deletion of nt 1846–1887/2280)': ['ptsP coding.2'], #
    'fusA1 R680C ptsP E335*': ['fusA1 R680C', 'ptsP E335*'], #
    'fusA1 T671A ptsP Δ1bp (deletion of nt 1122/2280)': ['fusA1 T671A', 'ptsP coding'], #
}

PA_MUTATIONS_TO_MEASURED_MIC = {
    'Ancestor': 0.5, #
    'fusA1 N592I': 2.0, #
    'fusA1 F596L': 2.0, #
    'fusA1 Q678L': 2.0, #
    'fusA1 R680C': 1.0, #
    'ptsP Δ42bp (deletion of nt 1846–1887/2280)': 1.0, #
    'fusA1 R680C ptsP E335*': 4.0, #
    'fusA1 T671A ptsP Δ1bp (deletion of nt 1122/2280)': 4.0, #
}

PA_MUTATIONS_TO_MEASURED_MIC_RANGE = {
    'Ancestor': (0.25, 1), #
    'fusA1 N592I': (2.0, 2.0), #
    'fusA1 F596L': (1.0, 2.0), #
    'fusA1 Q678L': (1.0, 2.0), #
    'fusA1 R680C': (1.0, 2.0), #
    'ptsP Δ42bp (deletion of nt 1846–1887/2280)': (1.0, 2.0), #
    'fusA1 R680C ptsP E335*': (2.0, 4.0), #
    'fusA1 T671A ptsP Δ1bp (deletion of nt 1122/2280)': (2.0, 8.0), #
}

############################################
#
# Helper functions
#
############################################

def init_a_dic(placeholder=False):
    if placeholder:
        return {medium: [None for _ in range(NUM_REPLICATES)] for medium in MEDIA}
    return {medium: [] for medium in MEDIA}


def set_reg(reg):
    global REG
    REG = reg


############################################
#
# Parsing data & results
#
############################################

def parse_traj(df):
    T, L = len(df.columns) - 3, len(df)
    # print(f'{L} mutations, {T} timepoints')
    traj = np.zeros((T, L), dtype=float)
    genes = ['' for l in range(L)]
    times = np.array(sorted([int(col) for col in df.columns if col.isdigit()]))
    for i, row in df.iterrows():
        l = int(row['Trajectory']) - 1
        genes[l] = row['Gene']
        for t, time in enumerate(times):
            traj[t, l] = float(row[str(time)]) / 100
    # return traj, times, genes
    return traj, TIMES_PA, genes


df_pa_allele = init_a_dic()
traj_pa = init_a_dic()
for medium in MEDIA:
    for rpl in range(NUM_REPLICATES):
        df_pa_allele[medium].append(pd.read_csv(PA_ALLELE_FILES[medium][rpl]))
        traj_pa[medium].append(parse_traj(df_pa_allele[medium][rpl]))


def parse_muller_files(prefix='pa', output_dir=LOLIPOP_OUTPUT_DIR):
    files = {medium: [None for _ in range(NUM_REPLICATES)] for medium in MEDIA}
    for medium in MEDIA:
        for rpl in range(NUM_REPLICATES):
            directory = f'{output_dir}/{prefix}_{medium.lower()}_{rpl+1}/graphics/lineage'
            filename = f'{medium}_{rpl+1}_mullerinput.mullerdiagram.annotated.png'
            files[medium][rpl] = f'{directory}/{filename}'
    return files


def get_reconstruction_from_traj(traj, times, mu=MU, defaultReg=REG, meanReadDepth=100, thFixed=0.98, thLogProbPerTime=10, thFreqUnconnected=1, debug=False, verbose=False, plot=False, evaluateReconstruction=True, evaluateInference=False):
    rec = RC.CladeReconstruction(traj, times=times, meanReadDepth=meanReadDepth, debug=debug, verbose=verbose, plot=plot, mu=mu)
    rec.setParamsForClusterization(weightByBothVariance=False, weightBySmallerVariance=True)
    rec.clusterMutations()
    rec.setParamsForReconstruction(thFixed=thFixed, thExtinct=0, numClades=None, percentMutsToIncludeInMajorClades=95, thLogProbPerTime=thLogProbPerTime, thFreqUnconnected=thFreqUnconnected)
    rec.checkForSeparablePeriodAndReconstruct()
    rec.setParamsForEvaluation(defaultReg=defaultReg)
    evaluation, inference = rec.evaluate(evaluateReconstruction=evaluateReconstruction, evaluateInference=evaluateInference)
    return rec, evaluation, inference


def parse_reconstructions():
    reconstructions = init_a_dic(placeholder=True)
    evaluations = init_a_dic(placeholder=True)
    inferences = init_a_dic(placeholder=True)

    for medium in MEDIA:
        for rpl in range(NUM_REPLICATES):
            traj, times = traj_pa[medium][rpl][:2]
            reconstructions[medium][rpl], evaluations[medium][rpl], inferences[medium][rpl] = get_reconstruction_from_traj(traj, times, mu=MU, evaluateReconstruction=True, evaluateInference=True)

    return reconstructions, evaluations, inferences


def parse_Lolipop_results(muForInference=MU, regularization=REG):
    intCov_lolipop, selection_lolipop, fitness_lolipop = init_a_dic(), init_a_dic(), init_a_dic()
    for medium in MEDIA:
        for rpl in range(NUM_REPLICATES):
            filename = f'{medium}_{rpl+1}_mullerinput'
            file = f'{LOLIPOP_PARSED_OUTPUT_DIR}/{filename}.npz'
            data = lolipop_helper.loadParsedOutput(file)
            intCov, selection, fitness = lolipop_helper.inferencePipeline(data, mu=muForInference, regularization=regularization)

            intCov_lolipop[medium].append(intCov)
            selection_lolipop[medium].append(selection)
            fitness_lolipop[medium].append(fitness)
    return intCov_lolipop, selection_lolipop, fitness_lolipop


def map_gene_to_selection_for_methods(inferences, selection_lolipop, methods=METHODS, genes=PA_GENES_PRESENT):
    pa_gene_to_selection_by_method = {method: {} for method in methods}

    # Lolipop
    for medium in MEDIA:
        for rpl in range(NUM_REPLICATES):
            traj, times, genes = traj_pa[medium][rpl]
            infered_selection = selection_lolipop[medium][rpl]
            for l, gene in enumerate(genes):
                if gene not in pa_gene_to_selection_by_method['Lolipop']:
                    pa_gene_to_selection_by_method['Lolipop'][gene] = [infered_selection[l]]
                else:
                    pa_gene_to_selection_by_method['Lolipop'][gene].append(infered_selection[l])

    # Other methods
    for method in methods:
        if method == 'Lolipop':
            continue
        for medium in MEDIA:
            for rpl in range(NUM_REPLICATES):
                traj, times, genes = traj_pa[medium][rpl]
                inference = inferences[medium][rpl]
                infered_selection = inference[method][2]
                for l, gene in enumerate(genes):
                    if gene not in pa_gene_to_selection_by_method[method]:
                        pa_gene_to_selection_by_method[method][gene] = [infered_selection[l]]
                    else:
                        pa_gene_to_selection_by_method[method][gene].append(infered_selection[l])

    return pa_gene_to_selection_by_method


def map_mutations_to_all_inferred_fitness(gene_to_selection, mutations_to_genes=PA_MUTATIONS_TO_GENES):
    all_inferred_fitness_by_mutations = {}
    for mutations, genes in mutations_to_genes.items():
        if genes:
            all_inferred_fitness_by_mutations[mutations] = 1
            for gene in genes:
                all_inferred_fitness_by_mutations[mutations] += np.array(gene_to_selection[gene])
        else:
            all_inferred_fitness_by_mutations[mutations] = np.array([1.0])
    return all_inferred_fitness_by_mutations


def map_mutations_to_mean_inferred_fitness(all_inferred_fitness_by_mutations):

    mean_inferred_fitness_by_mutations = {
        mutations: np.mean(value) for mutations, value in all_inferred_fitness_by_mutations.items()
    }
    return mean_inferred_fitness_by_mutations


def map_mutations_to_median_inferred_fitness(all_inferred_fitness_by_mutations):
    median_inferred_fitness_by_mutations = {
        mutations: np.median(value) for mutations, value in all_inferred_fitness_by_mutations.items()
    }
    return median_inferred_fitness_by_mutations


def parse_measured_MIC_list():
    measured_MIC_list = []
    for mutations, measured_MIC in PA_MUTATIONS_TO_MEASURED_MIC.items():
        measured_MIC_list.append(measured_MIC)
    return measured_MIC_list


def parse_measured_MIC_error_list():
    measured_MIC_error_list = []
    for mutations, measured_MIC in PA_MUTATIONS_TO_MEASURED_MIC.items():
        measured_MIC_error_list.append(np.absolute(np.array(PA_MUTATIONS_TO_MEASURED_MIC_RANGE[mutations]) - measured_MIC))
    return measured_MIC_error_list


def parse_median_inferred_fitness_list(median_inferred_fitness_by_mutations):
    median_inferred_fitness_list = []
    for mutations, measured_MIC in PA_MUTATIONS_TO_MEASURED_MIC.items():
        median_inferred_fitness_list.append(median_inferred_fitness_by_mutations[mutations])
    return median_inferred_fitness_list


def parse_mean_inferred_fitness_list(mean_inferred_fitness_by_mutations):
    mean_inferred_fitness_list = []
    for mutations, measured_MIC in PA_MUTATIONS_TO_MEASURED_MIC.items():
        mean_inferred_fitness_list.append(mean_inferred_fitness_by_mutations[mutations])
    return mean_inferred_fitness_list


def parse_median_inferred_fitness_list_of_methods(inferences, selection_lolipop, methods=METHODS):

    pa_gene_to_selection_by_method = map_gene_to_selection_for_methods(inferences, selection_lolipop, methods=methods)
    median_inferred_fitness_lists = []
    for method, gene_to_selection in pa_gene_to_selection_by_method.items():
        all_inferred_fitness_by_mutations = map_mutations_to_all_inferred_fitness(gene_to_selection)
        median_inferred_fitness_by_mutations = map_mutations_to_median_inferred_fitness(all_inferred_fitness_by_mutations)
        median_inferred_fitness_lists.append(parse_median_inferred_fitness_list(median_inferred_fitness_by_mutations))
    return median_inferred_fitness_lists


############################################
#
# Plotting functions
#
############################################

def display_muller_plots(files, NUM_REPLICATES=3, figsize=(10, 6)):
    fig = plt.figure(figsize=figsize)
    nRow = 6
    nCol = 4
    divider_rowspan, divider_colspan = 0, 0
    nRow = int(nRow + divider_rowspan)
    nCol = int(nCol + divider_colspan)
    ax2_ax1_rowspan_ratio = 3 / 2
    ax1_rowspan = int((nRow - divider_rowspan) / 3 * 2)
    ax2_rowspan = (nRow - divider_rowspan) // 3
    ax1_colspan = (nCol - divider_colspan) // 2
    ax2_colspan = (nCol - divider_colspan) // 4
    ax1 = plt.subplot2grid((nRow, nCol), (0, 0), rowspan=ax1_rowspan, colspan=ax1_colspan)
    ax2 = plt.subplot2grid((nRow, nCol), (ax1_rowspan + divider_rowspan, 0), rowspan=ax2_rowspan, colspan=ax2_colspan)
    ax3 = plt.subplot2grid((nRow, nCol), (ax1_rowspan + divider_rowspan, ax2_colspan), rowspan=ax2_rowspan, colspan=ax2_colspan)
    ax4 = plt.subplot2grid((nRow, nCol), (0, ax1_colspan + divider_colspan), rowspan=ax1_rowspan, colspan=ax1_colspan)
    ax5 = plt.subplot2grid((nRow, nCol), (ax1_rowspan + divider_rowspan, ax1_colspan + divider_colspan), rowspan=ax2_rowspan, colspan=ax2_colspan)
    ax6 = plt.subplot2grid((nRow, nCol), (ax1_rowspan + divider_rowspan, ax1_colspan + divider_colspan + ax2_colspan), rowspan=ax2_rowspan, colspan=ax2_colspan)
    axes = {'Planktonic': [ax1, ax2, ax3],
            'Biofilm': [ax4, ax5, ax6]}
    for medium in MEDIA:
        for rpl in range(NUM_REPLICATES):
            plt.sca(axes[medium][rpl])
            ax = plt.gca()
            display_muller_plot(files[medium][rpl], plotShow=False)
    plt.show()


def display_muller_plot(file, plotShow=True):

    img = mpimg.imread(file)
    imgplot = plt.imshow(img)
    plt.box(on=None)
    plt.xticks(ticks=[], labels=[])
    plt.yticks(ticks=[], labels=[])
    if plotShow:
        plt.show()


def plot_inference_pipeline(gene_to_selection, mutations_to_genes, measured_MIC_dic, measured_MIC_range_dic, titlePostfix='', figsize=(10, 3)):

    all_inferred_fitness_by_mutations = map_mutations_to_inferred_fitness_list(gene_to_selection, mutations_to_genes)
    mean_inferred_fitness_by_mutations = map_mutations_to_mean_inferred_fitness(all_inferred_fitness_by_mutations)
    median_inferred_fitness_by_mutations = map_mutations_to_median_inferred_fitness(all_inferred_fitness_by_mutations)

    (measured_MIC, measured_MIC_error, mean_inferred_fitness,
     median_inferred_fitness) = get_lists_from_dics(measured_MIC_dic, mean_inferred_fitness_by_mutations,
                                                 median_inferred_fitness_by_mutations, measured_MIC_range_dic)

    plot_inference_against_measurement([mean_inferred_fitness, median_inferred_fitness], measured_MIC, ys_error=measured_MIC_error,
                                   titlePostfix=titlePostfix, figsize=figsize)


def plot_inference_against_measurement(xss, ys, ys_error=None, plotRange=True, labels=['Mean across replicates', 'Median across replicates'], figsize=(10, 4), fontsize=12, titlePostfix=None, ylabel='Measured MIC', xlabels=['Mean inferred fitness', 'Median inferred fitness']):

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    for i, xs in enumerate(xss):
        label, xlabel = labels[i], xlabels[i]
        plt.sca(axes[i])
        at_left = (i == 0)
        plt.scatter(xs, ys, alpha=0.6, label=label)
        if plotRange and ys_error is not None:
            count = 0
            for x, y, yerror in zip(xs, ys, ys_error):
                plt.errorbar([x], [y], yerr=np.array([yerror]).T, color='orange', alpha=0.6, label='Measured range' if count == 0 else None)
                count += 1

        plt.xlabel(xlabel, fontsize=fontsize)
        if at_left:
            plt.ylabel(ylabel, fontsize=fontsize)
        else:
            plt.yticks(ticks=[], labels=[])
        plt.title(f'Spearmanr=%.2f ' % (stats.spearmanr(xs, ys)[0]) + titlePostfix, fontsize=fontsize)
        plt.legend(fontsize=fontsize)
    plt.show()
