"""
This Python file contains code for parsing data from Harris, K. B., Flynn, K. M. & Cooper, V. S. Polygenic Adaptation and Clonal Interference Enable Sustained Diversity in Experimental Pseudomonas aeruginosa Populations. Mol. Biol. Evol. 38, 5359–5375 (2021) in:
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

# sys.path.append('../paper-clade-reconstruction/src')
import MPL as MPL
import analyze_and_plot as AP
import reconstruct_clades as RC
import data_parser as DP
import simulation_helper as SH
import lolipop_helper


DATA_DIR = './data'
FIG_DIR = './figures'
LOCAL_JOBS_DIR = './data/local_jobs'
JOB_DIR = './jobs'

# relative directories looking from shell scripts in JOB_DIR
DATA_DIR_REL = '../data'
SRC_DIR_REL = f'../src'

# Trajectories
PALTE_DIR = 'PALTEanalysis'
TRAJ_FILENAME = 'Table_S3.csv'
PALTE_TRAJ_FILE = f'{PALTE_DIR}/{TRAJ_FILENAME}'

# Fitness
FITNESS_FILE = 'PALTEanalysis/Table_S1.xlsx'
SHEETNAMES = ['selection rates 0-24 hr', 'selection rates 24-48 hr',  'selection rates 0-48 hr']
CONDITIONS = ['Plank', 'biof - bead', 'biof - plank portion']
SELECTED_SHEET = 'selection rates 0-24 hr'
SELECTED_CONDITIONS = ['biof - bead', 'biof - plank portion']

# Lolipop
LOLIPOP_INPUT_DIR = f'{DATA_DIR}/lolipop/input'
LOLIPOP_OUTPUT_DIR = f'{DATA_DIR}/lolipop/output'
LOLIPOP_PARSED_OUTPUT_DIR = f'{DATA_DIR}/lolipop/parsed_output'

# Evoracle
EVORACLE_PALTE_PARSED_OUTPUT_DIR = f'{DATA_DIR}/evoracle/PALTE_parsed_output'
EVORACLE_PALTE_DIR = f'{DATA_DIR}/evoracle/PALTE'
EVORACLE_PALTE_DIR_REL = f'{DATA_DIR_REL}/evoracle/PALTE'
EVORACLE_PALTE_PARSED_OUTPUT_DIR_REL = f'{DATA_DIR_REL}/evoracle/PALTE_parsed_output'

# Constants
POPULATIONS = ['B1', 'B2', 'B3', 'P1', 'P2', 'P3']
MEASURED_POPULATIONS = ['WT'] + POPULATIONS
OUR_METHOD_NAME = 'dxdx'  # 'recovered'
TRUE_COV_NAME = 'True'  # 'true_cov'
EST_METHOD_NAME = 'LB'  # 'est_cov'
METHODS = [OUR_METHOD_NAME, 'Lolipop', 'Evoracle', EST_METHOD_NAME]  # [OUR_METHOD_NAME, 'Lolipop', 'Evoracle', EST_METHOD_NAME, 'SL']  # ['recovered', 'Lolipop', 'est_cov', 'SL']
TIMES = np.array([0, 17, 25, 44, 66, 75, 90]) * 600 // 90
MU = 1e-6
REG = 4  # Adjustable through set_reg(reg)

try:
    df_traj
    df_fitness
except:
    df_traj, df_fitness = None, None

def load_df(data_dir='./data'):
    global df_traj, df_fitness
    df_traj = pd.read_csv(f'{data_dir}/{PALTE_TRAJ_FILE}')
    try:
        df_fitness = pd.read_excel(f'{data_dir}/{FITNESS_FILE}', engine='openpyxl', sheet_name=SHEETNAMES)
    except:
        try:
            with pd.ExcelFile(f'{data_dir}/{FITNESS_FILE}', engine="openpyxl") as excel:
                df_fitness = pd.read_excel(excel)
        except:
            df_fitness = pd.read_excel(f'{data_dir}/{FITNESS_FILE}', sheet_name=SHEETNAMES)


############################################
#
# Helper functions
#
############################################

def set_reg(reg):
    global REG
    REG = reg


def is_float(element):
    try:
        float(element)
        return True
    except ValueError:
        return False


def interpolate_traj(traj, times):
    T, L = traj.shape
    for l in range(L):
        for t in range(T):
            if traj[t, l] < 0:
                tl, tr = t, t
                while traj[tl, l] < 0 and tl > 0:
                    tl -= 1
                while traj[tr, l] < 0 and tr < T - 1:
                    tr += 1
                if t == 0 or t == T - 1 or traj[tl, l] < 0 or traj[tr, l] < 0:
                    print(f'Can not interpolate. Must expolate. t={t}, l={l}, tl={tl}, {traj[tl, l]}, tr={tr}, {traj[tr, l]}')
                    return
                traj[t, l] = traj[tl, l] + (traj[tr, l] - traj[tl, l]) / (times[tr] - times[tl]) * (times[t] - times[tl])
    return traj


############################################
#
# Parsing data & results
#
############################################

def parse_traj(df):
    times = np.array(sorted([int(col[1:]) for col in df.columns if col[0] == 'X' and col[1:].isdigit()]))
    T, L = len(times), len(df)
    traj = np.zeros((T, L), dtype=float)
    l = 0
    for i, row in df.iterrows():
        for t, time in enumerate(times):
            value = row['X' + str(time)]
            if np.isnan(value):
                traj[t, l] = -1
            else:
                traj[t, l] = float(value)
        l += 1
    return traj, times


def parse_interpolated_traj(pop, df):
    traj, times = parse_traj(df[df['Population'] == pop])
    return interpolate_traj(traj, times)


def parse_trajectories():
    global df_traj
    if df_traj is None:
        load_df()
    trajectories = {}
    for pop in POPULATIONS:
        trajectories[pop] = parse_interpolated_traj(pop, df_traj)
    return trajectories


def parse_muller_files(populations, output_dir=LOLIPOP_OUTPUT_DIR):
    files = {pop: None for pop in populations}
    for pop in populations:
        filename = f'{pop}.mullerdiagram.annotated.png'
        files[pop] = f'{output_dir}/{pop}/graphics/lineage/{filename}'
    return files


def get_reconstruction_from_traj(traj, times, mu=MU, defaultReg=REG, meanReadDepth=100, thFixed=0.98, thLogProbPerTime=10, thFreqUnconnected=1, debug=False, verbose=False, plot=False, evaluateReconstruction=True, evaluateInference=False, assumeCooperationAmongSharedMuts=False):
    print(defaultReg)
    rec = RC.CladeReconstruction(traj, times=times, meanReadDepth=meanReadDepth, debug=debug, verbose=verbose, plot=plot, mu=mu, useEffectiveMu=False)
    rec.setParamsForClusterization(weightByBothVariance=False, weightBySmallerVariance=True)
    rec.clusterMutations()
    rec.setParamsForReconstruction(thFixed=thFixed, thExtinct=0, numClades=None, thLogProbPerTime=thLogProbPerTime, thFreqUnconnected=thFreqUnconnected)
    rec.checkForSeparablePeriodAndReconstruct()
    rec.setParamsForEvaluation(defaultReg=defaultReg, assumeCooperationAmongSharedMuts=assumeCooperationAmongSharedMuts)
    evaluation, inference = rec.evaluate(evaluateReconstruction=evaluateReconstruction, evaluateInference=evaluateInference)
    return rec, evaluation, inference


def parse_reconstructions(reg=REG, assumeCooperationAmongSharedMuts=False):
    reconstructions = {pop: None for pop in POPULATIONS}
    evaluations = {pop: None for pop in POPULATIONS}
    inferences = {pop: None for pop in POPULATIONS}

    global df_traj
    if df_traj is None:
        load_df()

    for pop in POPULATIONS:
        traj = parse_interpolated_traj(pop, df_traj)
        (reconstructions[pop], evaluations[pop],
         inferences[pop]) = get_reconstruction_from_traj(traj, TIMES, defaultReg=reg, mu=MU, evaluateReconstruction=True, evaluateInference=True, assumeCooperationAmongSharedMuts=assumeCooperationAmongSharedMuts)

    return reconstructions, evaluations, inferences


def parse_Lolipop_results(muForInference=MU, regularization=REG, mus=None, populations=POPULATIONS):
    intCov_lolipop, selection_lolipop, fitness_lolipop = {pop: None for pop in populations}, {pop: None for pop in populations}, {pop: None for pop in populations}
    for pop in populations:
        file = f'{LOLIPOP_PARSED_OUTPUT_DIR}/{pop}.npz'
        data = lolipop_helper.loadParsedOutput(file)
        mu = muForInference if mus is None else mus[pop]
        intCov, selection, fitness = lolipop_helper.inferencePipeline(data, mu=mu, regularization=regularization)

        intCov_lolipop[pop] = intCov
        selection_lolipop[pop] = selection
        fitness_lolipop[pop] = fitness
    return intCov_lolipop, selection_lolipop, fitness_lolipop


def parse_evoracle_results(muForInference=MU, regularization=REG, mus=None, populations=POPULATIONS):
    intCov_evoracle, selection_evoracle, fitness_evoracle = {pop: None for pop in populations}, {pop: None for pop in populations}, {pop: None for pop in populations}
    for pop in populations:
        res = load_evoracle(pop)
        traj = res['traj']
        intCov = res['int_cov']
        selection = res['selection']
        fitness = lolipop_helper.computeFitness(traj, selection)

        intCov_evoracle[pop] = intCov
        selection_evoracle[pop] = selection
        fitness_evoracle[pop] = fitness
    return intCov_evoracle, selection_evoracle, fitness_evoracle


def parse_all_measured_fitness(numRows=6, th_large_fitness=9):
    fitness_all = {sheet: {f'{pop} {cond}': [] for pop in MEASURED_POPULATIONS for cond in CONDITIONS}
                   for sheet in SHEETNAMES}
    for sheet in SHEETNAMES:
        df = df_fitness[sheet]
        for pop in MEASURED_POPULATIONS:
            for cond in CONDITIONS:
                col = pop + ' ' + cond
                if col not in set(df.columns):
                    col = pop + ' ' + cond.lower()
                if col not in set(df.columns):
                    col += ' '
                fitness_list = list(df[col])
                for i in range(numRows):
                    if fitness_list[i] <= th_large_fitness:
                        fitness_all[sheet][f'{pop} {cond}'].append(fitness_list[i])
    return fitness_all


def parse_measured_fitness():
    fitness_all = parse_all_measured_fitness()
    pop_to_measured_fitness = {pop: None for pop in MEASURED_POPULATIONS}
    for pop in MEASURED_POPULATIONS:
        pop_to_measured_fitness[pop] = np.mean([np.median(fitness_all[SELECTED_SHEET][f'{pop} {cond}']) for cond in SELECTED_CONDITIONS])
    return pop_to_measured_fitness


def parse_inferred_fitness_for_Lolipop(fitness_lolipop, populations):
    inferred_fitness = {'WT': 1}
    for pop in populations:
        inferred_fitness[pop] = fitness_lolipop[pop][-1]
    return inferred_fitness


def parse_inferred_fitness_for_evoracle(fitness_evoracle, populations):
    inferred_fitness = {'WT': 1}
    for pop in populations:
        inferred_fitness[pop] = fitness_evoracle[pop][-1]
    return inferred_fitness


def parse_inferred_fitness_for_a_method(inferences, method, populations, fitness_index=3):
    inferred_fitness = {'WT': 1}
    for pop in populations:
        inferred_fitness[pop] = inferences[pop][method][fitness_index][-1]
    return inferred_fitness


def parse_inferred_fitness_list(inferences, fitness_lolipop, fitness_evoracle):

    # reconstructions, evaluations, inferences = parse_reconstructions()
    # intCov_lolipop, selection_lolipop, fitness_lolipop = parse_Lolipop_results()

    inferred_fitness_list = []
    for method in METHODS:
        if method == 'Lolipop':
            inferred_fitness_list.append(parse_inferred_fitness_for_Lolipop(fitness_lolipop, POPULATIONS))
        elif method == 'Evoracle':
            inferred_fitness_list.append(parse_inferred_fitness_for_evoracle(fitness_evoracle, POPULATIONS))
        else:
            inferred_fitness_list.append(parse_inferred_fitness_for_a_method(inferences, method, POPULATIONS))
    return inferred_fitness_list


def parse_inferred_selections(methods=METHODS):
    gene_to_selection = {method: {} for method in methods}

    # Lolipop
    for medium in media:
        for rpl in range(num_replicates):
            traj, times, genes = traj_pa[medium][rpl]
            infered_selection = selection_lolipop[medium][rpl]
            for l, gene in enumerate(genes):
                if gene not in gene_to_selection['Lolipop']:
                    gene_to_selection['Lolipop'][gene] = [infered_selection[l]]
                else:
                    gene_to_selection['Lolipop'][gene].append(infered_selection[l])

    # Other methods
    for method in methods[:3]:
        for medium in media:
            for rpl in range(num_replicates):
                traj, times, genes = traj_pa[medium][rpl]
                sum_inference = res_pa[medium][rpl][1]
                infered_selection = sum_inference[method][1]
                for l, gene in enumerate(genes):
                    if gene not in gene_to_selection[method]:
                        gene_to_selection[method][gene] = [infered_selection[l]]
                    else:
                        gene_to_selection[method][gene].append(infered_selection[l])


def parse_fitness_by_pop_into_list(measured_fitness_by_pop, inferred_fitness_by_pop_list, populations=MEASURED_POPULATIONS):

    measured_fitness = [measured_fitness_by_pop[pop] for pop in populations]
    inferred_fitness_list = []
    for inferred_fitness_by_pop in inferred_fitness_by_pop_list:
        inferred_fitness = [inferred_fitness_by_pop[pop] for pop in populations]
        inferred_fitness_list.append(inferred_fitness)

    return measured_fitness, inferred_fitness_list


############################################
#
# Evoracle
#
############################################


def save_traj_for_evoracle(trajectories=None):
    if trajectories is None:
        trajectories = parse_trajectories()
    for pop in POPULATIONS:
        traj = trajectories[pop]
        file = f'{EVORACLE_PALTE_DIR}/{pop}/PALTE_{pop}_obsreads.csv'
        DP.save_traj_for_evoracle(traj, file, times=TIMES)


def generate_evoracle_scripts(env='evoracle', save_geno_traj=True, partition='intel', overwrite=False):
    
    job_prefix = f'evoracle_PALTE'
    jobnames = {pop: f'{job_prefix}_pop={pop}' for pop in POPULATIONS}

    for pop in POPULATIONS:
        command = ''
        job_pars = {
            '-o': f'{EVORACLE_PALTE_PARSED_OUTPUT_DIR_REL}/evoracle_parsed_output_PALTE_pop={pop}.npz',
            '-i': f'PALTE_{pop}_obsreads.csv',
            '-d': f'{EVORACLE_PALTE_DIR_REL}/{pop}',
        }
        if save_geno_traj:
            job_pars['--save_geno_traj'] = ''
        command += f'python {SRC_DIR_REL}/evoracle_batch_job_real_data.py '
        command += ' '.join([k + ' ' + str(v) for k, v in job_pars.items()])
        command += '\n'
        SH.generate_shell_script(JOB_DIR, jobnames[pop], command, hours=12, env=env)

    jobname = f'{job_prefix}_submission'
    command = ''
    for pop in POPULATIONS:
        if overwrite or not test_single_evoracle(pop):
            command += f'sbatch -p {partition} {jobnames[pop]}.sh\n'
    SH.generate_shell_script(JOB_DIR, jobname, command, env=env)


def test_single_evoracle(pop):
    try:
        res = load_evoracle(pop)
        for key in list(res.keys()):
            res[key]
        return True
    except:
        return False


def load_evoracle(pop, directory=EVORACLE_PALTE_PARSED_OUTPUT_DIR):
    file = f'{directory}/evoracle_parsed_output_PALTE_pop={pop}.npz'
    return np.load(file, allow_pickle=True)


############################################
#
# Plotting functions
#
############################################

def plot_trajectories(trajectories, populations):
    nRow, nCol = 2, 3
    fig, axes = plt.subplots(2, 3, figsize=(10, 7))
    n = 0
    for pop in populations:
        traj = trajectories[pop]
        plt.sca(axes[n//nCol, n%nCol])
        AP.plotTraj(traj, scatter=True, times=TIMES, annot=False, alpha=0.6, title=f'{pop}',
                    plotFigure=False, plotShow=False)
        plt.yticks(rotation=90)
        if n == 0:
            plt.xlabel('Generations')
            plt.ylabel('Frequency')
        n += 1
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    plt.show()


def display_muller_plots(files, populations, titles=None, wspace=0.1, hspace=0.1, figsize=(20, 12), fontsize=15):
    nRow, nCol = 2, 3
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    n = 0
    for pop in populations:
        plt.sca(axes[n//nCol, n%nCol])
        display_muller_plot(files[pop], plotShow=False)
        if titles is not None and pop in titles:
            plt.title(titles[pop], fontsize=fontsize)
        n += 1
    plt.subplots_adjust(wspace=wspace, hspace=hspace)
    plt.show()


def display_muller_plot(file, plotShow=True):

    img = mpimg.imread(file)
    imgplot = plt.imshow(img)
    plt.xticks(ticks=[], labels=[])
    plt.yticks(ticks=[], labels=[])
    if plotShow:
        plt.show()


def compare_measured_and_inferred_fitness(measured_fitness, inferred_fitness, measured_populations,
                                      figsize=(10, 3.3), title=''):
    plt.figure(figsize=figsize)
    xs = [measured_fitness[pop] for pop in measured_populations]
    ys = [inferred_fitness[pop] for pop in measured_populations]
    plt.scatter(xs, ys)
    for i, pop in enumerate(measured_populations):
        plt.annotate(pop, (xs[i], ys[i]))
    plt.title(title + ', Spearmanr = %.2f' % stats.spearmanr(xs, ys)[0])
    plt.show()


def compare_measured_and_inferred_fitness_list(measured_fitness, inferred_fitness_list, measured_populations,
                                          labels=None, colors=None, alpha=0.6, figsize=(10, 3.3), title=''):

    plt.figure(figsize=figsize)
    xs = [measured_fitness[pop] for pop in measured_populations]
    for i, inferred_fitness in enumerate(inferred_fitness_list):
        label = labels[i] if labels is not None else None
        color = colors[i] if colors is not None else None
        ys = [inferred_fitness[pop] for pop in measured_populations]
        plt.scatter(xs, ys, color=color, alpha=alpha,
                    label=label + ', Spearmanr=%.2f' % stats.spearmanr(xs, ys)[0])
        if i == 0:
            for i, pop in enumerate(measured_populations):
                plt.annotate(pop, (xs[i], ys[i]), color='grey')
    plt.legend()
    plt.title(title)
    plt.show()
