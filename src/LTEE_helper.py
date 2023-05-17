import sys
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import scipy
from scipy import stats
from datetime import datetime
from tabulate import tabulate
from timeit import default_timer as timer   # timer for performance

import pickle

import importlib

sys.path.append('./src')
import analyze_and_plot as AP
import reconstruct_clades as RC
import simulation_helper as SH
import LTEE
import lolipop_helper
import data_parser as DP

DATA_DIR = './data'
JOB_DIR = './jobs'

# relative directories looking from shell scripts in JOB_DIR
DATA_DIR_REL = '../data'
EVORACLE_SRC_DIR_REL = '../src'

# LTEE
LTEE_DATA_DIR = f'{DATA_DIR}/LTEE-metagenomic-master/data_files/'
CLUSTER_JOBS_DIR = f'{DATA_DIR}/cluster_jobs'
LTEE_TRAJ_DIR = f'{DATA_DIR}/LTEE_trajectories'
RECONSTRUCTION_OUTPUT_DIR = f'{DATA_DIR}/reconstruction_output'
CLUSTERIZATION_OUTPUT_DIR = f'{DATA_DIR}/clusterization_output'

# Lolipop
LOLIPOP_DIR = f'{DATA_DIR}/lolipop'
LOLIPOP_JOBS_DIR = f'{LOLIPOP_DIR}/jobs'
LOLIPOP_INPUT_DIR = f'{LOLIPOP_DIR}/input'
LOLIPOP_OUTPUT_DIR = f'{LOLIPOP_DIR}/output'
LOLIPOP_PARSED_OUTPUT_DIR = f'{LOLIPOP_DIR}/parsed_output'

# Relative directory paths looking from CLUSTER_JOBS_DIR
LTEE_TRAJ_DIR_REL = '../LTEE_trajectories'
CLUSTERIZATION_OUTPUT_DIR_REL = '../clusterization_output'
CLUSTERING_SRC_DIR_REL = '../../src'

# Relative directory paths looking from LOLIPOP_JOBS_DIR
LOLIPOP_INPUT_DIR_REL = '../input'
LOLIPOP_OUTPUT_DIR_REL = '../output'

# Evoracle
EVORACLE_LTEE_PARSED_OUTPUT_DIR = f'{DATA_DIR}/evoracle/LTEE_parsed_output'
EVORACLE_LTEE_DIR = f'{DATA_DIR}/evoracle/LTEE'
EVORACLE_LTEE_DIR_REL = f'{DATA_DIR_REL}/evoracle/LTEE'
EVORACLE_LTEE_PARSED_OUTPUT_DIR_REL = f'{DATA_DIR_REL}/evoracle/LTEE_parsed_output'


all_lines = ['m5','p2','p4','p1','m6','p5','m1','m2','m3','m4','p3','p6']
focal_populations = ['p2','m6','m1']
remaining_populations = ['m5','p1','p4','p5','m2','m3','m4','p3','p6']
complete_nonmutator_lines = ['m5','m6','p1','p2','p4','p5']

populations = ['m1', 'm2', 'm3', 'm4', 'm5', 'm6', 'p1', 'p2', 'p3', 'p4', 'p5', 'p6']
populations_doable = ['m1', 'm2', 'm3', 'm4', 'm5', 'm6', 'p1', 'p2', 'p3', 'p4', 'p5']
populations_clonal = ['m1', 'm2', 'm4', 'm5', 'm6', 'p1', 'p3', 'p5', 'p6']
populations_nonclonal = [pop for pop in populations if pop not in populations_clonal]
populations_nonmutator = ['m5', 'm6', 'p1', 'p2', 'p4', 'p5']
populations_mutator = ['m1', 'm2', 'm3', 'm4', 'p3', 'p6']
populations_doable_mutator = ['m1', 'm2', 'm3', 'm4', 'p3']
populations_sorted_by_num_alleles = ['p2', 'p4', 'p5', 'm6', 'm5', 'p1', 'm3', 'm2', 'm1', 'm4', 'p3', 'p6']
num_alleles_sorted = [174, 201, 202, 221, 406, 466, 3255, 3488, 4597, 4872, 6102, 10134]

clade_hmm_states = {'A':0,'E':1,'FB':2,'FM':3, 'Fm':4,'PB':5,'PM':6,'Pm':7,'PB*':8}
well_mixed_hmm_states = {'A':0,'E':1,'F':2,'P':3}

UNBORN = clade_hmm_states['A']
EXTINCT= clade_hmm_states['E']
ANCESTRAL_FIXED = clade_hmm_states['FB']
MINOR_FIXED=clade_hmm_states['Fm']
MAJOR_FIXED=clade_hmm_states['FM']
ANCESTRAL_POLYMORPHIC=clade_hmm_states['PB']
MINOR_POLYMORPHIC=clade_hmm_states['Pm']
MAJOR_POLYMORPHIC=clade_hmm_states['PM']

mu = 1e-10
THRESHOLD_TIMEPOINTS = 60
STARTING_INDEX = 0

COLORS = {
    1: 'grey', # Extinct
    2: '#d62728', # Ancestor fixed
    3: '#1f77b4', # Major fixed
    4: '#ff7f0e', # Minor fixed
    5: '#e377c2', # Ancestor polymorphic
    6: '#2ca02c', # Major polymorphic
    7: 'yellow', # NA
 }
LABELS = {
    EXTINCT: 'Extinct', # Extinct
    ANCESTRAL_FIXED: 'Ancestor fixed', # Ancestor fixed
    MAJOR_FIXED: 'Major fixed', # Major fixed
    MINOR_FIXED: 'Minor fixed', # Minor fixed
    ANCESTRAL_POLYMORPHIC: 'Ancestor polymorphic', # Ancestor polymorphic
    MAJOR_POLYMORPHIC: 'Major polymorphic'# Major polymorphic
}
TIMES_INTPL = np.arange(122) * 500
TH_FIXED = 0.98

try:
    data
except NameError:
    data = {pop: {} for pop in populations}

try:
    fitness_ts, fitness_xs, fitness_ws
except NameError:
    fitness_ts, fitness_xs, fitness_ws = {}, {}, {}

try:
    fitness_ts_mean_nonmutator, fitness_xs_mean_nonmutator, fitness_ws_mean_nonmutator
except NameError:
    fitness_ts_mean_nonmutator, fitness_xs_mean_nonmutator, fitness_ws_mean_nonmutator = [], [], []

try:
    fitness_ts_mean_doable, fitness_xs_mean_doable, fitness_ws_mean_doable
except NameError:
    fitness_ts_mean_doable, fitness_xs_mean_doable, fitness_ws_mean_doable = [], [], []

############################################
#
# Load Data
#
############################################

def loadData(populations_selected, data_directory=None, verbose=False):
    for pop in populations_selected:
        (data[pop]['sites'], data[pop]['times'], data[pop]['counts'], data[pop]['depths'], data[pop]['freqs'], data[pop]['sites_intpl'], data[pop]['counts_intpl'], data[pop]['depths_intpl'], data[pop]['freqs_intpl']) = getInterpolatedReads(pop, TIMES_INTPL, threshold_timepoints=THRESHOLD_TIMEPOINTS, data_directory=data_directory, verbose=verbose)
        data[pop]['traj'] = np.array(data[pop]['freqs_intpl']).T

        dummy_times, data[pop]['fmajors'], data[pop]['fminors'], haplotype_trajectories = LTEE.parse_haplotype_timecourse(pop, data_directory=data_directory)
        data[pop]['site_to_clade'] = {site: haplotype_trajectories[i][-1] for i, site in enumerate(data[pop]['sites'])}
        data[pop]['clade_to_sites'] = getCladeMembers(data[pop]['site_to_clade'])


def loadFitness(populations_selected, filename="./data/LTEE-metagenomic-master/additional_data/Concatenated.LTEE.data.all.csv"):
    trajectories, line_data = LTEE.parse_ancestor_fitnesses(filename=filename)
    for pop in populations_selected:
        fitness_ts[pop] = np.array(trajectories[pop][0])
        fitness_xs[pop] = np.array(trajectories[pop][1])
        fitness_ws[pop] = np.array(trajectories[pop][2])


def loadMeanFitness():

    if not np.all([pop in fitness_ts for pop in populations_doable]):
        self.loadFitness(populations_doable)

    # mean fitness of all nonmutators
    ref = populations_nonmutator[0]
    gv = globals()
    gv['fitness_ts_mean_nonmutator'] = fitness_ts[ref] # All fitness_ts are the same for nonmutators
    gv['fitness_ws_mean_nonmutator'] = np.array([np.mean([fitness_ws[pop][t] for pop in populations_nonmutator]) for t in range(len(fitness_ws[ref]))])
    gv['fitness_xs_mean_nonmutator'] = np.array([np.mean([fitness_xs[pop][t] for pop in populations_nonmutator]) for t in range(len(fitness_ws[ref]))])


    # mean fitness of all doable populations
    gv['fitness_ts_mean_doable'] = fitness_ts['m1']
    populations_same_ts = set()
    for pop in populations_doable:
        if np.array_equal(fitness_ts[pop], fitness_ts['m1']):
            populations_same_ts.add(pop)
    gv['fitness_ws_mean_doable'] = np.array([np.mean([fitness_ws[pop][t] for pop in populations_same_ts]) for t in range(len(fitness_ts_mean_doable))])
    gv['fitness_xs_mean_doable'] = np.array([np.mean([fitness_xs[pop][t] for pop in populations_same_ts]) for t in range(len(fitness_ts_mean_doable))])


def getMeanFreqAfterFixation(dic, pop):
    appearance_time, fixation_time, clade = getAppearanceFixationForOnePopulation(pop)
    meanFreqAfterFixation = []
    for i, tFix in enumerate(fixation_time):
        if tFix < 122 * 500 and clade[i] == 2:
            _ = np.mean(dic[pop]['traj'][tFix // 500:, i])
            if _ > 0.8:
                meanFreqAfterFixation.append(_)
    return meanFreqAfterFixation


def getInterpolatedReads(pop, times_intpl, data_directory=None, threshold_timepoints=60, verbose=False):
    sites, times, counts, depths, freqs = getNoninterpolatedReads(pop, data_directory=data_directory, verbose=verbose)
    if verbose:
        print(f'\nFor population {pop}, there are {len(sites)} mutations, involving {len(np.unique(sites))} sites')

    num_mutations = len(times)
    sites_intpl, freqs_intpl, counts_intpl, depths_intpl = [], [], [], []

    for i in range(num_mutations):
        if len(times[i]) >= threshold_timepoints:
            f = LTEE.create_interpolation_function(times[i], freqs[i])
            tmp_freqs, tmp_counts, tmp_depths = getInterpolatedReadsForOneMutation(times_intpl, f, times[i], depths[i], counts[i])
            sites_intpl.append(sites[i])
            freqs_intpl.append(tmp_freqs)
            counts_intpl.append(tmp_counts)
            depths_intpl.append(tmp_depths)

    return sites, times, counts, depths, freqs, sites_intpl, counts_intpl, depths_intpl, freqs_intpl


def getNoninterpolatedReads(pop, data_directory=None, verbose=False):
    # load mutations
    mutations, depth_tuple = LTEE.parse_annotated_timecourse(pop, data_directory=data_directory)
    population_avg_depth_times, population_avg_depths, clone_avg_depth_times, clone_avg_depths = depth_tuple
    if verbose:
        print(f'loaded {len(mutations)} mutations for pop {pop}')

    return_sites = []
    return_times = []
    return_counts = []
    return_depths = []
    return_freqs = []

    for mutation_idx in range(0, len(mutations))[STARTING_INDEX:]:

        location, gene_name, allele, var_type, test_statistic, pvalue, cutoff_idx, depth_fold_change, depth_change_pvalue, times, alts, depths, clone_times, clone_alts, clone_depths = mutations[mutation_idx]

        good_idxs, filtered_alts, filtered_depths = LTEE.mask_timepoints(times, alts, depths, var_type, cutoff_idx, depth_fold_change, depth_change_pvalue)

        freqs = LTEE.estimate_frequencies(filtered_alts, filtered_depths)

        masked_times = times[good_idxs]
        masked_counts = filtered_alts[good_idxs]
        masked_depths = filtered_depths[good_idxs]
        masked_freqs = freqs[good_idxs]

        return_sites.append(location)
        return_times.append(masked_times)
        return_counts.append(masked_counts)
        return_depths.append(masked_depths)
        return_freqs.append(masked_freqs)

    return return_sites, return_times, return_counts, return_depths, return_freqs


def getInterpolatedReadsForOneMutation(times_intpl, f, times, depths, counts):

    indexTimeExsited = 0
    medianDepth = int(np.median(depths))
    tmp_freqs, tmp_counts, tmp_depths = [], [], []
    for t in times_intpl:
        tmp_freqs.append(f(t))
        if t in times:
            tmp_depths.append(depths[indexTimeExsited])
            tmp_counts.append(counts[indexTimeExsited])
            indexTimeExsited += 1
        else:
            tmp_depths.append(medianDepth)
            tmp_counts.append(int(medianDepth * f(t)))

    return tmp_freqs, tmp_counts, tmp_depths


def getCladeForOnePopulation(pop, data_directory=None):
    clade = []
    mutations, depth_tuple = LTEE.parse_annotated_timecourse(pop, data_directory=data_directory)
    dummy_times, fmajors, fminors, haplotype_trajectories = LTEE.parse_haplotype_timecourse(pop, data_directory=data_directory)
    for mutation_idx in range(0, len(mutations))[STARTING_INDEX:]:
        Ls = haplotype_trajectories[mutation_idx]
        clade.append(int(Ls[-1]))
    return clade


def getCladeMembers(site_to_clade):
    clades = {'ancestor': [], 'major': [], 'minor': [], 'extinct': []}
    for site, Ls in site_to_clade.items():
        if Ls == ANCESTRAL_FIXED or Ls == ANCESTRAL_POLYMORPHIC:
            clades['ancestor'].append(site)
        elif Ls == MAJOR_FIXED or Ls == MAJOR_POLYMORPHIC:
            clades['major'].append(site)
        elif Ls == MINOR_FIXED or Ls == MINOR_POLYMORPHIC:
            clades['minor'].append(site)
        else:
            clades['extinct'].append(site)
    return clades

def getAppearanceFixationForOnePopulation(pop, data_directory=None, threshold_timepoints=60, verbose=False):
    sites, times, counts, depths, freqs = getNoninterpolatedReads(pop, data_directory=data_directory, verbose=verbose)
    dummy_times, fmajors, fminors, haplotype_trajectories = LTEE.parse_haplotype_timecourse(pop, data_directory=data_directory)
    clade = []
    num_mutations = len(times)
    appearance_time = []
    fixation_time = []
    for i in range(0, num_mutations)[STARTING_INDEX:]:
        if len(times[i]) >= threshold_timepoints:
            times_intpl = 500 * np.arange(len(haplotype_trajectories[i]))
            f = LTEE.create_interpolation_function(times[i], freqs[i])
            tmp_freqs, tmp_counts, tmp_depths = getInterpolatedReadsForOneMutation(times_intpl, f, times[i], depths[i], counts[i])
            app, fix, stay = LTEE.calculate_appearance_fixation_time_from_clade_hmm(times_intpl, tmp_freqs, haplotype_trajectories[i])
            appearance_time.append(app)
            fixation_time.append(fix)
            clade.append(int(haplotype_trajectories[i][-1]))

    return appearance_time, fixation_time, clade


def getInterpolatedReadsExcludingMajorAndMinorClades(pop, times_intpl, data_directory=None, threshold_timepoints=60, verbose=False):
    sites, times, counts, depths, freqs = getNoninterpolatedReads(pop, data_directory=data_directory)
    clade = getCladeForOnePopulation(pop, data_directory=data_directory)
    if verbose:
        print(f'\nFor population {pop}, there are {len(sites)} mutations, involving {len(np.unique(sites))} sites')

    num_mutations = len(times)
    sites_intpl, freqs_intpl, counts_intpl, depths_intpl = [], [], [], []

    for i in range(num_mutations):
        # [3, 4, 6, 7] are codes for 'FM', 'Fm', 'PM', 'Pm'
        if len(times[i]) >= threshold_timepoints and clade[i] not in [3, 4, 6, 7]:
            f = LTEE.create_interpolation_function(times[i], freqs[i])
            tmp_freqs, tmp_counts, tmp_depths = getInterpolatedReadsForOneMutation(times_intpl, f, times[i], depths[i], counts[i])
            sites_intpl.append(sites[i])
            freqs_intpl.append(tmp_freqs)
            counts_intpl.append(tmp_counts)
            depths_intpl.append(tmp_depths)

    return sites, counts_intpl, depths_intpl, freqs_intpl


def getInterpolatedReadsWithOnlyMajorAndMinorClades(pop, times_intpl, data_directory=None, threshold_timepoints=60, verbose=False):
    sites, times, counts, depths, freqs = getNoninterpolatedReads(pop, data_directory=data_directory)
    clade = getCladeForOnePopulation(pop, data_directory=data_directory)
    if verbose:
        print(f'\nFor population {pop}, there are {len(sites)} mutations, involving {len(np.unique(sites))} sites')

    num_mutations = len(times)
    sites_intpl, freqs_intpl, counts_intpl, depths_intpl = [], [], [], []
    # record clades information of returned mutations
    clades_return = []
    for i in range(num_mutations):
        if len(times[i]) >= threshold_timepoints and clade[i] in [3, 4, 6, 7]:
            f = LTEE.create_interpolation_function(times[i], freqs[i])
            tmp_freqs, tmp_counts, tmp_depths = getInterpolatedReadsForOneMutation(times_intpl, f, times[i], depths[i], counts[i])
            sites_intpl.append(sites[i])
            freqs_intpl.append(tmp_freqs)
            counts_intpl.append(tmp_counts)
            depths_intpl.append(tmp_depths)
            clades_return.append(clade[i])

    return sites, counts_intpl, depths_intpl, freqs_intpl, clades_return


def getAllelFreq(pop, data_directory=None, verbose=False):

    mutations, depth_tuple = LTEE.parse_annotated_timecourse(pop, data_directory=data_directory)
    population_avg_depth_times, population_avg_depths, clone_avg_depth_times, clone_avg_depths = depth_tuple
    if verbose:
        print(f'loaded {len(mutations)} mutations for pop {pop}')

    return_sites = []
    return_times = []
    return_freqs = []

    for mutation_idx in range(0,len(mutations))[STARTING_INDEX:]:

        location, gene_name, allele, var_type, test_statistic, pvalue, cutoff_idx, depth_fold_change, depth_change_pvalue, times, alts, depths, clone_times, clone_alts, clone_depths = mutations[mutation_idx]

        good_idxs, filtered_alts, filtered_depths = LTEE.mask_timepoints(times, alts, depths, var_type, cutoff_idx, depth_fold_change, depth_change_pvalue)

        freqs = LTEE.estimate_frequencies(filtered_alts, filtered_depths)

        masked_times = times[good_idxs]
        masked_freqs = freqs[good_idxs]

        return_sites.append(location)
        return_times.append(masked_times)
        return_freqs.append(masked_freqs)

    if verbose:
        print(f'after further masking, {len(return_freqs)} mutations remains')

    return return_sites, return_times, return_freqs


############################################
#
# Clusterization jobs
#
############################################  


def getFixationTime(freqs, thFixed=0.98, thNumTimePointsAfterFixation=2, thFractionTimePointsAfterFixation=0.05):
    """
    Returns fixation time for a mutation (index l), if it gets fixed. Otherwise returns -1.
    """
    T = len(freqs)
    maxFixationTime = max(0, T - max(thNumTimePointsAfterFixation, int(T * thFractionTimePointsAfterFixation)))
    freqSum = np.sum(freqs)
    for t in range(maxFixationTime):
        if freqSum / (T - t) >= thFixed and freqs[t] >= thFixed:
            return t
        freqSum -= freqs[t]
    return -1


def getExtinctionTime(freqs, thExtinct=0.01, thNumTimePointsAfterFixation=2, thFractionTimePointsAfterFixation=0.05):
    """
    Returns fixation time for a mutation (index l), if it gets fixed. Otherwise returns -1.
    """
    T = len(freqs)
    maxExtinctionTime = max(0, T - max(thNumTimePointsAfterFixation, int(T * thFractionTimePointsAfterFixation)))
    freqSum = np.sum(freqs)
    for t in range(maxExtinctionTime):
        if freqSum / (T - t) <= thExtinct and freqs[t] <= thExtinct:
            return t
        freqSum -= freqs[t]
    return -1


def get_flattened_traj(populations_selected=populations, thFixed=0.95, thExtinct=0.05):
    for pop in populations_selected:
        traj_flattened, num_flattened = flatten_traj_after_fixation_or_extinction(data[pop]['traj'], thFixed=thFixed, thExtinct=thExtinct)
        data[pop]['traj_flattened'] = traj_flattened


def flatten_traj_after_fixation_or_extinction(traj, thFixed=0.95, thExtinct=0.05):
    T, L = traj.shape
    copy = np.copy(traj)
    num_flattened = 0

    for l in range(L):
        fixationTime = getFixationTime(copy[:, l], thFixed=thFixed)
        extinctionTime = getExtinctionTime(copy[:, l], thExtinct=thExtinct)
        if fixationTime >= 0:
            copy[fixationTime:, l] = 1
            num_flattened += 1
        elif extinctionTime >= 0:
            copy[extinctionTime:, l] = 0
            num_flattened += 1

    return copy, num_flattened


def save_LTEE_trajectories(populations_selected=populations):
    for pop in populations_selected:
        with open(LTEE_TRAJ_DIR + f'/LTEE_traj_pop={pop}.npy', 'wb') as file:
            np.save(file, data[pop]['traj'])


def save_LTEE_trajectories_flattened(populations_selected=populations):
    for pop in populations_selected:
        with open(LTEE_TRAJ_DIR + f'/LTEE_traj_flattened_pop={pop}.npy', 'wb') as file:
            np.save(file, data[pop]['traj_flattened'])


def save_LTEE_trajectories_for_a_pop(pop, tStart=None, tEnd=None):
    if tStart is None:
        tStart = 0
    if tEnd is None:
        tEnd = len(TIMES_INTPL)

    with open(LTEE_TRAJ_DIR + f'/LTEE_traj_pop={pop}_tStart={tStart}_tEnd={tEnd}.npy', 'wb') as file:
        np.save(file, data[pop]['traj'][tStart:tEnd])


def generate_clustering_job_script(populations_selected=populations, env='env_pySCA', partition='batch', weightBySmallerInterpolatedVariance=True, use_flattened_traj=False):
    if use_flattened_traj:
        postfix = '_flattened'
    else:
        postfix = ''
    for pop in populations_selected:
        jobname = f'clusterization{postfix}_pop={pop}'
        job_pars = {'-i': f'{LTEE_TRAJ_DIR_REL}/LTEE_traj{postfix}_pop={pop}.npy',
                    '-o': f'{CLUSTERIZATION_OUTPUT_DIR_REL}/clusterization{postfix}_output_pop={pop}',}
        if weightBySmallerInterpolatedVariance:
            job_pars['--weightBySmallerInterpolatedVariance'] = ''
        else:
            job_pars['--weightBySmallerVariance'] = ''
        command = f'python {CLUSTERING_SRC_DIR_REL}/clusterization_cluster_job.py '
        command += ' '.join([k + ' ' + str(v) for k, v in job_pars.items()])
        command += '\n'
        if pop == 'p6':
            mem = 200
        elif pop == 'p3':
            mem = 64
        else:
            mem = 32
        SH.generate_shell_script(CLUSTER_JOBS_DIR, jobname, command, mem=mem, hours=12, env=env)
        
    jobname = f'clusterization{postfix}_submission'
    command = ''
    for pop in populations_selected:
        if pop == 'p6':
            command += f'sbatch -p highmem clusterization{postfix}_pop={pop}.sh\n'
        else:
            command += f'sbatch -p {partition} clusterization{postfix}_pop={pop}.sh\n'
    SH.generate_shell_script(CLUSTER_JOBS_DIR, jobname, command)


def generate_clustering_job_script_for_a_pop(pop, tStart_list, tEnd_list, env='env_pySCA', partition='batch', weightBySmallerInterpolatedVariance=True):
    for i, (tStart, tEnd) in enumerate(zip(tStart_list, tEnd_list)):
        jobname = f'clusterization_pop={pop}_tStart={tStart}_tEnd={tEnd}'
        job_pars = {'-i': f'{LTEE_TRAJ_DIR_REL}/LTEE_traj_pop={pop}_tStart={tStart}_tEnd={tEnd}.npy',
                    '-o': f'{CLUSTERIZATION_OUTPUT_DIR_REL}/clusterization_output_pop={pop}_tStart={tStart}_tEnd={tEnd}',}
        if weightBySmallerInterpolatedVariance:
            job_pars['--weightBySmallerInterpolatedVariance'] = ''
        else:
            job_pars['--weightBySmallerVariance'] = ''
        command = f'python {CLUSTERING_SRC_DIR_REL}/clusterization_cluster_job.py '
        command += ' '.join([k + ' ' + str(v) for k, v in job_pars.items()])
        command += '\n'
        if pop == 'p6':
            mem = 200
        elif pop == 'p3':
            mem = 64
        else:
            mem = 32
        SH.generate_shell_script(CLUSTER_JOBS_DIR, jobname, command, mem=mem, hours=12, env=env)
    
    jobname = f'clusterization_pop={pop}_submission'
    command = ''
    for i, (tStart, tEnd) in enumerate(zip(tStart_list, tEnd_list)):
        if pop == 'p6':
            command += f'sbatch -p highmem clusterization_pop={pop}_tStart={tStart}_tEnd={tEnd}.sh\n'
        else:
            command += f'sbatch -p {partition} clusterization_pop={pop}_tStart={tStart}_tEnd={tEnd}.sh\n'
    SH.generate_shell_script(CLUSTER_JOBS_DIR, jobname, command)

############################################
#
# Clade Reconstruction
#
############################################

def getClusterForAPopulation(mu=1e-10, debug=False, verbose=False, plot=True):
    simulation = RC.CladeReconstruction(data[pop]['traj'], mutantReads=data[pop]['counts_intpl'],
                   readDepths=data[pop]['depths_intpl'], times=TIMES_INTPL, mu=mu, hasInterpolation=True,
                   debug=debug, verbose=verbose, plot=plot)
    simulation.setParamsForClusterization(weightByBothVariance=False, weightBySmallerVariance=True)
    simulation.clusterMutations()
    return simulation.groups, simulation.segmentedIntDxdx


def reconstructForAPopulationAsOnePeriod(pop, tStart=0, tEnd=len(TIMES_INTPL), groups=None, segmentedIntDxdx=None, clusterResult=None, mu=1e-10, percentMutsToIncludeInMajorClades=90, percentSignalsToIncludeInMajorClades=95, thFixed=TH_FIXED, thExtinct=0.01, thFixedWithinClade=0.9, thCollapse=0.9, numClades=None, useEffectiveMu=False, debug=False, verbose=False, plot=True, timing=True):
    if plot:
        print('<' + '-'*90 + '>')
    if groups is None and clusterResult is not None:
        reconstruction = clusterResult
    else:
        reconstruction = RC.CladeReconstruction(data[pop]['traj'][tStart:tEnd], mutantReads=np.array(data[pop]['counts_intpl'])[:, tStart:tEnd], readDepths=np.array(data[pop]['depths_intpl'])[:, tStart:tEnd], times=TIMES_INTPL[tStart:tEnd], mu=mu, hasInterpolation=True, segmentedIntDxdx=segmentedIntDxdx, groups=groups, useEffectiveMu=useEffectiveMu, debug=debug, verbose=verbose, plot=plot, timing=timing)
        reconstruction.setParamsForClusterization(weightByBothVariance=False, weightBySmallerVariance=False, weightBySmallerInterpolatedVariance=True, timing=timing)
        reconstruction.clusterMutations()
    reconstruction.setParamsForReconstruction(thFixed=thFixed, thExtinct=thExtinct, thFixedWithinClade=thFixedWithinClade, numClades=numClades, percentSignalsToIncludeInMajorClades=percentSignalsToIncludeInMajorClades, thLogProbPerTime=10, timing=timing)
    reconstruction.reconstructCladeCompetition()
    reconstruction.wrapResultsAsOnePeriod()
    return reconstruction


def reconstruct_for_a_period(pop, tStart, tEnd, thFixed=0.98, thExtinct=0.01, thFixedWithinClade=0.9, timing=False, debug=True, verbose=True, plot=True):
    groups = load_clusterization_for_LTEE(pop, tStart=tStart, tEnd=tEnd)
    segmentedIntDxdx = load_dxdx_for_LTEE(pop, tStart=tStart, tEnd=tEnd)
    reconstruction = RC.CladeReconstruction(data[pop]['traj'][tStart:tEnd], mutantReads=np.array(data[pop]['counts_intpl'])[:, tStart:tEnd], 
        readDepths=np.array(data[pop]['depths_intpl'])[:, tStart:tEnd], times=TIMES_INTPL[tStart:tEnd], mu=1e-10, hasInterpolation=True, 
        segmentedIntDxdx=segmentedIntDxdx, groups=groups, useEffectiveMu=False, debug=debug, verbose=verbose, plot=plot, timing=timing)
    reconstruction.setParamsForClusterization(weightByBothVariance=False, weightBySmallerVariance=False, 
                                              weightBySmallerInterpolatedVariance=True, timing=timing)
    reconstruction.clusterMutations()
    reconstruction.setParamsForReconstruction(thFixed=thFixed, thExtinct=thExtinct, thFixedWithinClade=thFixedWithinClade, 
                                              percentSignalsToIncludeInMajorClades=95, thLogProbPerTime=10, timing=timing)
    reconstruction.reconstructCladeCompetition()
    reconstruction.wrapResultsAsOnePeriod()
    return reconstruction


def reconstructForAPopulation(pop, tStart=0, tEnd=len(TIMES_INTPL), groups=None, segmentedIntDxdx=None, clusterResult=None, mu=1e-10, thFixed=0.99, thExtinct=0.01, thFixedWithinClade=0.9, thCollapse=0.9, numClades=None, cladeFixedTimes=None, useEffectiveMu=False, debug=False, verbose=False, plot=True, timing=True):
    if plot:
        print('<' + '-'*90 + '>')
    if groups is None and clusterResult is not None:
        reconstruction = clusterResult
    else:
        reconstruction = RC.CladeReconstruction(data[pop]['traj'][tStart:tEnd], mutantReads=np.array(data[pop]['counts_intpl'])[:, tStart:tEnd], readDepths=np.array(data[pop]['depths_intpl'])[:, tStart:tEnd], times=TIMES_INTPL[tStart:tEnd], mu=mu, hasInterpolation=True, segmentedIntDxdx=segmentedIntDxdx, groups=groups, useEffectiveMu=useEffectiveMu, debug=debug, verbose=verbose, plot=plot, timing=timing)
        reconstruction.setParamsForClusterization(weightByBothVariance=False, weightBySmallerVariance=False, weightBySmallerInterpolatedVariance=True, timing=timing)
        reconstruction.clusterMutations()
    reconstruction.setParamsForReconstruction(thFixed=thFixed, thExtinct=thExtinct, thFixedWithinClade=thFixedWithinClade, numClades=numClades, percentSignalsToIncludeInMajorClades=95, thLogProbPerTime=10, timing=timing)
    reconstruction.checkForSeparablePeriodAndReconstruct(cladeFixedTimes=cladeFixedTimes)
    return reconstruction


def load_clusterization_for_LTEE(pop, directory=CLUSTERIZATION_OUTPUT_DIR, tStart=None, tEnd=None, flattened=False):

    if flattened:
        postfix = '_flattened'
    else:
        postfix = ''

    if tStart is None and tEnd is None:
        with open(directory + f'/clusterization{postfix}_output_pop={pop}.npz', 'rb') as fp:
            return np.load(fp, allow_pickle=True)['groups']
    else:
        if tStart is None:
            tStart = 0
        if tEnd is None:
            tEnd = len(TIMES_INTPL)
        with open(directory + f'/clusterization{postfix}_output_pop={pop}_tStart={tStart}_tEnd={tEnd}.npz', 'rb') as fp:
            return np.load(fp, allow_pickle=True)['groups']


def load_dxdx_for_LTEE(pop, directory=CLUSTERIZATION_OUTPUT_DIR, tStart=None, tEnd=None, flattened=False):
    if flattened:
        postfix = '_flattened'
    else:
        postfix = ''
    if tStart is None and tEnd is None:
        with open(directory + f'/clusterization{postfix}_output_pop={pop}.npz', 'rb') as fp:
            return np.load(fp, allow_pickle=True)['segmentedIntDxdx']
    else:
        if tStart is None:
            tStart = 0
        if tEnd is None:
            tEnd = len(TIMES_INTPL)
        with open(directory + f'/clusterization{postfix}_output_pop={pop}_tStart={tStart}_tEnd={tEnd}.npz', 'rb') as fp:
            return np.load(fp, allow_pickle=True)['segmentedIntDxdx']



def load_reconstruction_for_LTEE(pop, thFixed=TH_FIXED, directory=RECONSTRUCTION_OUTPUT_DIR, flattened=False):
    with open(directory + f'/reconstruction_output_pop={pop}_thFixed={thFixed}.obj', 'rb') as fp:
        return pickle.load(fp)


def load_reconstructions_for_LTEE(populations, thFixed=TH_FIXED, directory=RECONSTRUCTION_OUTPUT_DIR, flattened=False):
    return {pop: load_reconstruction_for_LTEE(pop, thFixed=thFixed, directory=directory, flattened=flattened) for pop in populations}


def reconstruct_for_LTEE_and_save(populations, thFixed=TH_FIXED, verbose=False, flattened=False):
    if flattened:
        postfix = '_flattened'
    else:
        postfix = ''
    for pop in populations:
        groups = load_clusterization_for_LTEE(pop, flattened=flattened)
        segmentedIntDxdx = load_dxdx_for_LTEE(pop, flattened=flattened)
        if verbose:
            print(f"Running for pop {pop}, {len(data[pop]['sites_intpl'])} mutations...")
        res = reconstructForAPopulationAsOnePeriod(pop, groups=groups, segmentedIntDxdx=segmentedIntDxdx, thFixed=thFixed,
            debug=False, verbose=False, plot=False, timing=False)
        if verbose:
            print('\tfinished. Saving...\n')
        fp = open(RECONSTRUCTION_OUTPUT_DIR + f'/reconstruction{postfix}_output_pop={pop}_thFixed={thFixed}.obj', 'wb')
        pickle.dump(res, fp, protocol=4)
        fp.close()


def print_reconstcution_info(reconstruction):
    pass


############################################
#
# Lolipop
#
############################################

def test_load_lolipop_input_LTEE(pop):
    try:
        _ = pd.read_table(LOLIPOP_INPUT_DIR + f'/LTEE_interpolated_pop={pop}.tsv')
        return True
    except:
        return False


def create_tables_for_lolipop(pop, overwrite=False):
    if not overwrite and test_load_lolipop_input_LTEE(pop):
        return
    filename = f'LTEE_interpolated_pop={pop}.tsv'
    lolipop_helper.saveTrajectoriesToTables(data[pop]['traj'], LOLIPOP_INPUT_DIR + f'/{filename}', sep='\t')
    print(f'Created for pop={pop}')


def generate_lolipop_command_for_LTEE_data(pop):
    filename = f'LTEE_interpolated_pop={pop}.tsv'
    output_directory = f'{LOLIPOP_OUTPUT_DIR_REL}/{pop}'
    command = f"lolipop lineage --input {LOLIPOP_INPUT_DIR_REL}/{filename} --output {output_directory}"
    return command


def generate_lolipop_scipts_for_LTEE_data(populations_selected=populations, env='env_pySCA', partition='batch'):

    job_prefix = 'lolipop_LTEE'
    for pop in populations_selected:
        jobname = f'{job_prefix}_pop={pop}'
        command = generate_lolipop_command_for_LTEE_data(pop)
        if pop in populations_nonmutator:
            hours = 16
            mem = 16
        else:
            hours = 336  # Two weeks
            if pop == 'p6':
                mem = 100
            else:
                mem = 32
        SH.generate_shell_script(LOLIPOP_JOBS_DIR, jobname, command, mem=mem, hours=hours, env=env)

    jobname = f'{job_prefix}_mkdir'
    command = ''
    for pop in populations_selected:
        command += f'mkdir {LOLIPOP_OUTPUT_DIR_REL}/{pop}\n'
    SH.generate_shell_script(LOLIPOP_JOBS_DIR, jobname, command, env=env)

    jobname = f'{job_prefix}_submission'
    command = ''
    for pop in populations_selected:
        if pop == 'p6':
            command += f'sbatch -p highmem {job_prefix}_pop={pop}.sh\n'
        else:
            command += f'sbatch -p {partition} {job_prefix}_pop={pop}.sh\n'
    SH.generate_shell_script(LOLIPOP_JOBS_DIR, jobname, command, env=env)



############################################
#
# Evoracle
#
############################################


def save_traj_for_evoracle(overwrite=False):
    for pop in populations:
        traj = data[pop]['traj']
        file = f'{EVORACLE_LTEE_DIR}/{pop}/LTEE_{pop}_obsreads.csv'
        DP.save_traj_for_evoracle(traj, file, times=TIMES_INTPL)


def generate_evoracle_scripts(env='evoracle', save_geno_traj=True, partition='batch', overwrite=False):
    
    job_prefix = f'evoracle_LTEE'
    jobnames = {pop: f'{job_prefix}_pop={pop}' for pop in populations}

    for pop in populations:
        command = ''
        job_pars = {
            '-o': f'{EVORACLE_LTEE_PARSED_OUTPUT_DIR_REL}/evoracle_parsed_output_LTEE_pop={pop}.npz',
            '-i': f'LTEE_{pop}_obsreads.csv',
            '-d': f'{EVORACLE_LTEE_DIR_REL}/{pop}',
        }
        if save_geno_traj:
            job_pars['--save_geno_traj'] = ''
        command += f'python {EVORACLE_SRC_DIR_REL}/evoracle_batch_job_real_data.py '
        command += ' '.join([k + ' ' + str(v) for k, v in job_pars.items()])
        command += '\n'
        if pop in populations_nonmutator:
            hours = 16
        else:
            hours = 336  # Two weeks
        SH.generate_shell_script(JOB_DIR, jobnames[pop], command, hours=hours, env=env)

    jobname = f'{job_prefix}_submission'
    command = ''
    for pop in populations:
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


def load_evoracle(pop, directory=EVORACLE_LTEE_PARSED_OUTPUT_DIR):
    file = f'{directory}/evoracle_parsed_output_LTEE_pop={pop}.npz'
    return np.load(file, allow_pickle=True)



############################################
#
# Compare running time
#
############################################


def get_all_methods_run_time(print=True):

    reconstruction_run_time = get_reconstruction_run_time()
    run_time = {
        'recovered': [],
        'Lolipop': [],
        'Evoracle': [],
    }
    for i, pop in enumerate(populations_sorted_by_num_alleles):
        run_time['recovered'].append(get_clustering_run_time_for_a_pop(pop) + reconstruction_run_time[i])
        run_time['Lolipop'].append(get_lolipop_run_time_for_a_pop(pop))
        run_time['Evoracle'].append(get_evoracle_run_time_for_a_pop(pop))

    if print:
        print_all_methods_run_time(run_time)

    return num_alleles_sorted, run_time


def print_all_methods_run_time(run_time):

    headers = [''] + populations_sorted_by_num_alleles
    row_num_alleles = ['num_alleles'] + num_alleles_sorted
    row_recovered = ['recovered (h)'] + ['%.2f' % _ for _ in run_time['recovered']]
    row_Lolipop = ['Lolipop (h)'] + ['%.2f' % _ if _ != 'nan' else _ for _ in run_time['Lolipop']]
    row_Evoracle = ['Evoracle (h)'] + ['%.2f' % _ if _ != 'nan' else _ for _ in run_time['Evoracle']]
    rows = [row_num_alleles, row_recovered, row_Lolipop, row_Evoracle]

    print(tabulate(rows, headers, tablefmt='plain', numalign="right"))


def load_reconstruction_run_time(filename='reconstruction_run_time'):
    # with open(f'{RECONSTRUCTION_OUTPUT_DIR}/reconstruction_run_time.pkl', 'rb') as f:
    #     return pickle.load(f)
    return np.load(f'{RECONSTRUCTION_OUTPUT_DIR}/{filename}.npy')


def get_reconstruction_run_time(overwrite=False, save=True, filename='reconstruction_run_time'):
    if not overwrite:
        try:
            run_time = load_reconstruction_run_time(filename=filename)
            return run_time
        except:
            pass

    run_time = []
    for pop in populations_sorted_by_num_alleles:
        start = timer()
        groups = load_clusterization_for_LTEE(pop)
        reconstruction = reconstructForAPopulationAsOnePeriod(pop, groups=groups, plot=False)
        time_in_hour = (timer() - start) / 3600
        run_time.append(time_in_hour)

    if save:
        np.save(f'{RECONSTRUCTION_OUTPUT_DIR}/{filename}.npy', run_time)

    return np.array(run_time)


def get_clustering_run_time():
    pass


def get_clustering_run_time_for_a_pop(pop):
    return extract_run_time_from_stdout_file(get_clustering_job_stdout_file(pop))


def get_lolipop_run_time_for_a_pop(pop):
    return extract_run_time_from_stdout_file(get_lolipop_job_stdout_file(pop))


def get_lolipop_process_for_a_pop(pop):
    return extract_last_info_from_stdout_file(get_lolipop_job_stdout_file(pop))


def get_evoracle_run_time_for_a_pop(pop):
    return extract_run_time_from_stdout_file(get_evoracle_job_stdout_file(pop))


def get_clustering_job_stdout_file(pop):
    return f'{CLUSTER_JOBS_DIR}/clusterization_pop={pop}.stdout'


def get_lolipop_job_stdout_file(pop):
    return f'{LOLIPOP_JOBS_DIR}/lolipop_LTEE_pop={pop}.stdout'


def get_evoracle_job_stdout_file(pop):
    return f'{JOB_DIR}/evoracle_LTEE_pop={pop}.stdout'


def extract_run_time_from_stdout_file(file, running_marker='nan'):
    with open(file, 'r') as fp:
        lines = fp.readlines()
        if len(lines) <= 2:
            return running_marker
        try:
            start_datetime = parse_date_from_stdout_line(lines[1])
            end_datetime = parse_date_from_stdout_line(lines[-1])
            run_time = get_hour_between_dates(start_datetime, end_datetime)
            return run_time
        except:
            return running_marker


def extract_last_info_from_stdout_file(file, num_last_chars=200):
    with open(file, 'r') as fp:
        lines = fp.readlines()
        info = ''
        for i in range(len(lines) - 1, -1, -1):
            info = lines[i][-num_last_chars:] + info
            if len(info) >= num_last_chars:
                break
        return info


def parse_date_from_stdout_line(line):
    # Sun Jan 22 20:09:42 PST 2023
    # print(line)
    return datetime.strptime(line.strip(), '%a %b %d %H:%M:%S %Z %Y')


def get_hour_between_dates(start_datetime, end_datetime):
    delta = end_datetime - start_datetime
    return delta.days * 24 + delta.seconds / 3600


############################################
#
# Plotting
#
############################################

def plotDataForAllPop(fontsize=15, alpha=0.3):
    fig, axes = plt.subplots(len(data), 1, figsize=(12, 4 * len(data)))
    for i, pop in enumerate(data.keys()):
        plt.sca(axes[i])
        for l, site in enumerate(data[pop]['sites']):
            c = data[pop]['site_to_clade'][site]
            plt.plot(data[pop]['times'][l], data[pop]['freqs'][l], color=COLORS[c], alpha=alpha)
        plt.title(f'pop {pop}', fontsize=fontsize)
    plt.show()


def plotDataForAllPop_TwoColumn(populations_selected=None, fontsize=10, alpha=0.05):
    if populations_selected is not None:
        nRow, nCol = len(populations_selected) // 2, 2
        nonmutator = np.intersect1d(populations_nonmutator, populations_selected)
        mutator = np.intersect1d(populations_mutator, populations_selected)
    else:
        nRow, nCol = len(data) // 2, 2
        nonmutator = populations_nonmutator
        mutator = populations_mutator
    fig, axes = plt.subplots(nRow, nCol, figsize=(10, 2 * nRow))
    for i, pop in enumerate(nonmutator):
        plt.sca(axes[i, 0])
        plotDataForOnePop(pop, fontsize=fontsize, alpha=alpha, plotFigure=False, plotShow=False, plotLegend=False)
        if i != nRow - 1:
            plt.xticks(ticks=[], labels=[])

    for i, pop in enumerate(mutator):
        plt.sca(axes[i, 1])
        plotDataForOnePop(pop, fontsize=fontsize, alpha=alpha, plotFigure=False, plotShow=False, plotLegend=(i==0))
        if i != nRow - 1:
            plt.xticks(ticks=[], labels=[])
    plt.show()


def plotDataForOnePop(pop, fontsize=15, alpha=0.3, figsize=(10, 3.3), plotFigure=True, plotShow=True, plotLegend=True, plotIntpl=False):
    if plotFigure:
        plt.figure(figsize=figsize)
    plotted = set()
    if plotIntpl:
        sites = data[pop]['sites_intpl']
    else:
        sites = data[pop]['sites']
    for l, site in enumerate(sites):
        if plotIntpl:
            times = TIMES_INTPL
            freqs = data[pop]['traj'][:, l]
        else:
            times = data[pop]['times'][l]
            freqs = data[pop]['freqs'][l]
        c = data[pop]['site_to_clade'][site]
        if c not in plotted:
            plt.plot(times, freqs, alpha=alpha, color=COLORS[c], label=LABELS[c])
            plotted.add(c)
        else:
            plt.plot(times, freqs, alpha=alpha, color=COLORS[c])
        plt.title(f"pop {pop} {len(sites)} mutations", fontsize=15)
    if plotLegend:
        plt.legend(fontsize=fontsize * 0.6)
    if plotShow:
        plt.show()


def plotFitnessXsForAll(figsize=(10, 3.3), alpha=0.3, fontsize=15, linewidth=2, plotFigure=True, plotShow=True, normalize=True, plotMean=True, title='1 + fitness_xs'):
    if plotFigure:
        plt.figure(figsize=figsize)
    for pop in populations_doable:
        fitnessXs = np.array(fitness_xs[pop])
        if normalize:
            fitnessXs -= fitnessXs[0]
        plt.plot(fitness_ts[pop], 1 + fitnessXs, marker='.', linewidth=linewidth, alpha=alpha, label=pop)
    if plotMean:
        meanXs = np.array(fitness_xs_mean_doable)
        if normalize:
            meanXs -= meanXs[0]
        plt.plot(fitness_ts[pop], 1 + meanXs, linewidth=linewidth * 2, alpha=alpha * 2, label='mean')
    normalized = ' (normalized)' if normalize else ''
    plt.title(title + normalized, fontsize=fontsize)
    plt.legend(fontsize=fontsize * 0.6)
    if plotShow:
        plt.show()


def plotFitnessWsForAll(tEnd=len(TIMES_INTPL), figsize=(10, 3.3), alpha=0.3, fontsize=15, linewidth=2, plotFigure=True, plotShow=True, normalize=True, plotMean=True, plotScatterMean=False, title='fitness_ws', plotMeanOnly=False, plotPowerModel=False):
    if plotFigure:
        plt.figure(figsize=figsize)
    if not plotMeanOnly:
        for pop in populations_doable:
            fitnessWs = np.array(fitness_ws[pop])
            if normalize:
                fitnessWs /= fitnessWs[0]
            plt.plot(fitness_ts[pop], fitnessWs, marker='.', linewidth=linewidth, alpha=alpha, label=pop)
    if plotMean or plotMeanOnly:
        meanWs = np.array(fitness_ws_mean_doable)
        if normalize:
            meanWs /= meanWs[0]
        if plotScatterMean:
            plt.scatter(fitness_ts_mean_doable, meanWs, alpha=alpha * 2, label='mean', color='black')
        else:
            plt.plot(fitness_ts_mean_doable, meanWs, linewidth=linewidth * 2, alpha=alpha * 2, label='mean', color='black')
    if plotPowerModel:
        ts = np.arange(0, 50000, 1)
        a, b = 1 / 12, 0.01
        for b in np.arange(0.001, 0.02, 0.001):
            fs = (1 + b * ts) ** a
            plt.plot(ts, fs, label='b=%.4f'%b)
    normalized = ' (normalized)' if normalize else ''
    plt.xlim(-2000, TIMES_INTPL[tEnd - 1])
    plt.ylim(0.95, 2.15)
    plt.title(title + normalized, fontsize=fontsize)
    plt.legend(fontsize=fontsize * 0.6)
    if plotShow:
        plt.show()


def plotFitnessXsAndWs(pop, tEnd=14, figsize=(10, 3.3), alpha=0.3, fontsize=15, plotFigure=True, plotShow=True, normalize=True):
    if plotFigure:
        plt.figure(figsize=figsize)
    earlyIndices = [i for i, t in enumerate(fitness_ts[pop]) if t < TIMES_INTPL[tEnd]]
    fitnessTimes = fitness_ts[pop][earlyIndices]
    fitnessWs = np.array(fitness_ws[pop][earlyIndices])
    fitnessXs = np.array(fitness_xs[pop][earlyIndices])
    if normalize:
        fitnessWs /= fitnessWs[0]
        fitnessXs -= fitnessXs[0]
    a_list = np.arange(0.1, 2.1, 0.1)
    i = np.argmin([RC.MAE(1 + (fitnessXs / a), fitnessWs) for a in a_list])
    a = a_list[i]
    normalized = ' (normalized)' if normalize else ''
    plt.plot(fitnessTimes, fitnessWs, marker='.', color='blue', linewidth=5, alpha=alpha, label='fitness_ws' + normalized)
    plt.plot(fitnessTimes, 1 + fitnessXs, marker='.', color='red', linewidth=5, alpha=alpha, label='1 + fitness_xs' + normalized)
    plt.plot(fitnessTimes, 1 + (fitnessXs / a), marker='.', color='green', linewidth=5, alpha=alpha, label=f'1 + fitness_xs / {a}' + normalized)
    plt.title(f"pop {pop}, fitness trajectories", fontsize=15)
    plt.legend(fontsize=fontsize * 0.6)
    if plotShow:
        plt.show()

def plotDataForEarlyFixedMutations(pop, tEnd=14, thFixed=0.9, figsize=(10, 3.3), fontsize=15, alpha=0.3, plotFigure=True, plotShow=True, plotFitness=True, plotMeanFitnessNonmutator=False, annot=False, plotAllMuts=False):
    if plotFigure:
        plt.figure(figsize=figsize)
    plotted = set()
    count = 0
    if plotAllMuts:
        states = list(np.arange(0, 8))
    else:
        states = [ANCESTRAL_FIXED, EXTINCT]
    if annot:
        # Avoid annotation overlapping
        ancestor_sites = [site for l, site in enumerate(data[pop]['sites_intpl']) if data[pop]['site_to_clade'][site] in states and data[pop]['freqs_intpl'][l][tEnd - 1] > thFixed]
        bucketToSite = {}
        bucketWidth = (TIMES_INTPL[tEnd] - TIMES_INTPL[0]) / (4 * len(ancestor_sites))
    for l, site in enumerate(data[pop]['sites_intpl']):
        c = data[pop]['site_to_clade'][site]
        if c in states and np.max(data[pop]['freqs_intpl'][l][:tEnd]) >= thFixed:
            count += data[pop]['freqs_intpl'][l][tEnd - 1] > max(0.9, thFixed)
            if c not in plotted:
                plt.plot(TIMES_INTPL[:tEnd], data[pop]['freqs_intpl'][l][:tEnd], alpha=alpha, color=COLORS[c], label=LABELS[c])
                plotted.add(c)
            else:
                plt.plot(TIMES_INTPL[:tEnd], data[pop]['freqs_intpl'][l][:tEnd], alpha=alpha, color=COLORS[c])
            earlyIndices = [i for i, t in enumerate(data[pop]['times'][l]) if t < TIMES_INTPL[tEnd]]

            # plt.scatter(data[pop]['times'][l][earlyIndices], data[pop]['freqs'][l][earlyIndices], alpha=0.3, color=COLORS[c])
            l_ = data[pop]['sites'].index(site)
            plt.scatter(data[pop]['times'][l_][earlyIndices], [count / depth for count, depth in zip(data[pop]['counts'][l_][earlyIndices], data[pop]['depths'][l_][earlyIndices])], alpha=0.3, color=COLORS[c])

            if annot:
                t = np.argmax(data[pop]['freqs_intpl'][l][:tEnd])
                bucketIndex = int(TIMES_INTPL[t] / bucketWidth)
                while bucketIndex in bucketToSite:
                    bucketIndex += 1
                time = bucketIndex * bucketWidth
                t = np.argmin(np.absolute(TIMES_INTPL - time))
                plt.text(TIMES_INTPL[t] + np.random.uniform(-200, 200), data[pop]['freqs_intpl'][l][t] + np.random.uniform(-0.15, 0.15), f'{l}', fontsize=fontsize * 0.6, color=COLORS[c])
                bucketToSite[bucketIndex] = site
    if plotFitness:
        earlyIndices = [i for i, t in enumerate(fitness_ts[pop]) if t < TIMES_INTPL[tEnd]]
        fitnessTimes = fitness_ts[pop][earlyIndices]
        fitnessWs = fitness_ws[pop][earlyIndices]
        plt.plot(fitnessTimes, fitnessWs, marker='.', color='blue', linewidth=5, alpha=alpha, label='Fitness')
        if annot:
            for i, t in enumerate(fitnessTimes):
                plt.text(t, fitnessWs[i] + 0.05, '%.2f'%fitnessWs[i], fontsize=fontsize * 0.6, color='blue')
    if plotMeanFitnessNonmutator:
        earlyIndices = [i for i, t in enumerate(fitness_ts[pop]) if t < TIMES_INTPL[tEnd]]
        fitnessTimes = fitness_ts_mean_nonmutator[earlyIndices]
        fitnessWs = fitness_ws_mean_nonmutator[earlyIndices]
        plt.plot(fitnessTimes, fitnessWs, marker='.', color='orange', linewidth=5, linestyle='dashed', alpha=alpha, label='Mean fitness\nfor all nonmutators')
    plt.title(f"pop {pop}, {count} mutations fixed by generation {TIMES_INTPL[tEnd]}", fontsize=15)
    plt.legend(fontsize=fontsize * 0.6)

    plt.xlim(-200, TIMES_INTPL[tEnd] + min(500, 0.2 * TIMES_INTPL[tEnd]))
    plt.xticks(ticks=TIMES_INTPL[:tEnd], labels=TIMES_INTPL[:tEnd])

    if plotShow:
        plt.show()


def plotDataForMutations(pop, l_intpl_list, tEnd=14, figsize=(10, 3.3), fontsize=15, alpha=0.3, plotFigure=True, plotShow=True, annot=True):
    if plotFigure:
        plt.figure(figsize=figsize)
    xlim = (-200, TIMES_INTPL[tEnd] + min(500, 0.2 * TIMES_INTPL[tEnd]))
    for l_intpl in l_intpl_list:
        site = data[pop]['sites_intpl'][l_intpl]
        c = data[pop]['site_to_clade'][site]
        plt.plot(TIMES_INTPL[:tEnd], data[pop]['freqs_intpl'][l_intpl][:tEnd], alpha=alpha, color=COLORS[c])
        earlyIndices = [i for i, t in enumerate(data[pop]['times'][l_intpl]) if t < TIMES_INTPL[tEnd]]

        l_non_intpl = data[pop]['sites'].index(site)
        plt.scatter(data[pop]['times'][l_non_intpl][earlyIndices], data[pop]['freqs'][l_non_intpl][earlyIndices], alpha=0.3, color=COLORS[c])
        if annot:
            t = np.argmax(data[pop]['freqs_intpl'][l_intpl][:tEnd])
            plt.text(TIMES_INTPL[t] + np.random.uniform(-100, 100), data[pop]['freqs_intpl'][l_intpl][t] + np.random.uniform(-0.15, 0.15), f'{l_intpl}', fontsize=fontsize * 0.6, color=COLORS[c])
    plt.plot([xlim[0], xlim[1]], [1, 1], color='blue', linestyle='dashed', linewidth=3, alpha=0.1)
    plt.title(f"pop {pop}, mutations {l_intpl_list}", fontsize=15)
    # plt.legend(fontsize=fontsize * 0.6)
    plt.xlim(xlim)
    plt.xticks(ticks=TIMES_INTPL[:tEnd], labels=TIMES_INTPL[:tEnd])
    if plotShow:
        plt.show()


def plotResForOnePop(res, fontsize=15, alpha=0.3):

    alpha_ = alpha if len(res.traj[0]) <= 1000 else 0.1
    AP.plotTotalCladeFreq(res.cladeFreqWithAncestor, cladeMuts=res.cladeMuts,
                          traj=res.traj, alpha=alpha_, plotFigure=True, plotShow=True)
    AP.plotMullerCladeFreq(res.mullerCladeFreq, res.mullerColors, res.times)


def plotDataAndRes(pops, reconstructions, fontsize=15, alpha=0.3):
    fig, axes = plt.subplots(len(pops), 2, figsize=(12, 4 * len(pops)))
    for i, pop in enumerate(pops):
        res = reconstructions[pop]
        alpha_ = alpha if len(data[pop]['sites_intpl']) <= 1000 else 0.1
        plt.sca(axes[i, 0])
        AP.plotTotalCladeFreq(res.cladeFreqWithAncestor, cladeMuts=res.cladeMuts,
                              traj=res.traj, alpha=alpha_, plotFigure=False, plotShow=False)

        plt.sca(axes[i, 1])
        plotDataForOnePop(data[pop], pop, fontsize=fontsize, alpha=alpha_, plotShow=False)
    plt.show()
