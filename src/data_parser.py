#!/usr/bin/env python
# Parse time-series genetic data from multiple papers/sources

import sys
import argparse
import numpy as np                          # numerical tools
from timeit import default_timer as timer   # timer for performance
from scipy import stats
from scipy import interpolate
import math
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
import MPL                                  # MPL inference tools
import simulation_helper as SH
import estimate_covariance as EC
# sys.path.append('../../evoracle/')  # path to Evoracle
# import evoracle

# haploSep
HAPLOSEP_INPUT_DIR = './data/haploSep/input'
HAPLOSEP_OUTPUT_DIR = './data/haploSep/output'

SIMULATION_PARAMS = {
    'mutation_rate': 0,
    'linear_reg': 1,
}


############################################
#
# Utility functions
#
############################################

def create_interpolation_function(times, freqs, tmax=100000, kind='linear'):
    # can create it for anything!

    interpolating_function = interpolate.interp1d(times, freqs, kind=kind, bounds_error=True)

    # padded_times = np.zeros(len(times)+1)
    # padded_freqs = np.zeros(len(times)+1)
    # padded_times[0:len(times)] = times
    # padded_freqs[0:len(times)] = freqs
    # padded_times[-1] = tmax
    # padded_freqs[-1] = freqs[-1]
    #
    # interpolating_function = interpolate.interp1d(padded_times, padded_freqs, kind=kind, bounds_error=True)

    return interpolating_function


def sort_msa_by_times(msa, tag, sort=False, TIME_INDEX=3, TF_TAG='B.FR.1983.HXB2-LAI-IIIB-BRU.K03455.19535', CONS_TAG='CONSENSUS'):
    """Return sequences and times collected from an input MSA and tags (optional: time order them)."""

    times = []
    for i in range(len(tag)):
        if tag[i] not in [TF_TAG, CONS_TAG]:
            tsplit = tag[i].split('.')
            times.append(int(tsplit[TIME_INDEX]))
        else:
            times.append(-1)

    if sort:
        t_sort = np.argsort(times)
        return np.array(msa)[t_sort], np.array(tag)[t_sort], np.array(times)[t_sort]
    else:
        return np.array(times)


def get_MSA(ref, noArrow=True):
    """Take an input FASTA file and return the multiple sequence alignment, along with corresponding tags. """

    temp_msa = [i.split() for i in open(ref).readlines()]
    temp_msa = [i for i in temp_msa if len(i) > 0]

    msa = []
    tag = []

    for i in temp_msa:
        if i[0][0] == '>':
            msa.append('')
            if noArrow:
                tag.append(i[0][1:])
            else:
                tag.append(i[0])
        else:
            msa[-1] += i[0]

    msa = np.array(msa)

    return msa, tag


def MAE(a, b):
    return np.mean(np.absolute(np.array(a) - np.array(b)))


def spearmanr_pearsonr_MAE(a, b):
    a, b = np.array(a), np.array(b)
    if len(a.shape) > 1:
        a, b = np.ndarray.flatten(a), np.ndarray.flatten(b)
    return stats.spearmanr(a, b)[0], stats.pearsonr(a, b)[0], MAE(a, b)


def binarize_genotype(genotype, wild_type='.', ref_seq=None):
    seq = np.zeros(len(genotype), dtype=int)
    if ref_seq is None:
        for i, locus in enumerate(genotype):
            if locus != wild_type:
                seq[i] = 1
    else:
        for i, locus in enumerate(genotype):
            if locus != ref_seq[i]:
                seq[i] = 1
    return seq


def compute_allele_traj_from_geno_traj(geno_traj, genotypes):
    L = len(genotypes[0])
    T = len(geno_traj)
    allele_traj = np.zeros((T, L))
    for t in range(T):
        for k, genotype in enumerate(genotypes):
            for i, locus in enumerate(genotype):
                allele_traj[t, i] += geno_traj[t, k] * locus
    return allele_traj


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


def copy_diagonal_terms_from(matrix):

    ret = np.zeros_like(matrix)
    for l in range(len(matrix)):
        ret[l, l] = matrix[l, l]
    return ret


def array_to_string(seq):
    return ''.join([str(_) for _ in seq])


############################################
#
# HPC Cluster
#
############################################


def scp_to_cluster(local_file, cluster_dir, directory=False):
    if directory == True:
        print(f'scp -r {local_file} yli354@cluster.hpcc.ucr.edu:{cluster_dir}')
    else:
        print(f'scp {local_file} yli354@cluster.hpcc.ucr.edu:{cluster_dir}')


def scp_from_cluster(local_dir, cluster_file, directory=False):
    if directory == True:
        print(f'scp -r yli354@cluster.hpcc.ucr.edu:{cluster_file} {local_dir}')
    else:
        print(f'scp yli354@cluster.hpcc.ucr.edu:{cluster_file} {local_dir}')


############################################
#
# Evoracle
#
############################################

def save_traj_for_evoracle(traj, file, times=None):
    T, L = traj.shape
    if times is None:
        times = np.arange(0, T)
    dic = {t: [] for t in times}
    dic['Symbols and linkage group index'] = []
    for j in range(len(traj[0])):
        dic['Symbols and linkage group index'].append(f'M {j}')  # Can use any character other than '.'
        dic['Symbols and linkage group index'].append(f'. {j}')
        for i, t in enumerate(times):
            key = t
            dic[key].append(traj[i, j])
            dic[key].append(1 - traj[i, j])

    df = pd.DataFrame(dic)
    df.to_csv(file, index=False)


def parse_obs_reads_df(file):
    df = pd.read_csv(f'{file}')
    return df


def parse_evoracle_results(obs_reads_file, output_directory, times=None, params=SIMULATION_PARAMS, save_geno_traj=True, compute_genotype_fitness=False, linear_reg=None):
    mu = params['mutation_rate']
    if linear_reg is None:
        linear_reg = params['linear_reg']

    df_traj = pd.read_csv(f'{output_directory}/{obs_reads_file}')
    traj = parse_evoracle_traj(df_traj)
    T, L = traj.shape
    if times is None:
        times = np.arange(0, T)

    df_geno = pd.read_csv(f'{output_directory}/_final_genotype_matrix.csv')
    binary_genotypes, geno_traj = parse_evoracle_geno_traj(df_geno)

    traj_from_geno_traj = compute_allele_traj_from_geno_traj(geno_traj, binary_genotypes)

    int_cov = MPL.integrateCovarianceFromStableGenotypes(binary_genotypes, geno_traj, times)
    D = MPL.computeD(traj, times, mu)
    D_from_geno_traj = MPL.computeD(traj_from_geno_traj, times, mu)
    selection = MPL.inferSelection(int_cov, D, linear_reg * np.identity(L))
    selection_from_geno_traj = MPL.inferSelection(int_cov, D_from_geno_traj, linear_reg * np.identity(L))

    results = {
        'traj': traj,
        'genotypes': binary_genotypes,
        'traj_from_geno_traj': traj_from_geno_traj,
        'int_cov': int_cov,
        'selection': selection,
        'selection_from_geno_traj': selection_from_geno_traj,
    }
    if save_geno_traj:
        results['geno_traj'] = geno_traj
    if compute_genotype_fitness:
        fitness = MPL.computeGenotypeFitnessFromSelection(binary_genotypes, selection)
        fitness_from_geno_traj = MPL.computeGenotypeFitnessFromSelection(binary_genotypes, selection_from_geno_traj)
        results['fitness'] = fitness
        results['fitness_from_geno_traj'] = fitness_from_geno_traj
    return results


def save_evoracle_results(results, file, complete=False):

    if complete:
        np.savez_compressed(file, **results)
    else:
        results_compact = {key: results[key] for key in ['traj', 'int_cov', 'selection']}
        np.savez_compressed(file, **results_compact)


def parse_evoracle_traj(df_traj):
    return df_traj.to_numpy()[0::2, :-1].T.astype(float)


def parse_evoracle_geno_traj(df_geno):

    K = len(df_geno)
    time_columns = list(df_geno.columns)[1:]
    T = len(time_columns)
    genotypes = list(df_geno['Unnamed: 0'])
    binary_genotypes = [binarize_genotype(genotype) for genotype in genotypes]
    geno_traj = np.zeros((T, K))
    for k, row in df_geno.iterrows():
        freqs = [row[_] for _ in time_columns]
        geno_traj[:, k] = freqs
    return binary_genotypes, geno_traj


def parse_performances_on_covariances_selections_all_methods(selections, covariances, performances_evoracle, performances_haplosep, p=0, q=0, selection_from_geno_traj=False):

    covariances_true, covariances_SL, covariances_est, covariances_est_unnormalized, covariances_linear, covariances_nonlinear, covariances_shrink, covariances_nonlinear_shrink = covariances['covariances_true'], covariances['covariances_SL'], covariances['covariances_est'], covariances['covariances_est_unnormalized'], covariances['covariances_linear'], covariances['covariances_nonlinear'], covariances['covariances_shrink'], covariances['covariances_nonlinear_shrink']

    covariances_evoracle = [[performances_evoracle[s][n]['int_cov'] for n in range(NUM_TRIALS)]
                            for s in range(NUM_SELECTIONS)]
    covariances_haplosep = [[performances_haplosep[s][n]['int_cov'] for n in range(NUM_TRIALS)]
                            for s in range(NUM_SELECTIONS)]

    selections_true = SS.load_selections()
    selections_MPL, selections_SL, selections_est, selections_est_unnormalzied, selections_linear, selections_nonlinear, selections_shrink, selections_nonlinear_shrink = selections['selections_basic'][:, :, p, q, 1], selections['selections_basic'][:, :, p, q, 0], selections['selections_basic'][:, :, p, q, 2], selections['selections_basic'][:, :, p, q, 3], selections['selections_linear'][:, :, p, q], selections['selections_dcorr'][:, :, p, q], selections['selections_shrink'][:, :, p, q], selections['selections_dcorr_shrink'][:, :, p, q]

    if selection_from_geno_traj:
        selections_evoracle = [[performances_evoracle[s][n]['selection_from_geno_traj'] for n in range(NUM_TRIALS)] for s in range(NUM_SELECTIONS)]
        selections_haplosep = [[performances_haplosep[s][n]['selection_from_geno_traj'] for n in range(NUM_TRIALS)] for s in range(NUM_SELECTIONS)]
    else:
        selections_evoracle = [[performances_evoracle[s][n]['selection'] for n in range(NUM_TRIALS)]
                                for s in range(NUM_SELECTIONS)]
        selections_haplosep = [[performances_haplosep[s][n]['selection'] for n in range(NUM_TRIALS)]
                                for s in range(NUM_SELECTIONS)]

    covariances_list = [
        covariances_true, covariances_SL, covariances_est, covariances_est_unnormalized,
        covariances_linear, covariances_nonlinear,
        # covariances_shrink, covariances_nonlinear_shrink,
        np.array(covariances_evoracle), np.array(covariances_haplosep)
        ]

    selections_list = [
        selections_MPL, selections_SL, selections_est, selections_est_unnormalzied,
        selections_linear, selections_nonlinear,
        # selections_shrink, selections_nonlinear_shrink,
        np.array(selections_evoracle), np.array(selections_haplosep)
        ]

    MAE_cov, spearmanr_cov, MAE_selection, spearmanr_selection = [], [], [], []
    method_list = [
        'MPL', 'SL', 'Est', 'Est (Unnormalized)',
        'Linear', 'Nonlinear',
        # 'Est-shrink', 'Nonlinear-shrink',
        'Evoracle', 'haploSep']

    # print(covariances_true.shape, selections_true.shape)

    for i, (cov, sel) in enumerate(zip(covariances_list, selections_list)):
        # print(method_list[i], cov.shape, sel.shape)
        MAE_cov.append([])
        spearmanr_cov.append([])
        MAE_selection.append([])
        spearmanr_selection.append([])
        for s in range(NUM_SELECTIONS):
            for n in range(NUM_TRIALS):
                spearmanr, pearsonr, MAE = spearmanr_pearsonr_MAE(EC.get_off_diagonal_terms(covariances_true[s, n]), EC.get_off_diagonal_terms(cov[s][n]))
                MAE_cov[i].append(MAE)
                spearmanr_cov[i].append(spearmanr)
                spearmanr, pearsonr, MAE = spearmanr_pearsonr_MAE(selections_true[s], sel[s][n])
                MAE_selection[i].append(MAE)
                spearmanr_selection[i].append(spearmanr)

    return MAE_cov, spearmanr_cov, MAE_selection, spearmanr_selection
