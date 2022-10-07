import sys
import numpy as np
import pandas as pd
import math
from scipy import stats
from scipy import interpolate
import matplotlib.pyplot as plt

import MPL


def inferencePipeline(cladeFreq, cladeCompo, dg=1, times=None, mu=0, reg_list=[1, 5, 10, 50], interpolate=True):
    traj = getSampledTraj(cladeFreq, cladeCompo)
    cov_noCorr = getIntCovFromFAndMu(traj, cladeFreq, cladeCompo, dg=dg, times=times, interpolate=interpolate)
    cov_maxCorr = getIntCovFromFAndMu(traj, cladeFreq, cladeCompo, dg=dg, times=times, maxCorrelation=True, interpolate=interpolate)
    fields_noCorr, fitness_noCorr = getFieldsAndFitness(traj, cov_noCorr, dg, mu, reg_list, times=times)
    fields_maxCorr, fitness_maxCorr = getFieldsAndFitness(traj, cov_maxCorr, dg, mu, reg_list, times=times)
    results = {'cladeFreq': cladeFreq, 'cladeCompo': cladeCompo, 'traj': traj, 'cov_noCorr': cov_noCorr, 'cov_maxCorr': cov_maxCorr, 'fields_noCorr': fields_noCorr, 'fields_maxCorr': fields_maxCorr, 'fitness_noCorr': fitness_noCorr, 'fitness_maxCorr': fitness_maxCorr}

    return results


def getSampledTraj(cladeFreq, cladeCompo):
    L, T, K = mu.shape
    traj = np.zeros((T, L), dtype = float)
    for t in range(T):
        for l in range(L):
            for k in range(K):
                traj[t, l] += cladeFreq[t, k] * cladeCompo[l, t, k]
    return traj


def getIntCovFromFAndMu(traj_sampled, cladeFreq, cladeCompo, dg=500, tStart=None, tEnd=None, times=None,
                        threshold_f=-1, maxCorrelation=False, interpolate=True,
                        masked_times=[], masked_sites=[]):
    L, T, K = mu.shape
    int_cov = np.zeros((L, L), dtype = float)
    if tStart is None:
        tStart = 0
    if tEnd is None:
        tEnd = T
    if interpolate:
        t = tStart
        next_x = np.zeros((L, L), dtype = float)
        cur_x = np.zeros((L, L), dtype = float)
        for i in range(L):
            for j in range(i + 1, L):
                next_x[i, j] = 0
                for k in range(K):
                    if cladeFreq[t, k] > threshold_f:
                        if maxCorrelation:
                            next_x[i, j] += cladeFreq[t, k] * min(cladeCompo[i, t, k], cladeCompo[j, t, k])
                        else:
                            next_x[i, j] += cladeFreq[t, k] * cladeCompo[i, t, k] * cladeCompo[j, t, k]

        for t in range(tStart, tEnd - 1):
            if times is not None:
                dg = times[t + 1] - times[t]
            for i in range(L):
                if t not in masked_times or i not in masked_sites:
                    int_cov[i, i] += integratedVariance(traj_sampled[t, i], traj_sampled[t + 1, i], dg)
                for j in range(i + 1, L):
                    # get cur_x[i, j]
                    cur_x[i, j] = next_x[i, j]
                    # get next_x[i, j]
                    next_x[i, j] = 0
                    for k in range(K):
                        if cladeFreq[t + 1, k] > threshold_f:
                            if maxCorrelation:
                                next_x[i, j] += cladeFreq[t + 1, k] * min(cladeCompo[i, t + 1, k], cladeCompo[j, t + 1, k])
                            else:
                                next_x[i, j] += cladeFreq[t + 1, k] * cladeCompo[i, t + 1, k] * cladeCompo[j, t + 1, k]
                    # get integrated cij in [t, t + 1]
                    if t not in masked_times or i not in masked_sites or j not in masked_sites:
                        int_cov[i, j] += integratedCovariance(traj_sampled[t, i], traj_sampled[t, j], cur_x[i, j],
                                                              traj_sampled[t + 1, i], traj_sampled[t + 1, j], next_x[i, j], dg)
    else:
        for t in range(tStart, tEnd):
            if times is not None:
                if t == 0:
                    dg = times[1] - times[0]
                elif t == len(times) - 1:
                    dg = times[t] - times[t - 1]
                else:
                    dg = (times[t + 1] - times[t - 1]) / 2
            for i in range(L):
                if t not in masked_times or i not in masked_sites:
                    int_cov[i, i] += dg * (traj_sampled[t, i] * (1 - traj_sampled[t, i]))
                for j in range(i + 1, L):
                    xij = 0
                    for k in range(K):
                        if cladeFreq[t, k] > threshold_f:
                            if maxCorrelation:
                                xij += cladeFreq[t, k] * min(cladeCompo[i, t, k], cladeCompo[j, t, k])
                            else:
                                xij += cladeFreq[t, k] * cladeCompo[i, t, k] * cladeCompo[j, t, k]
                    if t not in masked_times or i not in masked_sites or j not in masked_sites:
                        int_cov[i, j] += dg * (xij - traj_sampled[t, i] * traj_sampled[t, j])

    for i in range(L):
        for j in range(i + 1, L):
            int_cov[j, i] = int_cov[i, j]

    return int_cov


def getFieldsAndFitness(traj, cov, dg, mu, reg_list, times=None):

    fields, fitness = [], []
    getFieldsWithRegList(fields, cov, traj, dg=dg, times=times, reg_list=reg_list, mu=mu)
    getFitnessWithFieldsList(fitness, fields, traj)
    return fields, fitness


def getSelectionsAndFitnessWithEpistasis(x, cov, dg, mu, reg_list, times=None):

    selections, fitness = [], []
    getSelectionsWithRegListWithEpistasis(selections, cov, x, dg=dg, times=times, reg_list=reg_list, mu=mu)
    getFitnessWithSelectionsListWithEpistasis(fitness, selections, x)
    return selections, fitness


def getFieldsWithRegList(fields, cov_tmp, traj_tmp, reg_list=[0.1, 1, 10, 100], mu=1e-10, dg=500, times=None):
    cov_SL = np.zeros_like(cov_tmp)
    for l in range(len(cov_tmp)):
        cov_SL[l, l] = cov_tmp[l, l]
    fields.append(getFieldsFromCov(cov_SL, traj_tmp, regularization=1, mu=mu, dg=dg, times=times))
    for reg in reg_list:
        fields.append(getFieldsFromCov(cov_tmp, traj_tmp, regularization=reg, mu=mu, dg=dg, times=times))


def getFieldsFromCov(int_cov, traj_sampled, mu=1e-10, regularization=1, dg=500, times=None):

    T, L = traj_sampled.shape
    tStart, tEnd= 0, (T - 1)
    if times is None:
        times = dg * (np.arange(0, T))
    if len(times) != T:
        print("Error! length of times does not match length of trajectory.")
    D = MPL.computeD(traj_sampled, times, mu)
    IdentityMatrix = np.identity(L)
    fields = MPL.inferSelection(int_cov, D, regularization * IdentityMatrix)
    return fields


def getSelectionsWithRegListWithEpistasis(selections, cov_tmp, x_tmp, reg_list=[0.1, 1, 10, 100], mu=1e-10, dg=500, times=None):
    T, L, L = x_tmp.shape
    cov_SL = np.zeros((L, L), dtype=float)
    traj_tmp = np.zeros((T, L), dtype=float)
    for t in range(T):
        for l in range(L):
            traj_tmp[t, l] = x_tmp[t, l, l]
    for l in range(L):
        cov_SL[l, l] = cov_tmp[l, l]
    selections.append(getFieldsFromCov(cov_SL, traj_tmp, regularization=1, mu=mu, dg=dg, times=times))
    for reg in reg_list:
        selections.append(getSelectionsFromCovWithEpistasis(cov_tmp, x_tmp, regularization=reg, mu=mu, dg=dg, times=times))


def getSelectionsFromCovWithEpistasis(int_cov, x_double, mu=1e-10, regularization=1, dg=500, times=None):

    T, L, L = x_double.shape
    num_pairs = L * (L + 1) // 2
    if times is None:
        times = dg * (np.arange(0, T))
    if len(times) != T:
        print("Error! length of times does not match length of trajectory.")
    D = MPL.computeDWithEpistasis(x_double, times, mu)
    IdentityMatrix = np.identity(num_pairs)
    selectionVec = MPL.inferSelection(int_cov, D, regularization * IdentityMatrix)
    selection = np.zeros((L, L), dtype=float)
    for i in range(L):
        selection[i, i] = selectionVec[i]
    for index in range(L, num_pairs):
        i, j = MPL.getPairSitesByIndex(index, L)
        selection[i, j] = selectionVec[index]
        selection[j, i] = selection[i, j]
    return selection


def getFitnessWithFieldsList(fitness, fields_list, traj_tmp, reg_list = [0.1, 1, 10, 100]):
    for fields in fields_list:
        fitness.append(getPopulationFitness(fields, traj_tmp))


def getFitnessWithSelectionsListWithEpistasis(fitness, selections_list, x_tmp, reg_list = [0.1, 1, 10, 100]):
    T, L, L = x_tmp.shape
    traj_tmp = np.zeros((T, L), dtype=float)
    for t in range(T):
        for l in range(L):
            traj_tmp[t, l] = x_tmp[t, l, l]
    fitness.append(getPopulationFitness(selections_list[0], traj_tmp))
    for selections in selections_list[1:]:
        fitness.append(getPopulationFitnessWithEpistasis(selections, x_tmp))
