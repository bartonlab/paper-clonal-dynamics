#!/usr/bin/env python
# Functions related to MPL inference

import numpy as np                          # numerical tools
from copy import deepcopy


def processStandard(sequences, frequencies, times, q, totalCov, covAtEachTime=False, intCovTimes=[]):
    """Integrates covariance from time-series sequence and count data.

    Args:
        sequences: A list of lists of sequences; A time-series of Sequences present in the population.
        frequencies: A list of lists of sequence frequencies corresponding to sequences.
        times: A list of times corresponding to sequences.
        q: Number of states at each locus.
        totalCov: The covariance integrate to be updated.
        covAtEachTime: Optional; Whether or not return covariance at each time point.
        intCovTimes: Optional; A SORTED list of time points at which we store integrated covariance.

    Returns:
        cov: Optional; covariance at each time point.
        intCovAtTimes: Optional; A list of integrated covariance at time points specified in intCovTimes.
    """
    L = len(totalCov)
    p1 = np.zeros(L, dtype=float)
    p2 = np.zeros((L, L), dtype=float)
    lastp1 = np.zeros(L, dtype=float)
    lastp2 = np.zeros((L, L), dtype=float)
    # store the total_cov at each time point when dynamics is enabled.
    if covAtEachTime:
        cov = np.zeros((len(sequences), L, L), dtype=float)
    elif intCovTimes:
        intCovAtTimes = [np.zeros((L, L), dtype=float) for _ in intCovTimes]

    computeAllelFrequencies(sequences[0], frequencies[0], q, lastp1, lastp2)
    if covAtEachTime:
        cov[0] = computeCovariances(lastp1, lastp2)

    covTimeIndex = 0
    for k in range(1, len(sequences)):
        computeAllelFrequencies(sequences[k], frequencies[k], q, p1, p2)
        updateCovarianceIntegrate(times[k] - times[k-1], lastp1, lastp2, p1, p2, totalCov)
        if covAtEachTime:
            cov[k] = computeCovariances(p1, p2)
        elif intCovTimes and times[k] <= intCovTimes[covTimeIndex]:
            if k == len(sequences) - 1:
                # There is no next time point, but the projected next time point will pass intCovTimes[covTimeIndex]
                while covTimeIndex < len(intCovTimes) and 2 * times[k] - times[k - 1] > intCovTimes[covTimeIndex]:
                    intCovAtTimes[covTimeIndex] = deepcopy(totalCov)
                    covTimeIndex += 1
            elif times[k + 1] > intCovTimes[covTimeIndex]:
                # Next time point will pass intCovTimes[covTimeIndex]
                while times[k + 1] > intCovTimes[covTimeIndex]:
                    intCovAtTimes[covTimeIndex] = deepcopy(totalCov)
                    covTimeIndex += 1
        if not k == len(sequences) - 1:
            lastp1[:] = p1
            lastp2[:] = p2

    if covAtEachTime:
        return cov
    elif intCovTimes:
        return intCovAtTimes


def computeAllelFrequencies(sequences, frequencies, q, p1, p2):
    """Computes allele frequencies and pair-allele frequencies, recording only (q - 1) states.

    Args:
        sequences: A list of sequences present at a time point.
        frequencies: A list of sequence frequencies corresponding to sequences.
        q: Number of states at each locus.
        p1: A Numpy array of allele frequencies to be computed.
        p2: A Numpy array of pair-allele frequencies to be computed.
    """
    p1.fill(0)
    p2.fill(0)

    for k in range(len(sequences)):
        for i in range(len(sequences[k])):
            if not sequences[k][i] == 0:
                a = int(i * (q - 1) + sequences[k][i] - 1)
                p1[a] += frequencies[k]

                for j in range(i + 1, len(sequences[k])):
                    if not sequences[k][j] == 0:
                        b = int(j * (q - 1) + sequences[k][j] - 1)
                        p2[a, b] += frequencies[k]
                        p2[b, a] += frequencies[k]


def computeCovariances(p1, p2):
    """Computes covariances from allele & pair-allele frequencies."""

    L = len(p1)
    cov_tmp = np.zeros((L, L), dtype=float)
    for i in range(L):
        cov_tmp[i, i] = p1[i]
        for j in range(i + 1, L):
            cov_tmp[i, j] = p2[i, j] - p1[i] * p1[j]
            cov_tmp[j, i] = cov_tmp[i, j]
    return cov_tmp


def updateCovarianceIntegrate(dg, p1_0, p2_0, p1_1, p2_1, totalCov):
    """Interpolates allele frequencies and adds integrated covariance in a time interval to the integrate.

    Args:
        dg: Width of the time interval.
        p1_0: A Numpy array of allele frequencies at the beginning of the time interval.
        p2_0: A Numpy array of pair-allele frequencies at the beginning of the time interval.
        p1_1: A Numpy array of allele frequencies at the end of the time interval.
        p2_1: A Numpy array of pair-allele frequencies at the end of the time interval.
        totalCov: The covariance integrate to be updated.
    """
    N = len(p1_0)

    for a in range(N):
        totalCov[a, a] += integratedVariance(p1_0[a], p1_1[a], dg)

        for b in range(a + 1, N):
            totalCov[a, b] += integratedCovariance(p1_0[a], p1_0[b], p2_0[a, b], p1_1[a], p1_1[b], p2_1[a, b], dg)
            totalCov[b, a] = totalCov[a, b]


def computeD(freqs, times, mu):
    """Computes the 'Dx' term in MPL inference."""

    T = len(times)
    tStart, tEnd = 0, T - 1
    D = freqs[tEnd] - freqs[tStart] - mu * np.sum([(times[t + 1] - times[t]) * (1 - 2 * freqs[t]) for t in range(tStart, tEnd)], axis=0)
    return D


def computeDWithEpistasis(traj_pairs, times, mu, tStart=0, tEnd=None):

    T, L, L = traj_pairs.shape
    if tEnd is None:
        tEnd = T - 1
    num_pairs = L * (L + 1) // 2
    D = np.zeros(num_pairs, dtype=float)
    for index in range(num_pairs):
        i, j = getPairSitesByIndex(index, L)
        D[index] = traj_pairs[tEnd, i, j] - traj_pairs[tStart, i, j]
        if i == j:
            D[index] -= mu * np.sum([(times[t + 1] - times[t]) * (1 - 2 * traj_pairs[t, i, i]) for t in range(tStart, tEnd)])
        else:
            D[index] -= mu * np.sum([(traj_pairs[t, i, i] + traj_pairs[t, j, j] - 4 * traj_pairs[t, i, j]) for t in range(tStart, tEnd)])
    return D


def getPairSitesByIndex(index, L):
    """
    0, 1, 2, ..., L-1, (0, 1), (0, 2), (0, 3), ..., (0, L-1), (1, 2), (1, 3), ..., (1, L-1), ...
    index >= L, from index, calculate (i, j), where i < j
    """
    if index < L:
        return index, index
    max_index_with_i = L - 1
    for i in range(L-1):
        max_index_with_i += L - i - 1
        if index <= max_index_with_i:
            break
    j = index - L * (i + 1) + (i + 1) * (i + 2) // 2

    return i, j


def inferSelection(int_cov, D, regularization_matrix):
    """Infers selection coefficients.

    Args:
        int_cov: Integrated covariance matrix.
        D: 'Dx' term in MPL inference.
        regularization_matrix: A matrix added to integrated covariance matrix for regularization.
    """
    return np.dot(np.linalg.inv(int_cov + regularization_matrix), D)


def integratedVariance(xi_0, xi_1, dg):
    """
    Calculates integrated variance between two time points, where allele frequencies are linearly interpolated in between.
    """
    return dg * ( ((3 - 2 * xi_1) * (xi_0 + xi_1)) - 2 * (xi_0 * xi_0) ) / 6


def integratedCovariance(xi_0, xj_0, xij_0, xi_1, xj_1, xij_1, dg):
    """
    Calculates integrated covariance between two time points, where allele frequencies are linearly interpolated in between.
    """
    intCov1 = -dg * ((2 * xi_0 * xj_0) + (2 * xi_1 * xj_1) + (xi_0 * xj_1) + (xi_1 * xj_0)) / 6
    intCov2 = dg * 0.5 * (xij_0 + xij_1)
    return intCov1 + intCov2
