import sys
import numpy as np
import pandas as pd
import math
from copy import deepcopy
from scipy import stats
from scipy import interpolate
import matplotlib
import matplotlib.pyplot as plt
from tabulate import tabulate
from timeit import default_timer as timer   # timer for performance

import print_info as PRINT
import infer_fitness as INFER
import analyze_and_plot as PLOT
import estimate_covariance as EST
import LTEE
import MPL

EPSILON = 1e-10
PRECISION = 1e-16

EST_METHOD_NAME = 'LB'  # 'est_cov'
OUR_METHOD_NAME = 'dxdx'  # 'recovered'
TRUE_COV_NAME = 'True'  # 'true_cov'

class bcolors:
    BLACK = '\033[0m'
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    GREY = '\033[38;5;07m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    FAINT = '\033[2m'
    UNDERLINE = '\033[4m'


def MAE(a, b):
    return np.mean(np.absolute(np.array(a) - np.array(b)))


def Spearmanr(a, b):
    return stats.spearmanr(np.ndarray.flatten(a), np.ndarray.flatten(b))[0]


def Pearsonr(a, b):
    a, b = np.ndarray.flatten(a), np.ndarray.flatten(b)
    if len(a) < 2 or len(b) < 2:
        return 1
    return stats.pearsonr(a, b)[0]


def var(x):
    return x * (1 - x)


def topK(a, k):
    a = np.array(a)
    if k >= len(a):
        return a
    indices = np.argsort(-a)
    return a[indices[:k]]


def bottomK(a, k):
    a = np.array(a)
    if k >= len(a):
        return a
    indices = np.argsort(a)
    return a[indices[:k]]


def computeTimeIntervals(times):
    """Computes intervals in between each two adjacent time points."""

    T = len(times) - 1
    timeIntervals = np.zeros(T)
    for t in range(T):
        timeIntervals[t] = times[t + 1] - times[t]
    return timeIntervals


def computeDx(traj):
    """Computes change of frequency by next generation at each time point."""

    T = len(traj) - 1
    L = len(traj[0])
    dx = np.zeros((T, L), dtype=float)
    for t in range(T):
        dx[t] = traj[t + 1] - traj[t]
    return dx


def computeDxdx(traj):
    """Computes [dx_i * dx_j] matrix."""

    dx = computeDx(traj)
    T, L = dx.shape
    dxdx = np.zeros((T, L, L), dtype=float)
    for t in range(T):
        dxdx[t] = np.outer(dx[t], dx[t])
    return dxdx


def computeIntWeightedDxdx(reconstruction, start, end):
    dxdx = computeDxdx(reconstruction.traj)
    if reconstruction.weightByBothVariance:
        weightedDxdx = reconstruction.weightDxdxByBothVariance(dxdx)
    elif reconstruction.weightBySmallerVariance:
        weightedDxdx = reconstruction.weightDxdxBySmallerVariance(dxdx)
    elif reconstruction.weightBySmallerInterpolatedVariance:
        weightedDxdx = reconstruction.weightDxdxBySmallerInterpolatedVariance(dxdx)
    else:
        weightedDxdx = dxdx

    if len(reconstruction.times) > 1:
        intWeightedDxdx = np.zeros((reconstruction.L, reconstruction.L))
        timeIntervals = computeTimeIntervals(reconstruction.times)
        for t in range(start, min(end, len(weightedDxdx))):
            intWeightedDxdx += weightedDxdx[t] / timeIntervals[t]
        intWeightedDxdx *= np.mean(timeIntervals)
    else:
        intWeightedDxdx = np.sum(weightedDxdx, axis=0)

    for l in range(reconstruction.L):
        intWeightedDxdx[l, l] = 0

    return intWeightedDxdx


def computeIntegratedVariance(traj, times):
    T, L = traj.shape
    intVar = np.zeros((L, L), dtype=float)
    for t in range(0, T - 1):
        dg = times[t + 1] - times[t]
        for i in range(L):
            intVar[i, i] += MPL.integratedVariance(traj[t, i], traj[t + 1, i], dg)
    return intVar


def computeFitnessOfGenotypes(genotypes, selection):
    return np.array([1 + np.sum(genotype * selection) for genotype in genotypes])


def interpolateTraj(trajWithMissingValues, completeTimePoints):
    """
    Interpolates missing values in a trajectory, linearly according to given time points.
    """
    if len(trajWithMissingValues) != len(completeTimePoints):
        print('The number of time points must equal to the number of frequencies. ')
    for i in [0, -1]:
        if np.isnan(trajWithMissingValues[i]) or trajWithMissingValues[i] < 0 or trajWithMissingValues[i] < 0 > 1:
            print('Can not interpolate start or end time point.')
    for i, freq in enumerate(trajWithMissingValues):
        if np.isnan(freq):
            left = i - 1
            while np.isnan(trajWithMissingValues[left]):
                left -= 1
            right = i + 1
            while np.isnan(trajWithMissingValues[right]):
                right += 1
            trajWithMissingValues[i] = trajWithMissingValues[left] + (trajWithMissingValues[right] - trajWithMissingValues[left]) / (completeTimePoints[right] - completeTimePoints[left]) * (completeTimePoints[i] - completeTimePoints[left])


def segmentMatrix(matrix, groups):
    sites_sorted, groups_sorted = [], []
    for group in groups:
        group_sorted = sorted(group, key=lambda site: -np.sum(matrix[site, group]))
        groups_sorted.append(group_sorted)
        sites_sorted += group_sorted
    L = len(sites_sorted)
    seg = np.zeros((L, L))
    for i, s1 in enumerate(sites_sorted):
        seg[i, i] = matrix[s1, s1]
        for j in range(i + 1, L):
            s2 = sites_sorted[j]
            seg[i, j] = matrix[s1, s2]
            seg[j, i] = seg[i, j]
    return seg, groups_sorted


def undoSegmentMatrix(matrix, groups):
    i = 0
    i2s = {}
    for group in groups:
        for site in group:
            i2s[i] = site
            i += 1
    original = np.zeros_like(matrix)
    for i in range(len(matrix)):
        s1 = i2s[i]
        original[s1, s1] = matrix[i, i]
        for j in range(i + 1, len(matrix)):
            s2 = i2s[j]
            original[s1, s2] = matrix[i, j]
            original[s2, s1] = original[s1, s2]
    return original


def compute_next_x(x, s):
    return (1 + s) * x / (1 + s * x)


def compute_time_emerge_to_fix(population, s, initial_count=1, thFixed=0.99, thDetectable=0.01):
    x = initial_count / population
    xs = [x]
    time = 0
    time_detectable = None
    while x < thFixed:
        x = compute_next_x(x, s)
        xs.append(x)
        time += 1
        if time_detectable is None and x >= thDetectable:
            time_detectable = time
    return time, xs, time_detectable


def exponentialInterpolate(freqs, times, durationToFreqs, durationToSelection=None, inferSelection=False, plotCandidateInterpolations=True, plotLinearInterpolation=True, verbose=False):
    """
    Input:
    freqs: frequencies at each time. freqs[0]=0, freqs[-1]=1, all other freqs should be in (0, 1)
    times: times in unit of generation. len(times) >= 3
    Thoughts when writing this function:
    1. The population size determines mapping from time_taken_to_fixation to selection.
    2. An emergence time (when mutant count is 1), and a time_taken_to_fixation, determine the whole trajectory.
    """

    maxFixationTime = times[-1] - times[0]
    keys = list(durationToFreqs.keys())
    candidateDurations = np.array(maxFixationTime * np.arange(0.1, 1.1, 0.1)).astype(int)
    cmap = matplotlib.cm.get_cmap('coolwarm')
    colors = [cmap(_) for _ in np.arange(0.1, 1.1, 0.1)]
    if verbose:
        print(f'maximum fixation time: ', maxFixationTime)
        print(f'candidate durations: ', candidateDurations)
    candidateInterpolations = {}
    if verbose:
        plt.figure(figsize=(10, 3.3))
    for i, duration in enumerate(candidateDurations):
        closestKey, _ = closestValueAndIndex(list(durationToFreqs.keys()), duration)
        duration = closestKey
        closestFreqs = durationToFreqs[closestKey]
        f = LTEE.create_interpolation_function(list(np.arange(0, duration + 1)), closestFreqs)
        candidateEmergenceTimes = np.arange(times[0], times[1])
        candidateEmergenceTimeToMAE = {}
        for emergenceTime in candidateEmergenceTimes:
            candidateEmergenceTimeToMAE[emergenceTime] = MAE(freqs[1:-1], [f(time - emergenceTime) for time in times[1:-1]])
        if verbose:
            plt.plot(list(candidateEmergenceTimeToMAE.keys()), list(candidateEmergenceTimeToMAE.values()), alpha=0.5, label=f'duration={duration}', color=colors[i])
        _, emergenceTime = closestValueAndKey(candidateEmergenceTimeToMAE, 0)
        fixationTime = emergenceTime + duration
        freqs_intpl = [0] + [f(time - emergenceTime) for time in np.arange(emergenceTime, fixationTime + 1)] + [1] # len = duration + 3
        times_intpl = [times[0]] + list(np.arange(emergenceTime, fixationTime + 1)) + [times[-1]] # len = duration + 3
        candidateInterpolations[(emergenceTime, duration)] = LTEE.create_interpolation_function(times_intpl, freqs_intpl)

    if verbose:
        plt.ylabel('MAE of freqs', fontsize=12)
        plt.xlabel('candidate emergence times', fontsize=12)
        plt.legend(fontsize=10)
        plt.title(f'How MAE changes as emergence time changes, given a duration', fontsize=12)
        plt.show()

    labels = None
    if inferSelection:
        labels = []
        for (emergenceTime, duration), f in candidateInterpolations.items():
            label = f'duration={duration}'
            if durationToSelection is not None:
                label += '_s=%.3f' % durationToSelection[duration]
            times_intpl = range(times[0], times[-1] + 1)
            freqs_intpl = [f(t) for t in times_intpl]
            inferredSelection = inferSelectionForSelectiveSweep(freqs_intpl, times_intpl)
            label += '_inferS=%.3f' % inferredSelection
            labels.append(label)

    if plotLinearInterpolation:
        linearInterpolation = LTEE.create_interpolation_function(times, freqs)
        if inferSelection:
            linearLabel = 'linear'
            times_intpl = range(times[0], times[-1] + 1)
            freqs_intpl = [linearInterpolation(t) for t in times_intpl]
            inferredSelection = inferSelectionForSelectiveSweep(freqs_intpl, times_intpl)
            linearLabel += '_inferS=%.3f' % inferredSelection

    if plotLinearInterpolation:
        compareInterpolation = linearInterpolation
        compareLabel = linearLabel
    else:
        compareInterpolation = None
        compareLabel = None

    if plotCandidateInterpolations:
        PLOT.plotCandidateInterpolations(freqs, times, candidateInterpolations, compareInterpolation=compareInterpolation, colors=colors, labels=labels, compareLabel=compareLabel)
    return candidateInterpolations


def closestValueAndKey(dict, target):
    key = min(dict, key=lambda x: abs(dict[x] - target))
    return dict[key], key


def closestValueAndIndex(list, target):
    index = min(range(len(list)), key=lambda x: abs(list[x] - target))
    return list[index], index


def inferSelectionForSelectiveSweep(freqs, times, regularization=0, thFixed=0.99, mu=1e-10):
    intVariance = np.sum([integratedVariance(freqs[t], freqs[t+1], times[t+1] - times[t]) for t in range(len(freqs) - 1)])
    D = MPL.computeD(freqs, times, mu)
    selection = D / (intVariance + regularization)
    return selection


def computeLogLikelihood(traj, traj_sampled, readDepths=None, meanReadDepth=100):
    """
    Say traj is the underlying true frequencies of the evolving population with population size = N, then uncertainty of the true traj should be ~O(1/N).
    Say traj_sampled is the observed frequencies from a sample of size readDepth, then its resolution should be ~O(1/readDepth).
    We assume readDepth << N
    """

    T, L = traj.shape
    log = 0
    for t in range(T):
        for l in range(L):
            if readDepths is not None:
                readDepth = readDepths[t, l]
            else:
                readDepth = meanReadDepth
            log += computeLogBinomialProbability(traj[t, l], traj_sampled[t, l], readDepth)
    return log


def computeLogBinomialProbability(freq, freq_sampled, readDepth, uncertainty=EPSILON):

    minP, maxP = uncertainty, 1 - uncertainty
    if (freq_sampled > EPSILON and freq_sampled < 1 - EPSILON) or (freq > EPSILON and freq < 1 - EPSILON):
        p = min(maxP, max(minP, freq))
        return stats.binom.logpmf(round(readDepth * freq_sampled), readDepth, p)
    else:
        return 0


def get_clade_size_from_reconstruction(reconstruction):
    return [len(reconstruction.otherMuts)] + [len(_) for _ in reconstruction.cladeMuts]


def check_consistency_between_two_reconstructions_approximately(rec1, rec2):
    return np.array_equal(get_clade_size_from_reconstruction(rec1), get_clade_size_from_reconstruction(rec2))


def print_signal_percentage_by_number_of_groups(rec):
    groups, segmentedIntDxdx = rec.groups, rec.segmentedIntDxdx[0]
    signals = get_signals_by_number_of_groups(groups, segmentedIntDxdx)
        
    total_signal = signals[-1]
    percentages = [round(signal / total_signal, 3) for signal in signals]
    print(percentages)


def get_signals_by_number_of_groups(groups, segmentedIntDxdx):
    for l in range(len(segmentedIntDxdx)):
        segmentedIntDxdx[l, l] = 0
    signals = []
    start = len(groups[0])
    end = len(groups[0])
    for i, group in enumerate(groups[1:]):
        n = len(group)
        end += n
        signal = np.sum(np.absolute(segmentedIntDxdx[start:end, start:end]))
        signals.append(signal)
    return signals


def print_presence_percentage_of_each_group(rec, atLeast=10, largerThan=0.3):
    num_muts = get_num_muts_with_descent_presence_in_each_group(rec.traj, rec.groups, atLeast=atLeast, largerThan=largerThan)
    total_num = np.sum(num_muts)
    percentages = [round(num / total_num, 3) for num in num_muts]
    print(percentages)


def get_num_muts_with_descent_presence_in_each_group(traj, groups, atLeast=10, largerThan=0.3):
    """
    Returns how many mutations in a group has frequency > {largerThan} for at least {atLeast} time points.
    """
    return [len([l for l in group if np.sort(traj[:, l])[-atLeast] > largerThan]) for group in groups[1:]]


class CladeReconstruction:
    NEWCLADE = -1
    def __init__(self, traj, times=None, mu=None, useEffectiveMu=True, meanReadDepth=100, mutantReads=None, readDepths=None, hasInterpolation=False, segmentedIntDxdx=None, groups=None, debug=False, verbose=False, plot=True, timing=False):
        if timing:
            start = timer()
            print(f'{bcolors.OKGREEN}> Initializing variables...', end='\t')
        self.traj = traj
        self.T, self.L = self.traj.shape
        self.times = times if times is not None else np.arange(0, self.T)
        self.mu = mu
        self.useEffectiveMu = useEffectiveMu
        if meanReadDepth is None and readDepths is not None:
            self.meanReadDepth = np.mean(readDepths)
        else:
            self.meanReadDepth = meanReadDepth
        self.mutantReads = mutantReads
        self.readDepths = readDepths
        self.hasInterpolation = hasInterpolation # Lenski's LTEE data, allele frequencies are interpolated when mutant reads are not available at a time point.
        self.verbose = verbose
        self.debug = debug
        self.plot = plot
        self.timing = timing

        # To be constructed
        self.intWeightedDxdx = None
        self.segmentedIntDxdx = segmentedIntDxdx
        self.groups = groups
        self.cladeFreq = None  # clade frequency trajectories, excluding the ancestor clade.
        self.cladeMuts = None  # mutations that belong to a clade, excluding the ancestor clade.
        self.otherMuts = None  # mutations that do not belong to a clade.
        self.cladeMutsWithConfidence = None
        self.cladeFreqWithAncestor = None  # clade frequency trajectories, including the ancestor clade, at index=0.
        self.mutFreqWithinClades = None
        self.periods = None
        self.periodBoundaries = None
        self.refinedPeriods = None
        self.refinedPeriodBoundaries = None
        self.minRefinedPeriodLength = 1
        self.ancestorIndices = None
        # self.periodConnections = None
        self.mullerCladeFreq = None
        self.mullerColors = None

        if timing:
            print('took: %lfs' % (timer() - start))


    def setParamsForClusterization(self, debug=None, weightByBothVariance=False, weightBySmallerVariance=False, weightBySmallerInterpolatedVariance=True, timing=False):
        if debug is not None:
            self.debug = debug
        self.timing = timing
        self.weightByBothVariance = weightByBothVariance
        self.weightBySmallerVariance = weightBySmallerVariance
        self.weightBySmallerInterpolatedVariance = weightBySmallerInterpolatedVariance


    def setParamsForReconstruction(self, debug=None, thFixed=0.99, thExtinct=0, thFixedWithinClade=0.9, numClades=None, percentSignalsToIncludeInMajorClades=99, percentMutsToIncludeInMajorClades=95, preservedPercent=0.5, thLogProbPerTime=10, thFreqUnconnected=0.04, filterThresholds=(5, 0.2), timing=False):
        if debug is not None:
            self.debug = debug
        self.timing = timing
        self.thFixed = thFixed
        self.thExtinct = thExtinct
        self.thFixedWithinClade = thFixedWithinClade
        self.numClades = numClades
        if numClades is None:
            self.percentSignalsToIncludeInMajorClades = percentSignalsToIncludeInMajorClades
            self.percentMutsToIncludeInMajorClades = percentMutsToIncludeInMajorClades  # deprecated
            self.filterThresholds = filterThresholds  # deprecated
        self.preservedPercent = preservedPercent
        self.thLogProbPerTime = thLogProbPerTime
        self.thFreqUnconnected = thFreqUnconnected

        self.getEmergenceTimes()
        self.getFixationTimes()
        self.getExtinctionTimes()


    def setParamsForEvaluation(self, testMultipleMu=False, debug=None, plot=None, intCov=None, defaultReg=None, selection=None, fitness=None, fitness_times=None, flattenBlipsAfterFixation=False, collapseSimilarTrajectories=False, thCollapse=1, assumeCooperationAmongSharedMuts=False, timing=None):
        if debug is not None:
            self.debug = debug
        if plot is not None:
            self.plot = plot
        if timing is not None:
            self.timing = timing
        self.testMultipleMu = testMultipleMu
        self.intCov = intCov
        self.defaultReg = defaultReg
        self.selection = selection
        if selection is not None:
            self.fitness = self.computeFitness(self.traj, self.selection)
        else:
            self.fitness = fitness
        self.fitness_times = fitness_times
        # Postprocess options
        self.flattenBlipsAfterFixation = flattenBlipsAfterFixation
        self.collapseSimilarTrajectories = collapseSimilarTrajectories
        self.thCollapse = thCollapse
        # When computing covariance
        self.assumeCooperationAmongSharedMuts = assumeCooperationAmongSharedMuts
        # To be constructed
        self.recoveredTraj = None
        self.processedTraj = None
        self.recoveredIntCov = None
        self.recoveredSelection = None
        self.selectionByTrueCov = None
        self.recoveredFitness = None
        self.fitnessByTrueCov = None


    def getFixationTimes(self):
        self.fixationTimes = {}
        for l in range(self.L):
            tFixed = self.fixationTime(l)
            if tFixed >= 0:
                self.fixationTimes[l] = tFixed


    def getExtinctionTimes(self):
        self.extinctionTimes = {}
        for l in range(self.L):
            tExtinct = self.extinctionTime(l)
            if tExtinct >= 0:
                self.extinctionTimes[l] = tExtinct


    def fixationTime(self, l, thNumTimePointsAfterFixation=2, thFractionTimePointsAfterFixation=0.05):
        """
        Returns fixation time for a mutation (index l), if it gets fixed. Otherwise returns -1.
        """
        T = self.T
        maxFixationTime = max(0, T - max(thNumTimePointsAfterFixation, int(T * thFractionTimePointsAfterFixation)))
        freqSum = np.sum(self.traj[:T, l])
        for t in range(maxFixationTime):
            if freqSum / (T - t) >= self.thFixed and self.traj[t, l] >= self.thFixed:
                return t
            freqSum -= self.traj[t, l]
        return -1


    def extinctionTime(self, l, thNumTimePointsAfterExtinction=2, thFractionTimePointsAfterExtinction=0.05):
        """
        Returns extinction time for a mutation (index l), if it gets extinct. Otherwise returns -1.
        """
        T = self.T
        maxExtinctionTime = max(0, T - max(thNumTimePointsAfterExtinction, int(T * thFractionTimePointsAfterExtinction)))
        freqSum = np.sum(self.traj[:T, l])
        for t in range(maxExtinctionTime):
            if freqSum / (T - t) <= self.thExtinct and self.traj[t, l] <= self.thExtinct:
                return t
            freqSum -= self.traj[t, l]
        return -1


    def clusterMutations(self):
        """
        Cluster mutations into groups, so that mutations within the same group show cooperating behaviors, while mutations in different groups show competing behaviors. groups[0] contains mutations that show cooperating behaviors with multiple other groups.
        """
        # Using saved groups
        if self.groups is not None:
            if self.verbose:
                print(f'{bcolors.OKGREEN}> Using saved groups from previous computation...')
                PRINT.printGroups(self.groups)
            return

        # Calculating weighted dxdx matrix
        if self.timing:
            start = timer()
            print(f'{bcolors.OKGREEN}> Calculating weights...', end='\t')
        dxdx = computeDxdx(self.traj)
        if self.weightByBothVariance:
            weightedDxdx = self.weightDxdxByBothVariance(dxdx)
        elif self.weightBySmallerVariance:
            weightedDxdx = self.weightDxdxBySmallerVariance(dxdx)
        elif self.weightBySmallerInterpolatedVariance:
            weightedDxdx = self.weightDxdxBySmallerInterpolatedVariance(dxdx)
        else:
            weightedDxdx = dxdx
        if len(self.times) > 1:
            intWeightedDxdx = np.zeros((self.L, self.L))
            timeIntervals = computeTimeIntervals(self.times)
            for t in range(len(weightedDxdx)):
                intWeightedDxdx += weightedDxdx[t] / timeIntervals[t]
            intWeightedDxdx *= np.mean(timeIntervals)
        else:
            intWeightedDxdx = np.sum(weightedDxdx, axis=0)
        for l in range(self.L):
            intWeightedDxdx[l, l] = float('inf')

        # Initialize groups with the pair of sites with the most competing behavior
        minIndex = np.argmin(intWeightedDxdx)
        l1, l2 = minIndex // self.L, minIndex % self.L
        self.groups = [[], [l1]]
        unassigned = set(range(self.L))
        unassigned.remove(l1)
        coopWithGroups = {l: [np.sum([intWeightedDxdx[c, l] for c in clade]) for clade in self.groups[1:]] for l in unassigned}
        self.groups.append([l2])
        mutAdded = l2
        groupIndex = 2
        if l2 != l1:
            unassigned.remove(l2)
        if self.timing:
            print('took: %lfs' % (timer() - start))
            print(f'{bcolors.OKGREEN}> Clustering mutations...')
            start = timer()

        # Aggregate other un-classified sites
        while unassigned:
            maxScore, mutToAdd, targetCooperation = -float('inf'), -1, []
            for l in unassigned:
                if groupIndex > 0:
                    if groupIndex - 1 < len(coopWithGroups[l]):
                        coopWithGroups[l][groupIndex - 1] += intWeightedDxdx[mutAdded, l]
                    else:
                        coopWithGroups[l].append(intWeightedDxdx[mutAdded, l])
                meanCoopWithGroups = [coopWithGroups[l][g] / len(group) for g, group in enumerate(self.groups[1:])]
                score = 2 * np.max(meanCoopWithGroups) - np.sum(meanCoopWithGroups)
                if score > maxScore:
                    maxScore, mutToAdd, targetCooperation = score, l, meanCoopWithGroups
            sortedIndices = np.argsort(targetCooperation)
            maxCoop = targetCooperation[sortedIndices[-1]]
            secondMaxCoop = targetCooperation[sortedIndices[-2]]
            if maxCoop < -1e-5:
                # not cooperating with any groups, assign to a new group
                groupIndex = len(self.groups)
                self.groups.append([mutToAdd])
            elif secondMaxCoop > 1e-5:
                # cooperating with multiple groups, assign to shared
                groupIndex = 0
                self.groups[0].append(mutToAdd)
            else:
                groupIndex = 1 + sortedIndices[-1]
                self.groups[groupIndex].append(mutToAdd)
            unassigned.remove(mutToAdd)
            mutAdded = mutToAdd
            if self.timing:
                percent = 1 - len(unassigned) / self.L
                sys.stdout.write("\r%d%% \t\t\t\ttook %lfs" % (int(round(percent * 100)), timer() - start))
                sys.stdout.flush()
        if self.debug:
            print()

        self.groups = [self.groups[0]] + sorted(self.groups[1:], key=lambda x : - len(x))
        self.intWeightedDxdx = intWeightedDxdx
        self.segmentedIntDxdx = segmentMatrix(intWeightedDxdx, self.groups)
        if self.verbose:
            print(f'{bcolors.OKGREEN}> Forming groups from dxdx matrix...')
            PRINT.printGroups(self.groups)
            # PLOT.plotConstructedClades(self.groups)


    def weightDxdxByBothVariance(self, dxdx):
        for t in range(len(dxdx)):
            var = [self.traj[t, i] * (1 - self.traj[t, i]) for i in range(self.L)]
            for i in range(self.L):
                for j in range(i + 1, self.L):
                    dxdx[t, i, j] *= math.sqrt(var[i] * var[j])
                    dxdx[t, j, i] = dxdx[t, i, j]
        return dxdx


    def weightDxdxBySmallerVariance(self, dxdx):
        for t in range(len(dxdx)):
            var = [self.traj[t, i] * (1 - self.traj[t, i]) for i in range(self.L)]
            for i in range(self.L):
                for j in range(i + 1, self.L):
                    dxdx[t, i, j] *= min(var[i], var[j])
                    dxdx[t, j, i] = dxdx[t, i, j]
        return dxdx


    def weightDxdxBySmallerInterpolatedVariance(self, dxdx):
        for t in range(len(dxdx)):
            interpolated_freq = [(self.traj[t, i] + self.traj[t + 1, i]) / 2 for i in range(self.L)]
            var = [interpolated_freq[i] * (1 - interpolated_freq[i]) for i in range(self.L)]
            for i in range(self.L):
                for j in range(i + 1, self.L):
                    dxdx[t, i, j] *= min(var[i], var[j])
                    dxdx[t, j, i] = dxdx[t, i, j]
        return dxdx


    def checkForSeparablePeriod(self, dedupCladeFixedEvents=True, detectFixationInSharedGroupAfterOneGroupFixes=True, checkForSharedMuts=False):
        """
        Returns cladeFixedEventsDedup, a list of tuples (clade_index, time_fixed_index, signature_mutation_index)
        """

        if self.numClades is None:
            self.setNumCladesAutomatically()
        cladeFixedEvents = []
        if checkForSharedMuts:
            kStart = 0
        else:
            kStart = 1
        for k in range(kStart, self.numClades + 1):
            if self.debug:
                print(f'{bcolors.WARNING} Running checkForSeparablePeriodAndReconstruct()... group {k} has mutations {self.groups[k]}')
            for l in self.groups[k]:
                tFixed = np.min([self.T] + [t for t in range(self.T) if self.traj[t, l] >= self.thFixed])
                tFixedLast = np.max([tFixed] + [t for t in range(self.T) if self.traj[t, l] >= self.thFixed and self.traj[t, l] != 1])
                meanAfterFixed = np.mean(self.traj[tFixed:self.T, l]) if tFixed < self.T else 0
                meanLastTimepoints = np.mean(self.traj[max(0, self.T - 3):self.T, l])
                timeExtinct = np.max([0] + [t for t in range(self.T) if self.traj[t, l] > self.thExtinct])
                meanAfterExtinct = np.mean(self.traj[timeExtinct:self.T, l])
                if tFixed < self.T - 1 and meanAfterFixed >= self.thFixed or meanLastTimepoints >= self.thFixed:
                    # if tFixedLast < tFixed + min(self.T / 20, 20): # TODO: Think about this, do we want it?
                    #     tFixed = tFixedLast
                    cladeFixedEvents.append((k, tFixed, l))
        if cladeFixedEvents and detectFixationInSharedGroupAfterOneGroupFixes:
            firstFixationTime = cladeFixedEvents[0][1]
            k = 0
            for l in self.groups[k]:
                tFixed = np.min([self.T] + [t for t in range(self.T) if self.traj[t, l] >= self.thFixed])
                tFixedLast = np.max([tFixed] + [t for t in range(self.T) if self.traj[t, l] >= self.thFixed and self.traj[t, l] != 1])
                meanAfterFixed = np.mean(self.traj[tFixed:self.T, l]) if tFixed < self.T else 0
                meanLastTimepoints = np.mean(self.traj[max(0, self.T - 3):self.T, l])
                timeExtinct = np.max([0] + [t for t in range(self.T) if self.traj[t, l] > self.thExtinct])
                meanAfterExtinct = np.mean(self.traj[timeExtinct:self.T, l])
                if tFixed > firstFixationTime and tFixed < self.T - 1 and meanAfterFixed >= self.thFixed or meanLastTimepoints >= self.thFixed:
                    # if tFixedLast < tFixed + min(self.T / 20, 20): # TODO: Think about this, do we want it?
                    #     tFixed = tFixedLast
                    cladeFixedEvents.append((k, tFixed, l))

        if dedupCladeFixedEvents:
            cladeFixedEventsDedup = []
            for i, event in enumerate(cladeFixedEvents):
                if i == 0:
                    cladeFixedEventsDedup.append(event)
                k, tFixed, l = event
                if i > 0:
                    if k != cladeFixedEvents[i - 1][0]:
                        cladeFixedEventsDedup.append(event)
        else:
            # No deduplication
            cladeFixedEventsDedup = cladeFixedEvents

        return cladeFixedEventsDedup


    def checkForSeparablePeriodAndReconstruct(self, dedupCladeFixedEvents=True, detectFixationInSharedGroupAfterOneGroupFixes=True, checkForSharedMuts=False, minDurationTimepoint=3, minDurationRatio=0.05, cladeFixedTimes=None, startEndToGroups=None, startEndToDxdx=None):
        """
        The full period of all time points available might be divided into 2 or more periods, if at some time point, the population gets dominated by a clade and the subsequently branches into several sub-clades of it, in which case the reconstruction is better to be done seperately for each period.
        """
        if cladeFixedTimes is None:
            cladeFixedEventsDedup = self.checkForSeparablePeriod(dedupCladeFixedEvents=dedupCladeFixedEvents, detectFixationInSharedGroupAfterOneGroupFixes=detectFixationInSharedGroupAfterOneGroupFixes, checkForSharedMuts=checkForSharedMuts)

            if cladeFixedEventsDedup:
                cladeFixedTimes = sorted(list(np.unique([event[1] for event in cladeFixedEventsDedup])))
                # Merge periods that are too short
                durations = []
                for i in range(len(cladeFixedTimes) + 1):
                    if i == 0:
                        duration = cladeFixedTimes[i]
                    elif i == len(cladeFixedTimes):
                        duration = self.T - cladeFixedTimes[i-1]
                    else:
                        duration = cladeFixedTimes[i] - cladeFixedTimes[i-1]
                    durations.append(duration)
                if self.debug:
                    print('Durations of periods: ', durations)
                timeToRemove = set()
                for i in range(len(cladeFixedTimes) + 1):
                    if durations[i] < max(minDurationTimepoint, self.T * minDurationRatio):
                        if i == 0:
                            timeToRemove.add(cladeFixedTimes[i])
                        elif i == len(cladeFixedTimes):
                            timeToRemove.add(cladeFixedTimes[i-1])
                        else:
                            # Merge this period into the shorter period of two adjacent periods
                            if durations[i-1] < durations[i+1]:
                                timeToRemove.add(cladeFixedTimes[i-1])
                            else:
                                timeToRemove.add(cladeFixedTimes[i])
                for t in timeToRemove:
                    cladeFixedTimes.remove(t)
        else:
            cladeFixedEventsDedup = cladeFixedTimes

        if len(cladeFixedEventsDedup) == 0 or len(cladeFixedTimes) == 0:
            self.reconstructCladeCompetition()
            self.wrapResultsAsOnePeriod()
        else:
            self.periods = []
            self.periodBoundaries = []
            for i in range(len(cladeFixedTimes)):
                tStart = cladeFixedTimes[i - 1] if i > 0 else 0
                tEnd = cladeFixedTimes[i] + 1
                if startEndToGroups is not None and startEndToDxdx is not None:
                    key = min(startEndToGroups, key=lambda x: abs(x[0] - tStart) + abs(x[1] - tEnd))
                    groups = startEndToGroups[key]
                    segmentedIntDxdx = startEndToDxdx[key]
                else:
                    groups = None
                    segmentedIntDxdx = None
                self.periodBoundaries.append((tStart, tEnd))
                self.periods.append(self.reconstructForPeriod(tStart, tEnd, groups=groups, segmentedIntDxdx=segmentedIntDxdx))
            tStart, tEnd = cladeFixedTimes[-1], self.T
            if startEndToGroups is not None and startEndToDxdx is not None:
                key = min(startEndToGroups, key=lambda x: abs(x[0] - tStart) + abs(x[1] - tEnd))
                groups = startEndToGroups[key]
                segmentedIntDxdx = startEndToDxdx[key]
            else:
                groups = None
                segmentedIntDxdx = None
            self.periodBoundaries.append((tStart, tEnd))
            self.periods.append(self.reconstructForPeriod(tStart, tEnd, groups=groups, segmentedIntDxdx=segmentedIntDxdx))

            self.connectCladeFreq()
            self.getMullerCladeFreq()
            if self.verbose:
                PLOT.plotTotalCladeFreq(self.cladeFreqWithAncestor)
                PLOT.plotMullerCladeFreq(self.mullerCladeFreq, self.mullerColors, self.times)

            if self.emergenceTimes is None:
                self.getEmergenceTimes()
            emergenceTimes = self.emergenceTimes
            self.refinedPeriods = []
            self.refinedPeriodBoundaries = []
            for i, (tStart, tEnd) in enumerate(self.periodBoundaries):
                tStart, tEnd = self.periodBoundaries[i]
                tStartRefined = tEndRefined if i > 0 else 0
                _ = [tEnd]
                # tEndRefined is set as when earliest signature mutation emerge.
                if i < len(self.periods) - 1:
                    self.periods[i+1].getCladeMutsWithConfidence()
                    for clade in self.periods[i+1].cladeMutsWithConfidence:
                        signatureMut = max(clade, key=clade.get)
                        _.append(emergenceTimes[signatureMut])
                        # print(f'clade {clade}, signature mutation {signatureMut}, emerges at time {emergenceTimes[signatureMut]}')
                tEndRefined = max(np.min(_), tStartRefined + self.minRefinedPeriodLength)
                if startEndToGroups is not None:
                    key = min(startEndToGroups, key=lambda x: abs(x[0] - tStartRefined) + abs(x[1] - max(tEndRefined, tStart)))
                    groups = startEndToGroups[key]
                    segmentedIntDxdx = startEndToDxdx[key]
                else:
                    groups = None
                self.refinedPeriodBoundaries.append((tStartRefined, max(tEndRefined, tStart)))
                self.refinedPeriods.append(self.reconstructForPeriod(tStartRefined, max(tEndRefined, tStart), groups=groups, segmentedIntDxdx=segmentedIntDxdx))

            if self.plot:
                print("cladeFixedTimes =", cladeFixedTimes)
                print('period boundaries: ', self.periodBoundaries)
                print('ancestors: ', self.getAncestorIndices())
                print('refined period boundaries: ', self.refinedPeriodBoundaries)

            self.amendConnectedCladeFreq()
            if self.plot:
                PLOT.plotTotalCladeFreq(self.cladeFreqWithAncestor, cladeMuts=self.cladeMuts, traj=self.traj, times=self.times)
            self.resolveUnconnectedCladeFreq()
            self.removeUnconnectedAndAbsentClade()
            self.getMullerCladeFreq()
            self.connectCladeMuts()
            if self.plot:
                PLOT.plotTotalCladeFreq(self.cladeFreqWithAncestor, cladeMuts=self.cladeMuts, traj=self.traj, times=self.times)
                PLOT.plotMullerCladeFreq(self.mullerCladeFreq, self.mullerColors, self.times)


    def reconstructForPeriod(self, tStart, tEnd, groups=None, segmentedIntDxdx=None):
        period = CladeReconstruction(self.traj[tStart:tEnd], times=self.times[tStart:tEnd], debug=self.debug, verbose=self.verbose, groups=groups, segmentedIntDxdx=segmentedIntDxdx)
        period.setParamsForClusterization(weightByBothVariance=self.weightByBothVariance, weightBySmallerVariance=self.weightBySmallerVariance)
        period.setParamsForReconstruction(thFixed=self.thFixed, thExtinct=self.thExtinct, percentMutsToIncludeInMajorClades=self.percentMutsToIncludeInMajorClades, preservedPercent=self.preservedPercent, thLogProbPerTime=self.thLogProbPerTime)

        period.clusterMutations()
        period.reconstructCladeCompetition()
        return period


    def reconstructCladeCompetition(self):
        """
        Reconstructs clade competition.
        """
        if self.timing:
            start = timer()

        if self.numClades is None:
            self.setNumCladesAutomatically()

        if self.verbose:
            PLOT.plotCladeSitesTraj(self.traj, self.groups, times=self.times, thFixed=self.thFixed, thExtinct=self.thExtinct, numClades=self.numClades)

        # Exclude fixed and extinct mutations by specified thresholds
        sharedMuts, fixedMuts, self.cladeMuts, extinctMuts, minorMuts = self.extractPolyCladeMuts()

        # Remove clade that does not gather any polymorphic sites.
        self.cladeMuts = [_ for _ in self.cladeMuts if len(_) > 0]
        self.numClades = len(self.cladeMuts)

        # Initial estimation of cladeFreq
        self.cladeFreq = self.getCladeFreqByKDE(normalize=False)

        # Re-classify mutations and obtain estimated cladeFreq recursively
        if self.debug:
            print(f'{bcolors.OKGREEN}> Reclassifying mutations...')
        reclassified, round = {}, 1
        while round == 1 or len(reclassifiedThisRound) > 0:
            # if self.debug:
            #     print(f'{bcolors.GREY}\tround {round}')
            round += 1
            reclassifiedThisRound = self.reclassifyByProb()
            self.cladeFreq = self.getCladeFreqByKDE(normalize=False)
            reclassified.update(reclassifiedThisRound)
        if reclassified and self.debug:
            print(f'{bcolors.OKBLUE}> \treclassified {len(reclassified)} mutations')

        if self.debug:
            print(f'{bcolors.OKGREEN}> Excluding shared mutations...')
        # Exclude high-freq mutations that are likely shared by multiple clades
        excludedMuts = self.excludeSharedMutations()
        if excludedMuts and self.debug:
            print(f'{bcolors.OKBLUE}> \texcluded {len(excludedMuts)} mutations')

        # Final estimation of cladeFreq
        self.cladeFreq = self.getCladeFreqByKDE(normalize=False)

        allMuts = self.cladeMuts + [sharedMuts, fixedMuts, minorMuts, extinctMuts, excludedMuts]
        names = [f'clade {k+1}' for k in range(self.numClades)] + ['shared', 'fixed', 'minor', 'extinct', 'excluded']
        if self.debug:
            print(tabulate([[str(len(_)) for _ in allMuts]], names, tablefmt='plain'))
            # print('\t'.join(names))
            # print('\t'.join([str(len(_)) for _ in allMuts]))
        self.otherMuts = np.concatenate((sharedMuts, fixedMuts, minorMuts, extinctMuts)).astype(int)

        if self.debug:
            print(f'{bcolors.OKGREEN}> Incorporating other mutations...')
        incorporated = self.incorporateOtherMutations()
        if incorporated and self.debug:
            print(f'{bcolors.OKBLUE}> incorporated {len(incorporated)} mutations')

        if self.debug:
            print(f'{bcolors.OKGREEN}> Excluding shared mutations again...')
        excludedAfterIncorporated = self.excludeSharedMutations()
        if excludedAfterIncorporated and self.debug:
            print(f'{bcolors.OKBLUE}> excluded {len(excludedAfterIncorporated)} incorporated mutations')

        for l, owner in excludedAfterIncorporated + excludedMuts:
            self.otherMuts.append(l)
        if self.verbose and incorporated:
            PLOT.plotIncorporated(self.traj, self.cladeFreq, self.cladeMuts, incorporated, excludedAfterIncorporated, times=self.times)

        # Final estimation of cladeFreq TODO: should we get cladeFreq again after incorporating more muts?
        self.cladeFreq = self.getCladeFreqByKDE(normalize=False)

        # Split clade, this will possibly increase number of clades
        self.splitClade()

        self.getCladeFreqWithAncestor()

        if self.verbose:
            PLOT.plotCladeAndMutFreq(self.cladeMuts, self.cladeFreq, self.traj, times=self.times, alpha=0.3, linewidth=1)

        if self.debug:
            print(f'{bcolors.OKGREEN}> Final clade info...')
            PRINT.printFinalCladesInfo(self.traj, self.cladeMuts, self.otherMuts)

        if self.timing:
            print('Reconstructing... took %.3fs' % (timer() - start))


    def wrapResultsAsOnePeriod(self):
        self.cladeFreqWithAncestor = np.hstack((np.expand_dims(1 - np.sum(self.cladeFreq, axis=1), axis=1), self.cladeFreq))
        self.periods = [self]
        self.periodBoundaries = [(0, self.T)]
        self.getAncestorIndices()
        self.getCladeMutsWithConfidence()
        self.getMullerCladeFreq()
        if self.plot:
            print(f'{bcolors.BLACK}No clade getting fixed. Treating the entire evolution as a whole competition period.  ')
            PLOT.plotTotalCladeFreq(self.cladeFreqWithAncestor, cladeMuts=self.cladeMuts, traj=self.traj, times=self.times)
            PLOT.plotMullerCladeFreq(self.mullerCladeFreq, self.mullerColors, self.times)


    def setNumCladesAutomatically(self):
        """
        Sets self.numClades (the number of clades to reconstruct) to include more than x% signals, where x is self.percentMutsToIncludeInMajorClades.
        """
        signals = get_signals_by_number_of_groups(self.groups, self.segmentedIntDxdx[0])
        total_signal = signals[-1]
        for i, signal in enumerate(signals):
            # print(f'i={i}, %.3f' % (signal / total_signal))
            if signal >= self.percentSignalsToIncludeInMajorClades / 100 * total_signal:
                self.numClades = i + 1
                break
        # print(self.numClades, len(self.groups) - 1)
        if self.debug:
            print(f'{bcolors.OKBLUE}> Automatically set numClades={self.numClades}')


    # def setNumCladesAutomatically(self):
    #     """
    #     Sets self.numClades (the number of clades to reconstruct) to include more than x% of mutations, where x is self.percentMutsToIncludeInMajorClades.
    #     """
    #     atLeast, largerThan = self.filterThresholds

    #     if self.T >= 20 or self.L >= 100:
    #         numMutsForGroups = get_num_muts_with_descent_presence_in_each_group(self.traj, self.groups, atLeast=atLeast, largerThan=largerThan)
    #     else:
    #         numMutsForGroups = [len(group) for group in self.groups[1:]]

    #     # PRINT.printNumSignificantMutsInGroups(numMutsForGroups)
    #     numMutsNotShared = np.sum(numMutsForGroups)

    #     numIncluded = 0
    #     for k, group in enumerate(self.groups[1:]):
    #         numIncluded += numMutsForGroups[k]
    #         # print(k, numIncluded, numMutsNotShared)
    #         if numIncluded >= self.percentMutsToIncludeInMajorClades / 100 * numMutsNotShared and numIncluded >= 1:
    #             self.numClades = k + 1
    #             break
    #     if self.debug:
    #         print(f'{bcolors.OKBLUE}> Automatically set numClades={self.numClades}')


    def extractPolyCladeMuts(self):
        """
        1. Extract mutations that get fixed or extinct, according to given thresholds. Exclude them from later steps.
        2. According to the preset (or else automatically set according to percentMutsToIncludeInMajorClades) number of clades to infer, put that many clades into cladeMuts (fixed/extinct mutations has been excluded).
        3. The other mutations are put into minorMuts.
        """
        sharedMuts = []
        cladeMuts = [[] for k in range(self.numClades)]
        minorMuts = []
        fixedMuts = []
        extinctMuts = []

        if self.fixationTimes is None:
            self.getFixationTimes()
        if self.extinctionTimes is None:
            self.getExtinctionTimes()

        for k in range(self.numClades + 1, len(self.groups)):
            minorMuts += self.groups[k]

        for k in range(self.numClades + 1):
            for l in self.groups[k]:
                if l in self.fixationTimes:
                    fixedMuts.append(l)
                elif l in self.extinctionTimes:
                    extinctMuts.append(l)
                elif k == 0:
                    sharedMuts.append(l)
                else:
                    cladeMuts[k - 1].append(l)

        cladeMuts = [clade for clade in cladeMuts if len(clade) > 0]

        return sharedMuts, fixedMuts, cladeMuts, extinctMuts, minorMuts


    def getCladeFreqByKDE(self, normalize=False, thNumNonZeroFreqs=4):
        """
        Estimates clade frequency from allele frequencies assigned to clades, by taking average over allele frequencies around the most densest point, except for values close to 0.
        """
        cladeFreq = np.zeros((self.T, self.numClades), dtype=float)
        freqs = np.arange(0, 1.01, 0.01)
        for t in range(self.T):
            for k in range(self.numClades):
                nonZeroFreqs = [freq for freq in self.traj[t, self.cladeMuts[k]] if freq > 0.01]
                if len(np.unique(self.traj[t, self.cladeMuts[k]])) == 1 or len(nonZeroFreqs) == 0:
                    cladeFreq[t, k] = np.mean(self.traj[t, self.cladeMuts[k]])
                elif len(np.unique(nonZeroFreqs)) < thNumNonZeroFreqs:
                    # too few non-zero freqs compute KDE. Resort to use mean of all non-zero freqs
                    cladeFreq[t, k] = np.mean(nonZeroFreqs)
                else:
                    # find the densest point, and compute mean freq around it.
                    kernel = stats.gaussian_kde(nonZeroFreqs)
                    freq = freqs[np.argmax(kernel(freqs))]
                    freqsAround = [self.traj[t, l] for l in self.cladeMuts[k] if abs(self.traj[t, l] - freq) < 0.1 and self.traj[t, l] > 0.01]
                    if freqsAround:
                        mean = np.mean(freqsAround)
                    else:
                        mean = np.mean(nonZeroFreqs)

                    if np.isnan(mean):
                        print(f'{bcolors.FAIL}Error in getCladeFreqByKDE(cladeMuts, normalize={normalize}, thNumNonZeroFreqs={thNumNonZeroFreqs})')
                        cladeFreq[t, k] = np.mean(nonZeroFreqs)
                    else:
                        cladeFreq[t, k] = mean

            if np.sum(cladeFreq[t]) < 1:
                # who has more mutations with freq > (0.1 + densest point)?
                numLargeFreqs = np.array([np.sum([self.traj[t, l] > 0.1 + cladeFreq[t, k] for l in self.cladeMuts[k]]) for k in range(self.numClades)])
                maxCladeFreq = np.array([np.max(self.traj[t, self.cladeMuts[k]]) for k in range(self.numClades)])
                # indices = np.argsort(-maxCladeFreq)
                indices = np.argsort(-numLargeFreqs)
                for i in indices:
                    # Here 0.5 is a ceiling to avoid the cladeFreq jumping from 0 to 1 at this step
                    # Changed to 1 (no ceiling) for Lenski's data population m5
                    cladeFreq[t, i] += np.min([1 - np.sum(cladeFreq[t]), maxCladeFreq[i] - cladeFreq[t, i], 1 + np.mean(self.traj[t, self.cladeMuts[i]])])
                    if np.sum(cladeFreq[t]) >= 1:
                        break

            if normalize and np.sum(cladeFreq[t]) < 1:
                for k in range(self.numClades):
                    clade = cladeMuts[k]
                    cladeFreq[t, k] = max(cladeFreq[t, k], np.max(self.traj[t, clade]))
                    if np.sum(cladeFreq[t]) > 1:
                        break

            if np.sum(cladeFreq[t]) > 1:
                cladeFreq[t] /= np.sum(cladeFreq[t])

            for k in range(self.numClades):
                clade = self.cladeMuts[k]
                cladeFreq[t, k] = min(cladeFreq[t, k], np.max(self.traj[t, clade]))

        return cladeFreq


    def reclassifyByProb(self):
        """
        Reclassify assigned mutations, by evaluating binomial probabilities.
        Avoid reclassifying mutations that have coop score larger than a threshold specified by self.preservedPercent.
        """
        mutToRemove = [[] for k in range(self.numClades)]
        reclassified = {}
        coopScore = self.getCooperationScoreForAll()
        preservedScore = [np.quantile(list(coopScore[k].values()), self.preservedPercent) for k in range(self.numClades)]
        for k in range(self.numClades):
            for l in self.cladeMuts[k]:
                probMutInClades, numTimePointsCompared = self.getProbMutInClades(l)
                newOwner = np.argmax(probMutInClades)
                if probMutInClades[newOwner] - probMutInClades[k] > self.thLogProbPerTime * numTimePointsCompared and coopScore[k][l] < preservedScore[k]:
                    mutToRemove[k].append(l)
                    self.cladeMuts[newOwner].append(l)
                    reclassified[l] = (k, newOwner, probMutInClades, numTimePointsCompared)
            for l in mutToRemove[k]:
                self.cladeMuts[k].remove(l)
        if self.debug:
            for k in range(self.numClades):
                if mutToRemove[k]:
                    print(f'{bcolors.OKBLUE}\tclade {k + 1} percent %.4f score threshold %.4f' % (self.preservedPercent, preservedScore[k]))
                    for l in mutToRemove[k]:
                        _, newOwner, probMutInClades, numTimePointsCompared = reclassified[l]
                        print(f'{bcolors.OKBLUE}mut {l}, prob {probMutInClades}, num_times {numTimePointsCompared}')
        return reclassified


    def getCooperationScoreForAll(self):
        coopScore = [{} for k in range(self.numClades)]
        for k in range(self.numClades):
            for l in self.cladeMuts[k]:
                coopScore[k][l] = self.getCooperationScore(self.cladeFreq[:, k], self.traj[:, l])
        return coopScore


    def getCooperationScore(self, traj1, traj2, times=None):
        """
        Computes 'cooperation' score, which measures to what extent do two trajectories tend to change in the same direction.
        """
        if times is None:
            times = self.times
        score = 0
        dx1 = computeDx(np.expand_dims(traj1, axis=-1))[:, 0]
        dx2 = computeDx(np.expand_dims(traj2, axis=-1))[:, 0]
        if self.weightByBothVariance:
            weight = np.array([sqrt(var(traj1[t]) * var(traj2[t])) for t in range(len(traj1) - 1)])
        elif self.weightBySmallerVariance:
            weight = np.array([min(var(traj1[t]), var(traj2[t])) for t in range(len(traj1) - 1)])
        elif self.weightBySmallerInterpolatedVariance:
            traj1_intpl = [(traj1[t] + traj1[t + 1]) / 2 for t in range(len(traj1) - 1)]
            traj2_intpl = [(traj2[t] + traj2[t + 1]) / 2 for t in range(len(traj1) - 1)]
            weight = np.array([min(var(traj1_intpl[t]), var(traj2_intpl[t])) for t in range(len(traj1) - 1)])
        else:
            weight = np.full(len(traj1) - 1, 1)
        if len(times) > 1:
            timeIntervals = computeTimeIntervals(times)
            # print(dx1.shape, dx2.shape, weight.shape, timeIntervals.shape)
            score = np.sum(dx1 * dx2 * weight / timeIntervals) * np.mean(timeIntervals)
        else:
            score = np.sum(dx1 * dx2 * weight)
        return score


    def getProbMutInClades(self, l, confident_timepoints=3, min_prob=1e-14):
        """
        Computes probabilities for a mutation to belong to all clades.
        """
        T, K = self.cladeFreq.shape
        probabilities = []
        T_nonExtended = T - 1
        # A step specifically for Lenski's LTEE data, where mutantReads without complete time points are extended with the last available value.
        if self.hasInterpolation:
            while T_nonExtended > 0 and self.traj[T_nonExtended, l] == self.traj[T_nonExtended - 1, l] and self.traj[T_nonExtended, l] < 1 and self.traj[T_nonExtended, l] > 0:
                T_nonExtended -= 1
        if self.mutantReads is not None and self.readDepths is not None:
            for k in range(K):
                probabilities.append([max(min_prob, stats.binom.pmf(k=self.mutantReads[l][t], n=self.readDepths[l][t], p=self.cladeFreq[t, k])) for t in range(T_nonExtended + 1) if self.traj[t, l] > self.cladeFreq[t, k]])
        else:
            for k in range(K):
                probabilities.append([max(min_prob, stats.binom.pmf(k=int(self.meanReadDepth * self.traj[t, l]), n=self.meanReadDepth, p=self.cladeFreq[t, k])) for t in range(T_nonExtended + 1) if self.traj[t, l] > self.cladeFreq[t, k]])

        probMutInClades = np.zeros(K, dtype=np.float64)
        if K > 0:
            minNumTimePoints = np.min([len(probabilities[k]) for k in range(K)])
            numTimePointsToCompare = max(minNumTimePoints, confident_timepoints)
        else:
            numTimePointsToCompare = confident_timepoints
        for k in range(K):
            if len(probabilities[k]) == 0:
                probMutInClades[k] = 1
            else:
                # prob[k] = np.prod(bottomK(probabilities[k], topSeveral))
                probMutInClades[k] = np.sum(np.log10(bottomK(probabilities[k], numTimePointsToCompare)))
        return probMutInClades, numTimePointsToCompare


    def excludeSharedMutations(self, thFreqDiff=0.4, thMaxDiff=0.7):
        """
        Cure the misclassification where a mutation with nearly fixed frequency gets classified to a clade which is not dominant.
        """
        coopScore = self.getCooperationScoreForAll()
        preservedScore = [np.quantile(list(coopScore[k].values()), self.preservedPercent) for k in range(self.numClades)]
        meanScore = [np.mean(list(coopScore[k].values())) for k in range(self.numClades)]

        if self.emergenceTimes is None:
            self.getEmergenceTimes()
        emergenceTimes = self.emergenceTimes
        meanPresentDurations = np.array([np.mean(self.times[-1] - emergenceTimes[self.cladeMuts[k]]) for k in range(self.numClades)])
        # meanEmergenceTimes = np.array([np.mean(emergenceTimes[self.cladeMuts[k]]) for k in range(self.numClades)])

        remove = [[] for k in range(self.numClades)]
        excluded = []
        for k in range(self.numClades):
            for l in self.cladeMuts[k]:
                if emergenceTimes[l] < 0:
                    # Skip mutation that does not appear
                    continue
                toBeExcluded = False
                for t in range(self.T):
                    oneNeighborMean = np.mean(self.cladeFreq[max(0, t-1):min(self.T, t+1), k])
                    twoNeighborMean = np.mean(self.cladeFreq[max(0, t-2):min(self.T, t+2), k])
                    cladeFreqRobust = np.max([self.cladeFreq[t, k], oneNeighborMean, twoNeighborMean])
                    # Either it's too impossible; Or it's somewhat improbable and the coop score is not high
                    # if emergenceTimes[l] > meanEmergenceTimes[k] and self.T - emergenceTimes[l] > 10:
                    #     coeff = (self.T - meanEmergenceTimes[k]) / (self.T - emergenceTimes[l])
                    # else:
                    #     coeff = 1
                    coeff = meanPresentDurations[k] / max(self.times[-1] - emergenceTimes[l], 1)  # avoid divide by 0 in case emergenceTimes[l] == self.times[-1]
                    if cladeFreqRobust < self.traj[t, l] - thMaxDiff:
                        toBeExcluded = True
                    elif coeff * coopScore[k][l] < min(meanScore[k], preservedScore[k]):
                        if (cladeFreqRobust < (self.traj[t, l] - thFreqDiff) or (self.traj[t, l] > self.thFixed and cladeFreqRobust < (self.traj[t, l] - 0.6) / 0.5)):
                            toBeExcluded = True
                    if toBeExcluded:
                        excluded.append((l, k))
                        remove[k].append(l)
                        break
            for l in remove[k]:
                self.cladeMuts[k].remove(l)
        if excluded and self.debug:
            for k in range(self.numClades):
                print(f'{bcolors.GREY}\tclade {k + 1} percent %.4f score threshold %.4f mean score %.4f' % (self.preservedPercent, preservedScore[k], meanScore[k]))
        return excluded


    def splitClade(self, thPresent=0.01):
        """
        Sometimes a clade is actually two or more successive clades where prior ones are extinct.
        """
        numClades = self.numClades
        timesPresent = [[t for t in range(self.T) if self.traj[t, l] > thPresent] for l in range(self.L)]
        firstPresence = [np.min([self.T] + timesPresent[l]) for l in range(self.L)]
        lastPresence = [np.max([-1] + timesPresent[l]) for l in range(self.L)]
        splits = []
        for k, clade in enumerate(self.cladeMuts):
            timesNotPresent = [t for t in range(self.T) if self.cladeFreq[t, k] <= thPresent]
            for t in timesNotPresent:
                if np.all([firstPresence[l] > t or lastPresence[l] < t for l in clade]) and np.any([firstPresence[l] > t and firstPresence[l] != self.T for l in clade]) and np.any([lastPresence[l] < t and lastPresence[l] != -1 for l in clade]):
                    copy = np.zeros(self.T, dtype=float)
                    splits.append((t, k))
                    break
        for splitTime, k in splits:
            if self.debug:
                print(f'Clade {k+1} splits at time tStart+{splitTime}')
            self.splitCladeAtTime(splitTime, k, firstPresence, lastPresence)


    def splitCladeAtTime(self, splitTime, k, firstPresence, lastPresence):
        splitFreq = np.zeros((self.T, 1), dtype=float)
        for t in range(0, splitTime):
            splitFreq[t, 0] = self.cladeFreq[t, k]
            self.cladeFreq[t, k] = 0
        splitMuts = [l for l in self.cladeMuts[k] if lastPresence[l] < splitTime]
        for l in splitMuts:
            self.cladeMuts[k].remove(l)
        self.cladeMuts.append(splitMuts)
        self.cladeFreq = np.hstack((self.cladeFreq, splitFreq))
        self.numClades += 1


    def getCladeFreqWithAncestor(self):
        ancestorFreq = 1 - np.sum(self.cladeFreq, axis=1)
        self.cladeFreqWithAncestor = np.hstack((np.expand_dims(ancestorFreq, axis=-1), self.cladeFreq))


    def incorporateOtherMutations(self):
        """
        Incorporate other mutations that likely belong to one of the clades.
            1. Has high coopScore
            2. Do not have low probability to belong
        """
        if self.numClades == 0:
            return
        numOtherMuts = len(self.otherMuts)
        # Get quantile to determine threshold
        coopScore = self.getCooperationScoreForAll()
        coopScoreThresholds = [max(0.001, np.quantile(list(coopScore[k].values()), self.preservedPercent)) for k in range(self.numClades)]

        # Evaluate other mutations and incorporate
        logProb = np.zeros((numOtherMuts, self.numClades))
        coopScoreOtherMuts = np.zeros((numOtherMuts, self.numClades))
        incorporated = []
        incorporated_indices = []
        for i, l in enumerate(self.otherMuts):
            logProb[i], numTimePointsCompared = self.getProbMutInClades(l)
            for k in range(self.numClades):
                coopScoreOtherMuts[i, k] = self.getCooperationScore(self.cladeFreq[:, k], self.traj[:, l])
            cladeCandidate = np.argmax(coopScoreOtherMuts[i])
            # cooperating?
            if coopScoreOtherMuts[i, cladeCandidate] > coopScoreThresholds[k]:
                # probable by traj?
                probBelong = np.argmax(logProb[i])
                if logProb[i, probBelong] - logProb[i, cladeCandidate] <= self.thLogProbPerTime * numTimePointsCompared:
                    incorporated.append((l, cladeCandidate))
                    incorporated_indices.append(i)
                    self.cladeMuts[cladeCandidate].append(l)

        self.otherMuts = [self.otherMuts[i] for i in range(numOtherMuts) if i not in incorporated_indices]
        if incorporated and self.debug:
            for k in range(self.numClades):
                print(f'{bcolors.OKBLUE}\tclade {k + 1} percent %.4f score threshold %.4f, incorporates mutations {[l for l, clade in incorporated if clade == k]}' % (self.preservedPercent, coopScoreThresholds[k]))

        return incorporated


    def getEmergenceTimes(self, thPresent=0.01):
        """
        Computes emergence time of all mutations.
        """
        self.emergenceTimes = np.full(self.L, -1, dtype=int)
        for l in range(self.L):
            for t in range(self.T):
                if self.traj[t, l] > thPresent:
                    self.emergenceTimes[l] = t
                    break


    def getCladeMutsWithConfidence(self):
        """
        Returns a list of dict {mut: confidence}
        """
        self.cladeMutsWithConfidence = [{} for k in range(self.numClades)]
        for k, clade in enumerate(self.cladeMuts):
            for mut in clade:
                self.cladeMutsWithConfidence[k][mut] = self.getCooperationScore(self.cladeFreq[:, k], self.traj[:, mut])


    def getAncestorIndices(self):
        """
        Return ancestor indices of each period in cladeFreq.
        """
        if self.ancestorIndices is not None:
            return self.ancestorIndices
        numExistingClades = 0
        ancestorIndex = 0
        self.ancestorIndices = [ancestorIndex]
        for indexPeriod, period in enumerate(self.periods):
            if np.sum(period.cladeFreq[-1]) > 0.5:
                ancestorIndex = numExistingClades + np.argmax(period.cladeFreq[-1]) + 1
            else:
                ancestorIndex = ancestorIndex
            numExistingClades += period.numClades
            self.ancestorIndices.append(ancestorIndex)
        return self.ancestorIndices


    def connectCladeFreq(self):
        """
        Connect inferred cladeFreq in each period into a complete cladeFreq throughout the evolution.
        """
        self.removeInferredSubcladeBeforeFixation()
        self.numClades = np.sum([period.numClades for period in self.periods])
        self.cladeFreqWithAncestor = np.zeros((self.T, self.numClades + 1), dtype=float)
        # timeIndexOffset = 0
        self.getAncestorIndices()
        numExistingClades = 0
        for i, period in enumerate(self.periods):
            tStart, tEnd = self.periodBoundaries[i]
            if i < len(self.periods) - 1:
                tEnd -= 1
            ancestorIndex = self.ancestorIndices[i]
            # ancestor
            for t in range(tStart, tEnd):
                self.cladeFreqWithAncestor[t, ancestorIndex] = (1 - np.sum(period.cladeFreq[t-tStart]))
            # clades
            for k in range(period.numClades):
                for t in range(tStart, tEnd):
                    self.cladeFreqWithAncestor[t, numExistingClades + k + 1] = period.cladeFreq[t-tStart, k]
            numExistingClades += period.numClades
            # timeIndexOffset += len(period.times) - 1
        self.cladeFreq = self.cladeFreqWithAncestor[:, 1:]


    def removeInferredSubcladeBeforeFixation(self, tolerance=0.02):
        """
        At the end of each period, the dominant clade should have its freq >= thFixed. However, a subclade might have emerged and might be inferred as well, in which case we can remove that for now, and later recover it from inference in refinedPeriod. This should not be done on the last period.
        We need to be careful not to remove the competing clade.
        """
        nextAncestor = 0
        for i in range(len(self.periods) - 1):
            period = self.periods[i]
            if self.debug:
                print(f'{bcolors.WARNING}Running removeInferredSubcladeBeforeFixation()... period {i} length: {period.T} cladeFreq at the end = {period.cladeFreq[-1]} threshold = {self.thFixed} - {tolerance} = {self.thFixed - tolerance}')
            if np.sum(period.cladeFreq[-1]) > 0.5:
                nextAncestor = np.argmax(period.cladeFreq[-1])
                cladesToModify = []
                if period.cladeFreq[-1, nextAncestor] < self.thFixed - tolerance and period.numClades > 2:
                    curEndFreq = period.cladeFreq[-1, nextAncestor]
                    for k in sorted([k for k in range(period.numClades) if k != nextAncestor], key=lambda k: -period.cladeFreq[-1, k]):
                        cladesToModify.append(k)
                        curEndFreq += period.cladeFreq[-1, k]
                        if curEndFreq >= self.thFixed - tolerance or period.numClades - len(cladesToModify) <= 2:
                            break
                    cladesToRemove = []
                    for k in cladesToModify:
                        numNearestTimepoints = min(len(period.cladeFreq) // 5, 100)
                        # Either frequency of persisting clade is too large, or it is an emerging subclade with increasing frequency
                        # TODO: See if it is possible to try to remove emerging subclade again, after we have inferred for refinedPeriods.
                        # recoveredByNextRefinedPeriod = False
                        # nextRefinedPeriod = self.refinedPeriods[i + 1]
                        # for k2 in range(nextPeriod.numClades):
                        #     numOverlappingTimepoints = period.tEnd - nextRefinedPeriod.tStart
                        #     freqDiff = period.cladeFreq[-numOverlappingTimepoints:, k] - nextRefinedPeriod.cladeFreq[:numOverlappingTimepoints, k2]
                        #     if np.mean(np.absolute(freqDiff)) < 0.03:
                        #         recoveredByNextRefinedPeriod = True
                        if period.cladeFreq[-1, k] >= 0.3 or np.mean(period.cladeFreq[-numNearestTimepoints:, k]) < np.mean(period.cladeFreq[-numNearestTimepoints//2:, k]):
                            if self.debug:
                                print(f'{bcolors.WARNING}(0-indexed) clade {k} freq at the end {period.cladeFreq[-1, k]} {np.mean(period.cladeFreq[-numNearestTimepoints:, k])} < {np.mean(period.cladeFreq[-numNearestTimepoints//2:, k])} ?')
                            cladesToRemove.append(k)
                    for k in cladesToRemove:
                        cladesToModify.remove(k)

                    if self.debug:
                        print(f'{bcolors.WARNING}(0-indexed) cladesToModify: {cladesToModify}\n(0-indexed) cladesToRemove: {cladesToRemove}')

                    # Before removing, map old indices to new indices (after removing some clades)
                    oldIndexToNewIndex = {}
                    newIndex = 0
                    for k in range(period.numClades):
                        if k not in cladesToRemove:
                            oldIndexToNewIndex[k] = newIndex
                            newIndex += 1

                    # Remove clades in cladesToRemove
                    for t in range(len(period.cladeFreq)):
                        period.cladeFreq[t, nextAncestor] += np.sum(period.cladeFreq[t, cladesToRemove])
                    newCladeFreq = np.zeros((len(period.cladeFreq), period.numClades - len(cladesToRemove)))
                    for i, k in enumerate([k for k in range(period.numClades) if k not in cladesToRemove]):
                        for t in range(len(period.cladeFreq)):
                            newCladeFreq[t, i] = period.cladeFreq[t, k]
                    period.cladeFreq = newCladeFreq
                    period.numClades -= len(cladesToRemove)
                    for k in cladesToRemove:
                        period.otherMuts += period.cladeMuts[k]
                    period.cladeMuts = [clade for k, clade in enumerate(period.cladeMuts) if k not in cladesToRemove]

                    # Set new indices
                    nextAncestor = oldIndexToNewIndex[nextAncestor]
                    for i in range(len(cladesToModify)):
                        cladesToModify[i] = oldIndexToNewIndex[cladesToModify[i]]

                    # Modify clades in cladesToModify
                    if cladesToModify:
                        originalTraj = np.copy(period.cladeFreq[:, nextAncestor])
                        if self.thFixed - tolerance - period.cladeFreq[-1, nextAncestor] < 0.03:
                            period.cladeFreq[-1, nextAncestor] += self.thFixed - tolerance - period.cladeFreq[-1, nextAncestor]
                        else:
                            gradient = np.linspace(0, 1, len(period.cladeFreq))
                            highestPossibleFreqs = np.sum(period.cladeFreq[:, [nextAncestor] + cladesToModify], axis=1)
                            for t in range(len(period.cladeFreq)):
                                highestFreq = np.max(period.traj[t, period.cladeMuts[nextAncestor]])
                                period.cladeFreq[t, nextAncestor] = max(min(gradient[t] * highestFreq, highestPossibleFreqs[t]), period.cladeFreq[t, nextAncestor])
                        missingFreq = period.cladeFreq[:, nextAncestor] - originalTraj
                        sortedCladesToModify = sorted([k for k in cladesToModify], key=lambda k: self.getCooperationScore(period.cladeFreq[:, k], period.cladeFreq[:, nextAncestor], times=period.times))
                        for t in range(len(period.cladeFreq)):
                            for k in sortedCladesToModify:
                                tmpFreq = max(0, period.cladeFreq[t, k] - missingFreq[t])
                                missingFreq[t] -= period.cladeFreq[t, k] - tmpFreq
                                period.cladeFreq[t, k] = tmpFreq
                                if missingFreq[t] < 1e-4:
                                    continue
            else:
                # The ancestor clade in previous period remains to be dominant at the beginning of this period
                pass


    def amendConnectedCladeFreq(self):
        """
        Use inference results on refinedPeriods to amend cladeFreq inferred by splitting into periods at fixation times. Results from refinedPeriods have frequencies of sub-clades when they just emerge. We identify
        For each pair of (refinedPeriod, period)
        refinedPeriod: [timeSubcladeEmerge, timeNextSubcladeEmerge]
        period: [timeDominantCladeFix, timeNextDominantCladeFix]
        We amened cladeFreq in time range [timeSubcladeEmerge, timeDominantCladeFix]
        """
        for i in range(len(self.periods)):
            self.periods[i].getCladeMutsWithConfidence()
            self.refinedPeriods[i].getCladeMutsWithConfidence()
        numExistingClades = self.periods[0].numClades
        for i in range(1, len(self.periods)):
            tStart, tEnd = self.periodBoundaries[i]
            tStartRefined, tEndRefined = self.refinedPeriodBoundaries[i]
            ancestorIndex = self.ancestorIndices[i]
            period, refinedPeriod = self.periods[i], self.refinedPeriods[i]
            precursorBySuccesor = self.getPrecursorForCladesInLatterPeriod(refinedPeriod, period)
            for t in range(tStartRefined, tStart):
                if self.debug:
                    print(f'{bcolors.WARNING} Running amendConnectedCladeFreq()... for {i}th period (0-indexed). Modifying cladeFreq in timepoints [{tStartRefined}, {tStart}). ancestorIndex={ancestorIndex}')
                addedFreq = 0
                for k in range(period.numClades):
                    if k in precursorBySuccesor:
                        precursor = precursorBySuccesor[k]
                        # Skip clade that are not emerging new subclades TODO: Think more carefully about this filtering
                        numNearestTimepoints = min(10, refinedPeriod.T // 5)
                        if np.mean(refinedPeriod.cladeFreq[0:numNearestTimepoints, precursor]) > 0.1:
                            continue
                        if self.debug and t == tStartRefined:
                            print(f'{bcolors.OKBLUE}period {i} (0-indexed), clade {precursor+1} in refined period == clade {k+1} in period.')
                        if self.debug:
                            print(f't={t}, Clade {k+1} in period has freq={self.cladeFreqWithAncestor[t, k + numExistingClades + 1]}')
                            print(f't={t}, Clade {precursor+1} in refined period has freq={refinedPeriod.cladeFreq[t - tStartRefined, precursor]}')
                        addedFreq += refinedPeriod.cladeFreq[t - tStartRefined, precursor] - self.cladeFreqWithAncestor[t, k + numExistingClades + 1]
                        self.cladeFreqWithAncestor[t, k + numExistingClades + 1] = refinedPeriod.cladeFreq[t - tStartRefined, precursor]
                if self.cladeFreqWithAncestor[t, ancestorIndex] < addedFreq:
                    for k in range(len(self.cladeFreqWithAncestor[t])):
                        if k != ancestorIndex:
                            self.cladeFreqWithAncestor[t, k] *= self.cladeFreqWithAncestor[t, ancestorIndex] / addedFreq
                    self.cladeFreqWithAncestor[t, ancestorIndex] = 0
                else:
                    self.cladeFreqWithAncestor[t, ancestorIndex] -= addedFreq

            numExistingClades += period.numClades


    def getPrecursorForCladesInLatterPeriod(self, period1, period2):
        """
        Connects clades inferred in two periods in the order of connection score.
        The starting time of period2 needs to be later than the starting time of period1.
        """
        precursors = {}
        connectionScores = []
        if period2.cladeMutsWithConfidence is None:
            period2.getCladeMutsWithConfidence()
        if period1.cladeMutsWithConfidence is None:
            period1.getCladeMutsWithConfidence()
        for k2, clade2 in enumerate(period2.cladeMutsWithConfidence):
            for k1, clade1 in enumerate(period1.cladeMutsWithConfidence):
                connectionScores.append((k2, k1, self.getConnectionScore(clade2, clade1)))
        connectionScores = sorted(connectionScores, key=lambda x: x[2], reverse=True)
        for successor, precursor, score in connectionScores:
            if successor not in precursors:
                precursors[successor] = precursor
        return precursors


    def getConnectionScore(self, clade1WithConfidence, clade2WithConfidence):
        """
        Compute how confidently can we connect clade a in a period with clade b in its adjacent period.
        """
        commonMuts = [mut for mut in clade1WithConfidence if mut in clade2WithConfidence]
        score = np.sum([clade1WithConfidence[mut] * clade2WithConfidence[mut] for mut in commonMuts])
        return score


    def resolveUnconnectedCladeFreq(self):
        """
        Sometimes two unconnected clades can be assembled as a clade.
        """
        cladeToTimeUnconnected = {}
        timeUnconnectedToClade = {}
        for k in range(1, self.numClades + 1):
            if k >= len(self.cladeFreqWithAncestor[0]):
                break
            tUnconnected = self.isUnconnected(self.cladeFreqWithAncestor[:, k])
            if tUnconnected is not None:
                if self.debug:
                    print(f'clade {k} unconnected at {tUnconnected}')
                cladeToTimeUnconnected[k] = tUnconnected
                timeUnconnectedToClade[tUnconnected] = k
                otherTUnconnected = -1 - tUnconnected
                if otherTUnconnected in timeUnconnectedToClade:
                    otherK = timeUnconnectedToClade[otherTUnconnected]
                    resolved = self.tryResolvingClades(k, otherK)
                    if resolved:
                        timeUnconnectedToClade.pop(tUnconnected)
                        timeUnconnectedToClade.pop(otherTUnconnected)
                        cladeToTimeUnconnected.pop(k)
                        cladeToTimeUnconnected.pop(otherK)
        # print(timeUnconnectedToClade)


    def isUnconnected(self, freqs):
        """
        Determines if a frequency trajectory is unconnected somewhere.
        Returns
            - negative time: before that time, the freqs are all 0.
            - positive time: after that time, the freqs are all 0.
        """
        thFreqUnconnected = self.thFreqUnconnected
        T = len(freqs)
        deltaFreqs = [freqs[t + 1] - freqs[t] for t in range(T - 1)]
        nonZeroFreqs = [freqs[t] for t in range(T) if freqs[t] >= 1e-6]
        nonZeroDeltaFreqs = [deltaFreqs[t] for t in range(T - 1) if deltaFreqs[t] >= 1e-6]
        meanNonZeroDeltaFreqs = np.mean(nonZeroDeltaFreqs)
        timeCliff = np.argmax(np.absolute(deltaFreqs))
        # if self.debug:
        #     print(f'timeCliff: ', timeCliff)
        if timeCliff == 0 or timeCliff == T - 2:
            return None
        if np.all(freqs[:timeCliff + 1] < 1e-6) and freqs[timeCliff + 1] > max(thFreqUnconnected, meanNonZeroDeltaFreqs):
            return - (timeCliff + 1)
        if np.all(freqs[timeCliff + 1:] < 1e-6) and freqs[timeCliff] > max(thFreqUnconnected, meanNonZeroDeltaFreqs):
            return timeCliff
        return None


    def tryResolvingClades(self, k1, k2, thMerge=0.05, thPresent=0.01):
        """
        Assembles two unconnected clades as a clade. Helper function of assembleUnconnectedCladeFreq()
        """
        cladeToPeriod = self.mapCladeToPeriod()
        p1, p2 = cladeToPeriod[k1], cladeToPeriod[k2]
        if p1 > p2:
            return self.tryResolvingClades(k2, k1)
        tUnconnected1 = abs(self.isUnconnected(self.cladeFreqWithAncestor[:, k1]))
        tUnconnected2 = abs(self.isUnconnected(self.cladeFreqWithAncestor[:, k2]))
        k1InPeriod = k1 - 1 - int(np.sum([_period.numClades for _period in self.periods[:p1]]))
        k2InPeriod = k2 - 1 - int(np.sum([_period.numClades for _period in self.periods[:p2]]))
        if self.debug:
            print(f"{bcolors.OKGREEN}>Trying assemble clade {k1InPeriod} in period {p1} and clade {k2InPeriod} in period {p2}...")
        period1, period2 = self.periods[p1], self.periods[p2]
        if k1InPeriod < 0 or k2InPeriod < 0:
            return False

        # Determine if they can be seamlessly connected at tUnconnected
        canSeamlesslyConnect = abs(self.cladeFreqWithAncestor[tUnconnected1, k1] - self.cladeFreqWithAncestor[tUnconnected2, k2]) < thMerge
        if self.debug:
            print(f"{bcolors.OKBLUE}Can seamlessly connect? ({abs(self.cladeFreqWithAncestor[tUnconnected1, k1] - self.cladeFreqWithAncestor[tUnconnected2, k2])} < {thMerge}? )", canSeamlesslyConnect)
        if not canSeamlesslyConnect:
            return False

        # Determine if k1 and k2 should be assembled (if the assembled trajectory better cooperates with the signature mutations)
        period1.getCladeMutsWithConfidence()
        period2.getCladeMutsWithConfidence()
        mutsToConf1 = period1.cladeMutsWithConfidence[k1InPeriod]
        mutsToConf2 = period2.cladeMutsWithConfidence[k2InPeriod]
        sigMut1 = max(mutsToConf1, key=mutsToConf1.get)
        sigMut2 = max(mutsToConf2, key=mutsToConf2.get)
        assembledTraj = self.cladeFreqWithAncestor[:, k1] + self.cladeFreqWithAncestor[:, k2]
        if self.debug:
            plt.figure(figsize=(10, 4))
            plt.plot(range(len(assembledTraj)), assembledTraj, linestyle='dashed', alpha=0.5, linewidth=10)
            # plt.plot(range(len(period1.cladeFreq[:, k1InPeriod])), period1.cladeFreq[:, k1InPeriod], linestyle='dashed', alpha=0.5, linewidth=5)
            # plt.plot(range(len(period2.cladeFreq[:, k2InPeriod])), period2.cladeFreq[:, k2InPeriod], linestyle='dashed', alpha=0.5, linewidth=5)
            for l in [sigMut1, sigMut2]:
                plt.plot(range(self.T), self.traj[:, l], label=f'{l}')
            plt.legend()
            plt.title(f'Assembled cladeFreq, and freq-traj of signature mutations of assembled clades')
            plt.show()
        betterCooperatingBehavior = True
        for l in [sigMut1, sigMut2]:
            if self.getCooperationScore(assembledTraj, self.traj[:, l]) < max(self.getCooperationScore(self.cladeFreqWithAncestor[:, k1], self.traj[:, l]), self.getCooperationScore(self.cladeFreqWithAncestor[:, k2], self.traj[:, l])):
                betterCooperatingBehavior = False
        if self.debug:
            print(f"{bcolors.OKBLUE}Better cooperating behavior? ", betterCooperatingBehavior)

        if betterCooperatingBehavior:
            # Assemble the cladeFreq.
            for t in range(tUnconnected2, self.T):
                self.cladeFreqWithAncestor[t, k1] += self.cladeFreqWithAncestor[t, k2]
                self.cladeFreqWithAncestor[t, k2] = 0

            # print(f"{bcolors.OKBLUE}Before assembling, shape of cladeFreqWithAncestor: ", self.cladeFreqWithAncestor.shape)
            # self.cladeFreqWithAncestor = np.hstack((self.cladeFreqWithAncestor[:, :k2], self.cladeFreqWithAncestor[:, k2+1:]))
            # print(f"{bcolors.OKBLUE}After assembling, shape of cladeFreqWithAncestor: ", self.cladeFreqWithAncestor.shape)

            # Add mutations that belong to the latter clade, to its previous equivalent in previous period.
            # for l in period2.cladeMuts[k2InPeriod]:
            #     if l not in period1.cladeMuts[k1InPeriod]:
            #         for k, muts in enumerate([period1.otherMuts] + period1.cladeMuts):
            #             if l in muts:
            #                 muts.remove(l)
            #         period1.cladeMuts[k1InPeriod].append(l)
            # Remove the latter clade in latter period
            # period2.cladeMuts.pop(k2InPeriod)
            # period2.numClades = len(period2.cladeMuts)
            # self.numClades = len(self.cladeFreqWithAncestor[0]) - 1
            return True

        # Determine if k1 and k2 should split at a different time (different from tUnconnected)
        tExtinctSigMut1 = np.max([-1] + [t for t in range(self.T) if self.traj[t, sigMut1] > thPresent])
        tEmergeSigMut2 = np.min([self.T] + [t for t in range(self.T) if self.traj[t, sigMut2] > thPresent])
        if self.debug:
            print(f'{bcolors.OKBLUE}mut {sigMut1} extinct at {tExtinctSigMut1}')
            print(f'{bcolors.OKBLUE}mut {sigMut2} emerge at {tEmergeSigMut2}')
        splitTimeToScore = {}
        tStart, tEnd = min(tExtinctSigMut1, tEmergeSigMut2), max(tExtinctSigMut1, tEmergeSigMut2)
        for tSplit in range(tStart, tEnd):
            traj1, traj2 = np.copy(assembledTraj), np.copy(assembledTraj)
            for t in range(tSplit, self.T):
                traj1[t] = 0
            for t in range(0, tSplit):
                traj2[t] = 0
            splitTimeToScore[tSplit] = self.getCooperationScore(traj1, self.traj[:, sigMut1]) + self.getCooperationScore(traj2, self.traj[:, sigMut2])
        bestSplitTime = max(splitTimeToScore, key=splitTimeToScore.get)
        bestScore = splitTimeToScore[bestSplitTime]
        originalScore = self.getCooperationScore(self.cladeFreqWithAncestor[:, k1], self.traj[:, sigMut1]) + self.getCooperationScore(self.cladeFreqWithAncestor[:, k2], self.traj[:, sigMut2])
        if bestScore > originalScore:
            if self.debug:
                print(f'{bcolors.OKBLUE}Best split time: ', bestSplitTime)
            # Rearrange the cladeFreq.
            for t in range(0, bestSplitTime):
                self.cladeFreqWithAncestor[t, k1] = assembledTraj[t]
                self.cladeFreqWithAncestor[t, k2] = 0
            for t in range(bestSplitTime, self.T):
                self.cladeFreqWithAncestor[t, k1] = 0
                self.cladeFreqWithAncestor[t, k2] = assembledTraj[t]
            # Even out the split point.
            defaultSmoothWindow = np.sum(assembledTraj > thPresent) // 5
            if tExtinctSigMut1 > tEmergeSigMut2:
                smoothWindow = min(tExtinctSigMut1 - tEmergeSigMut2, defaultSmoothWindow)
            else:
                smoothWindow = defaultSmoothWindow
            if self.debug:
                print(f'{bcolors.OKBLUE}Default smooth window: ', defaultSmoothWindow)
                print(f'{bcolors.OKBLUE}Smooth window: ', smoothWindow)
            tStart, tEnd = max(0, bestSplitTime - smoothWindow // 2), min(bestSplitTime + smoothWindow // 2, self.T)
            for t in range(tStart, tEnd):
                totalFreq = self.cladeFreqWithAncestor[t, k1] + self.cladeFreqWithAncestor[t, k2]
                sigMutTotalFreq = self.traj[t, sigMut1] + self.traj[t, sigMut2]
                self.cladeFreqWithAncestor[t, k1] = totalFreq * self.traj[t, sigMut1] / sigMutTotalFreq
                self.cladeFreqWithAncestor[t, k2] = totalFreq * self.traj[t, sigMut2] / sigMutTotalFreq
                # self.cladeFreqWithAncestor[t, k1] = totalFreq * (tEnd - t) / (tEnd - tStart)
                # self.cladeFreqWithAncestor[t, k2] = totalFreq * (t - tStart) / (tEnd - tStart)
            return True
        return False


    def removeUnconnectedAndAbsentClade(self):
        """
        Remove clade whose frequency trajectory is not properly connected, and clades with frequency=0 throughout the evolution.
        """
        cladeToPeriod = self.mapCladeToPeriod()
        periodToRemovedClades = {i: [] for i in range(len(self.periods))}
        removedClades = set()
        for k in range(1, self.numClades + 1):
            tUnconnected = self.isUnconnected(self.cladeFreqWithAncestor[:, k])
            if self.debug and tUnconnected is not None:
                print(f'clade {k} unconnected at {tUnconnected}')
            if tUnconnected is not None or np.all(self.cladeFreqWithAncestor[:, k] < 1e-6):
                periodIndex = cladeToPeriod[k]
                period = self.periods[periodIndex]
                ancestorIndex = self.ancestorIndices[periodIndex + 1]
                numCladesInPreviousPeriods = int(np.sum([_period.numClades for _period in self.periods[:periodIndex]]))
                cladeIndexInPeriod = k - 1 - numCladesInPreviousPeriods
                ancestorIndexInPeriod = ancestorIndex - 1 - numCladesInPreviousPeriods
                if ancestorIndex == k or ancestorIndexInPeriod < 0:
                    continue
                # print(f'clade {k} is to be removed, the ancestor of next period is {ancestorIndex}')
                for t in range(self.T):
                    self.cladeFreqWithAncestor[t, ancestorIndex] += self.cladeFreqWithAncestor[t, k]
                    self.cladeFreqWithAncestor[t, k] = 0
                # print(f'tranferring mutations of clade {k} to clade {ancestorIndex}')
                # print(f'clade {k}: ', period.cladeMuts[cladeIndexInPeriod])
                # print(f'clade {ancestorIndex}: ', period.cladeMuts[ancestorIndexInPeriod])
                for l in period.cladeMuts[cladeIndexInPeriod]:
                    period.cladeMuts[ancestorIndexInPeriod].append(l)
                period.cladeMuts[cladeIndexInPeriod] = []
                # print(f'successfully removed clade {k}')
                periodToRemovedClades[periodIndex].append(cladeIndexInPeriod)
                removedClades.add(k)

        for i, period in enumerate(self.periods):
            # print(f'period {i}')
            if periodToRemovedClades[i]:
                period.cladeMuts = [period.cladeMuts[k] for k in range(period.numClades) if k not in periodToRemovedClades[i]]
                period.numClades = len(period.cladeMuts)
                period.getCladeMutsWithConfidence()
        if self.debug:
            print('Removed clades are: ', removedClades)
        presentClades = [k for k in range(self.numClades + 1) if k not in removedClades]
        oldToNewCladeIndices = {k: i for i, k in enumerate(presentClades)}
        self.numClades = len(oldToNewCladeIndices) - 1
        self.cladeFreqWithAncestor = self.cladeFreqWithAncestor[:, presentClades]
        self.ancestorIndices = [oldToNewCladeIndices[k] for k in self.ancestorIndices]


    def getMullerCladeFreq(self, mullerColors=None):
        if mullerColors is None:
            mullerColors = ['grey'] + PLOT.COLORS[:self.numClades]
        mullerCladeFreq = np.copy(self.cladeFreqWithAncestor)
        ancestorSet = set()
        for i in range(len(self.periods) - 1, -1, -1):
            period = self.periods[i]
            tStart, tEnd = self.periodBoundaries[i]
            ancestorIndex = self.ancestorIndices[i]
            if ancestorIndex not in ancestorSet:
                for t in range(self.T):
                    mullerCladeFreq[t, ancestorIndex] /= 2
                ancestorCopy = np.copy(mullerCladeFreq[:, ancestorIndex:ancestorIndex+1])
                mullerCladeFreq = np.hstack((mullerCladeFreq, ancestorCopy))
                mullerColors.append(mullerColors[ancestorIndex])
            ancestorSet.add(ancestorIndex)
        self.mullerCladeFreq = mullerCladeFreq
        self.mullerColors = deepcopy(mullerColors)


    def connectCladeMuts(self):
        """
        Resolve potential discrepancy among cladeMuts in periods.
        """
        mutToCladeAndConfidence = {}
        cladeMutsWithConfidenceInPeriod = [period.cladeMutsWithConfidence for period in self.periods]
        numExistingClades = 0
        if self.emergenceTimes is None:
            self.getEmergenceTimes()
        emergenceTimes = self.emergenceTimes
        for i, period in enumerate(self.periods):
            tStart, tEnd = self.periodBoundaries[i]
            laterMuts = []
            for k, clade in enumerate(period.cladeMuts):
                for l in clade:
                    if emergenceTimes[l] >= tEnd:
                        laterMuts.append(l)
                    elif l not in mutToCladeAndConfidence or cladeMutsWithConfidenceInPeriod[i][k][l] > mutToCladeAndConfidence[l][1]:
                        mutToCladeAndConfidence[l] = (k + numExistingClades + 1, cladeMutsWithConfidenceInPeriod[i][k][l])
            # Mutations that are not assigned to clades, should be assigned to the ancestor clade of the period in which it emerges during the first half period.
            for l in period.otherMuts:
                lastTStart, lastTEnd = self.periodBoundaries[i - 1] if i > 0 else (0, 0)
                lastTMid = (lastTStart + lastTEnd) / 2
                tMid = (tStart + tEnd) / 2
                if emergenceTimes[l] >= lastTMid and emergenceTimes[l] < tEnd:
                    if l not in mutToCladeAndConfidence:
                        mutToCladeAndConfidence[l] = (self.ancestorIndices[i], self.getCooperationScore(self.traj[:tEnd, l], self.cladeFreqWithAncestor[:tEnd, self.ancestorIndices[i]], times=self.times[:tEnd]))
            numExistingClades += period.numClades

        self.cladeMuts = [[] for _ in range(self.numClades)]
        self.otherMuts = []
        for l, (k, _) in mutToCladeAndConfidence.items():
            if k == 0:
                self.otherMuts.append(l)
            else:
                self.cladeMuts[k - 1].append(l)

        self.otherMuts += [l for l in range(self.L) if l not in mutToCladeAndConfidence]

        self.cladeMutsWithConfidence = [{mut: mutToCladeAndConfidence[mut][1] for mut in clade} for clade in self.cladeMuts]
        # for k, clade in enumerate(self.cladeMuts):
        #     for mut in clade:
        #         # self.cladeMutsWithConfidence[k][mut] = 1 / (1e-6 + np.mean(np.abs(self.traj[:, mut] - self.cladeFreq[:, k])))
        #         self.cladeMutsWithConfidence[k][mut] = self.getCooperationScore(self.cladeFreq[:, k], self.traj[:, mut])


        # TODO: remove the test block below
        #----- TEST BLOCK START -----
        # mutToCladeAndConfidence_test = {l: [] for l in mutToCladeAndConfidence}
        # numExistingClades = 0
        # for i, period in enumerate(self.periods):
        #     for k, clade in enumerate(period.cladeMuts):
        #         for l in clade:
        #             mutToCladeAndConfidence_test[l].append((i, k + numExistingClades + 1, cladeMutsWithConfidenceInPeriod[i][k][l]))
        #     numExistingClades += period.numClades
        #
        # for l in mutToCladeAndConfidence_test:
        #     if len(mutToCladeAndConfidence_test[l]) > 1:
        #         mutToCladeAndConfidence_test[l] = sorted(mutToCladeAndConfidence_test[l], key=lambda x: -x[2])
        #         if self.debug:
        #             self.plotDiscrepancy(l, mutToCladeAndConfidence_test[l])
        #----- TEST BLOCK END -----


    def plotDiscrepancy(self, l, candidates):
        plt.figure(figsize=(10, 4))
        plt.plot(range(self.T), self.traj[:, l], label=f'mut {l}', linewidth=5, linestyle='dashed', alpha=0.5)
        for rank, (i, k, score) in enumerate(candidates):
            plt.plot(range(self.T), self.cladeFreqWithAncestor[:, k], linewidth=1, label=f'clade {k} score %.4f' % score, alpha=1 - rank / len(candidates) / 2)
        plt.ylim(0, 1)
        plt.legend()
        plt.show()


    def evaluate(self, evaluateReconstruction=True, evaluateInference=True):
        """
        Evaluates results.
        """

        self.getMutFreqWithinClades()
        self.getRecoveredTraj()
        self.getProcessedTraj()
        self.recoveredIntCov = self.computeCovariance(self.processedTraj)

        if evaluateReconstruction:
            summary_reconstruction = self.evaluateReconstruction()
        else:
            summary_reconstruction = None

        if evaluateInference:
            summary_inference = self.evaluateInference()
        else:
            summary_inference = None

        return summary_reconstruction, summary_inference


    def evaluateReconstruction(self):
        """
        Evaluates how good is the reconstruction in terms of consistency between recoveredTraj and original traj.
        """
        T, L = self.T, self.L
        MAE_recoveredTraj = MAE(self.recoveredTraj, self.traj)

        logP_recovered = computeLogLikelihood(self.recoveredTraj, self.traj, readDepths=self.readDepths, meanReadDepth=self.meanReadDepth)
        logP_processed = computeLogLikelihood(self.processedTraj, self.traj, readDepths=self.readDepths, meanReadDepth=self.meanReadDepth)
        logP_recovered_normalized = logP_recovered / (T * L)
        logP_processed_normalized = logP_processed / (T * L)

        summary = {
            'MAE_recoveredTraj<0.001': MAE_recoveredTraj < 0.001,
            'MAE_recoveredTraj<0.01': MAE_recoveredTraj < 0.01,
            'MAE_recoveredTraj<0.05': MAE_recoveredTraj < 0.05,
            'MAE_recoveredTraj': MAE_recoveredTraj,
            'logP_recovered': logP_recovered,
            'logP_recovered_normalized': logP_recovered_normalized,
            'logP_processed': logP_processed,
            'logP_processed_normalized': logP_processed_normalized,
        }
        return summary


    def evaluateInferenceFitnessOnly(self, muList=[1e-10], regList=[1e-2, 1e-1, 1, 2, 4, 6, 8, 16, 32], windowList=[0, 1, 2, 4, 8, 16, 32, 64, 128], defaultReg=1, defaultWindow=2):

        # muList=[0, 1e-10, 1e-8, 1e-6, 1e-5, 5e-5, 1e-4, 2e-4, 3e-4, 4e-4, 5e-4, 6e-4, 7e-4, 8e-4, 9e-4, 1e-3]
        if self.timing:
            start = timer()
        if self.defaultReg is not None:
            defaultReg = self.defaultReg
        res = {}
        muList = muList if self.testMultipleMu else [self.mu]
        for mu in muList:
            for reg in regList + [defaultReg]:
                recoveredSelection = self.inferSelection(self.recoveredIntCov, mu=mu, regularization=reg)
                recoveredFitness = self.computeFitness(self.traj, recoveredSelection)
                recoveredFitnessByProcessedTraj = self.computeFitness(self.processedTraj, recoveredSelection)
                res[(mu, reg)] = (self.recoveredIntCov, recoveredSelection, recoveredFitness, recoveredFitnessByProcessedTraj)

        res_est = {}
        for window in windowList:
            if window >= self.T // 2:
                break
            estCov = EST.getRegularizedEstimate(self.traj, self.times, window)
            selectionByEstCov =self.inferSelection(estCov, regularization=defaultReg)
            fitnessByEstCov = self.computeFitness(self.traj, selectionByEstCov)
            fitnessByEstCovAndProcessedTraj = self.computeFitness(self.processedTraj, selectionByEstCov)
            res_est[window] = (estCov, selectionByEstCov, fitnessByEstCov, fitnessByEstCovAndProcessedTraj)
            # print('window=', window, selectionByEstCov[0])
        if defaultWindow not in res_est.keys():
            defaultWindow = np.max(list(res_est.keys()))
        # print(f'default window = {defaultWindow}')

        intVar = computeIntegratedVariance(self.traj, self.times)
        selection_SL = self.inferSelection(intVar, mu=self.mu, regularization=defaultReg)
        fitness_SL = self.computeFitness(self.traj, selection_SL)

        if self.fitness is None:
            _, self.recoveredSelection, self.recoveredFitness, self.recoveredFitnessByProcessedTraj = res[(self.mu, defaultReg)]
            estCov, selectionByEstCov, fitnessByEstCov, fitnessByEstCovAndProcessedTraj = res_est[defaultWindow]
            self.summary = {
                'mu_recover': self.mu,
                'reg_recover': 1,
                'true': (self.traj, self.intCov, self.selection, self.fitness),
                'SL': (self.traj, intVar, selection_SL, fitness_SL),
                EST_METHOD_NAME: (self.processedTraj, estCov, selectionByEstCov, fitnessByEstCov, fitnessByEstCovAndProcessedTraj),
                OUR_METHOD_NAME: (self.processedTraj, self.recoveredIntCov, self.recoveredSelection, self.recoveredFitness, self.recoveredFitnessByProcessedTraj),
                'all_fitness': {k: v[2] for k, v in res.items()},
                'all_selection': {k: v[1] for k, v in res.items()},
            }
            if self.timing:
                print('Evaluating... took %.3fs' % (timer() - start))
            return self.summary

        if self.fitness_times is not None:
            indices = [list(self.times).index(t) for t in self.fitness_times]
        else:
            indices = list(range(len(self.times)))

        # (mu_recover, reg_recover) = max(res.keys(), key=lambda x: -MAE(self.fitness, res.get(x)[2][indices]))
        # _, _, self.recoveredFitness, self.recoveredFitnessByProcessedTraj = res[(mu_recover, reg_recover)]
        _, _, self.recoveredFitness, self.recoveredFitnessByProcessedTraj = res[(mu_recover, defaultReg)]

        estCov, selectionByEstCov, fitnessByEstCov, fitnessByEstCovAndProcessedTraj = res_est[defaultWindow]

        if self.plot:
            self.plotTraj(self.recoveredTraj)
            self.plotTraj(self.processedTraj, name='Processed traj')
            # PLOT.plotRegularizationComparison(res)
            # PLOT.plotWindowComparison(res_est)
            print(f'{bcolors.BLACK}Plotting recovered results using mu of 1e-10 and regularization of 1')
            _, _, self.recoveredFitness, self.recoveredFitnessByProcessedTraj = res[(1e-10, 1)]
            # self.plotRecoveredSelection(1)
            self.plotRecoveredFitness(anotherFitnessToCompare=self.recoveredFitnessByProcessedTraj, anotherLabel='by_processed_traj')
            print(f'{bcolors.BLACK}Plotting recovered results using mu of {mu_recover} and regularization of {reg_recover}')
            _, _, recoveredFitness_tmp, recoveredFitnessByProcessedTraj_tmp = res[(mu_recover, reg_recover)]
            # self.plotRecoveredSelection(reg_recover)
            self.plotRecoveredFitness(anotherFitnessToCompare=recoveredFitnessByProcessedTraj_tmp, anotherLabel='by_processed_traj')
        self.summary = {
            'mu_recover': mu_recover,
            'reg_recover': reg_recover,
            'true': (self.traj, self.fitness),
            'SL': (self.traj, fitness_SL),
            EST_METHOD_NAME: (self.processedTraj, estCov, selectionByEstCov, fitnessByEstCov, fitnessByEstCovAndProcessedTraj),
            OUR_METHOD_NAME: (self.processedTraj, self.recoveredFitness, self.recoveredFitnessByProcessedTraj),
            'all_fitness': {k: v[2] for k, v in res.items()},
            'all_selection': {k: v[1] for k, v in res.items()},
        }
        if self.timing:
            print('Evaluating... took %.3fs' % (timer() - start))
        return self.summary


    def evaluateInference(self, regList=[1, 2, 4, 8, 16, 32, 64, 128], windowList=[0, 1, 2, 4, 8, 16, 32, 64, 128]):
        """
        Evaluates results of clade reconstruction, and fitness / selection coefficients inference.
        """

        if self.selection is None or self.intCov is None:
            return self.evaluateInferenceFitnessOnly()

        if self.timing:
            start = timer()
        res = {}
        for reg in regList:
            selectionByTrueCov = self.inferSelection(self.intCov, regularization=reg)
            recoveredSelection = self.inferSelection(self.recoveredIntCov, regularization=reg)
            fitnessByTrueCov = self.computeFitness(self.traj, selectionByTrueCov)
            recoveredFitness = self.computeFitness(self.traj, recoveredSelection)
            res[reg] = (self.selection, selectionByTrueCov, recoveredSelection, self.fitness, fitnessByTrueCov, recoveredFitness)

        res_est = {}
        selectionByTrueCov, fitnessByTrueCov = res[1][1], res[1][4]
        for window in windowList:
            estCov = EST.getRegularizedEstimate(self.traj, self.times, window)
            selectionByEstCov =self.inferSelection(estCov, regularization=1)
            fitnessByEstCov = self.computeFitness(self.traj, selectionByEstCov)
            res_est[window] = (self.selection, selectionByTrueCov, selectionByEstCov, self.fitness, fitnessByTrueCov, fitnessByEstCov, self.intCov, estCov)

        reg_true = 1
        # reg_true = max([1, 2, 4, 8, 16], key=lambda x: -MAE(self.selection, res.get(x)[1])) # reg for best true-cov performance (MAE)
        reg_recover = max([1, 2, 4, 8, 16], key=lambda x: -MAE(self.selection, res.get(x)[2])) # reg for best est-cov performance (MAE)
        reg_true = max([1, 2, 4, 8, 16], key=lambda x: -MAE(self.selection, res.get(x)[1]))
        print(f"reg_true={reg_true}, reg_recover={reg_recover}")
        # reg_recover = 1
        # reg = max([1, 2, 4, 8, 16], key=lambda x: stats.spearmanr(self.selection, res.get(x)[2])[0] - 10 * MAE(self.selection, res.get(x)[2])) # reg for best est-cov performance (spearmanr & MAE)
        # reg = max([1, 2, 4, 8, 16], key=lambda x: stats.spearmanr(self.selection, res.get(x)[1])[0] - 10 * MAE(self.selection, res.get(x)[1])) # reg for best true-cov performance (spearmanr & MAE)
        # _, self.selectionByTrueCov, self.recoveredSelection, _, self.fitnessByTrueCov, self.recoveredFitness = res[reg]
        _, self.selectionByTrueCov, _, _, self.fitnessByTrueCov, _ = res[reg_true]
        _, _, self.recoveredSelection, _, _, self.recoveredFitness = res[reg_recover]

        window_est = max(windowList, key=lambda x: -MAE(self.selection, res_est.get(x)[2]))
        _, _, self.selectionByEstCov, _, _, self.fitnessByEstCov, _, self.estCov = res_est[window_est]

        intVar = computeIntegratedVariance(self.traj, self.times)
        selection_SL = self.inferSelection(intVar, regularization=1)
        fitness_SL = self.computeFitness(self.traj, selection_SL)

        if self.plot:
            self.plotTraj(self.recoveredTraj)
            self.plotRecoveredIntCov()
            PLOT.plotRegularizationComparison(res)
            PLOT.plotWindowComparison(res_est)
            print(f'{bcolors.BLACK}Plotting recovered results using regularization of 1, window of {window_est}')
            _, _, self.recoveredSelection, _, _, self.recoveredFitness = res[1]
            self.plotRecoveredSelection(1)
            self.plotRecoveredFitness(anotherFitnessToCompare=fitnessByEstCov, anotherLabel='by_est_cov')
            print(f'{bcolors.BLACK}Plotting recovered results using regularization of {reg_recover}, window of {window_est}')
            _, _, self.recoveredSelection, _, _, self.recoveredFitness = res[reg_recover]
            self.plotRecoveredSelection(reg_recover)
            self.plotRecoveredFitness(anotherFitnessToCompare=fitnessByEstCov, anotherLabel='by_est_cov')

        self.summary = {
            'reg_recover': reg_recover,
            'reg': reg_true,
            'window_est':window_est,
            'true': (self.traj, self.intCov, self.selection, self.fitness),
            'SL': (self.traj, intVar, selection_SL, fitness_SL),
            TRUE_COV_NAME: (self.traj, self.intCov, self.selectionByTrueCov, self.fitnessByTrueCov),
            OUR_METHOD_NAME: (self.processedTraj, self.recoveredIntCov, self.recoveredSelection, self.recoveredFitness),
            EST_METHOD_NAME: (self.traj, self.estCov, self.selectionByEstCov, self.fitnessByEstCov),
            'all_fitness': {k: v[5] for k, v in res.items()},
        }
        if self.timing:
            print('Evaluating... took %.3fs' % (timer() - start))
        return self.summary


    def evaluateInferenceForTrueCovAndSL(self, reg=1):
        self.selectionByTrueCov = self.inferSelection(self.intCov, regularization=reg)
        self.fitnessByTrueCov = self.computeFitness(self.traj, self.selectionByTrueCov)
        intVar = computeIntegratedVariance(self.traj, self.times)
        selection_SL = self.inferSelection(intVar, mu=self.mu, regularization=reg)
        fitness_SL = self.computeFitness(self.traj, selection_SL)
        self.summary = {
            'reg': reg,
            'true': (self.traj, self.intCov, self.selection, self.fitness),
            'SL': (self.traj, intVar, selection_SL, fitness_SL),
            TRUE_COV_NAME: (self.traj, self.intCov, self.selectionByTrueCov, self.fitnessByTrueCov),
        }
        return self.summary


    def getMutFreqWithinClades(self):
        """
        Calculate allele frequencies within each clade. Together with self.cladeFreqWithAncestor, they should recover self.traj.
        """
        ancestorToDescendant = self.mapAncestorToDescendant()
        self.mutFreqWithinClades = np.zeros((self.T, self.numClades + 1, self.L))  # 0 is for basal clade

        for k in range(1, self.numClades + 1):
            if ancestorToDescendant is not None and k in ancestorToDescendant:
                descendants = ancestorToDescendant[k]
                descendants.add(k)
                totalFreq = np.sum(self.cladeFreqWithAncestor[:, list(descendants)], axis=1)
                for l in self.cladeMuts[k - 1]:
                    for t in range(self.T):
                        if totalFreq[t] > 1e-6:
                            for d in descendants:
                                self.mutFreqWithinClades[t, d, l] = min(1, self.traj[t, l] / totalFreq[t])
            else:
                for l in self.cladeMuts[k - 1]:
                    for t in range(self.T):
                        if self.cladeFreqWithAncestor[t, k] > 1e-6:
                            self.mutFreqWithinClades[t, k, l] = min(1, self.traj[t, l] / self.cladeFreqWithAncestor[t, k])

        for k in range(self.numClades + 1):
            for l in self.otherMuts:
                for t in range(self.T):
                    self.mutFreqWithinClades[t, k, l] = self.traj[t, l]


    def getRecoveredTraj(self):
        """
        Calculate traj by cladeFreqWithAncestor and mutFreqWithinClades, e.g.,
            recoveredTraj[t, l] = \sum_{k} cladeFreqWithAncestor[t, k] * mutFreqWithinClades[t, k, l]
        """
        self.recoveredTraj = np.sum(np.expand_dims(self.cladeFreqWithAncestor, axis=-1) * self.mutFreqWithinClades, axis=1)


    def getProcessedTraj(self):
        """
        Post-processes recovered trajectories for cleaner trajectories and potentially better inference.
        """
        self.processedTraj = self.recoveredTraj
        if self.flattenBlipsAfterFixation:
            self.flattenBlipsAfterFixationFunc(self.processedTraj)
        if self.collapseSimilarTrajectories:
            self.collapseSimilarTrajectoriesFunc(self.processedTraj)
        pass


    def flattenBlipsAfterFixationFunc(self, traj):
        """
        Trajectories of some mutations may have small blips after they get fixed, which are probably sampling noises.
        """
        if self.timing:
            start = timer()
        T, L = traj.shape

        if self.fixationTimes is None:
            self.getFixationTimes()

        for l, tFixed in self.fixationTimes.items():
            for t in range(tFixed, T):
                traj[t, l] = 1

        if self.timing:
            print('Flattening... took %.3fs' % (timer() - start))


    def collapseSimilarTrajectoriesFunc(self, traj, amendCladeFreq=False):
        """
        Trajectories that are extremely similar are likely from mutations that are fixed within the same clade.
        """
        if self.timing:
            start = timer()
        T, L = traj.shape
        self.fixedWithinClade = [{t: [] for t in range(T)} for k in range(self.numClades)]

        if self.readDepths is not None:
            cladeMeanReadDepths = np.zeros((self.T, self.numClades))
            for t in range(T):
                for k in range(self.numClades):
                    cladeMeanReadDepths[t, k] = np.mean([self.readDepths[l][t] for l in self.cladeMuts[k]])
        else:
            cladeMeanReadDepths = None
        for k in range(self.numClades):
            for l in self.cladeMuts[k]:
                if cladeMeanReadDepths is not None:
                    cladeCdfs = [stats.binom.cdf(k=cladeMeanReadDepths[t, k] * self.cladeFreq[t, k], n=cladeMeanReadDepths[t, k], p=self.cladeFreq[t, k]) for t in range(T)]
                else:
                    cladeCdfs = [stats.binom.cdf(k=self.meanReadDepth * self.cladeFreq[t, k], n=self.meanReadDepth, p=self.cladeFreq[t, k]) for t in range(T)]

                confidences = []
                for t, cladeCdf in enumerate(cladeCdfs):
                    if self.readDepths is not None:
                        mutCdf = stats.binom.cdf(k=self.readDepths[l][t] * traj[t, l], n=self.readDepths[l][t], p=self.cladeFreq[t, k])
                    else:
                        mutCdf = stats.binom.cdf(k=self.meanReadDepth * traj[t, l], n=self.meanReadDepth, p=self.cladeFreq[t, k])
                    if traj[t, l] <= self.cladeFreq[t, k]:
                        confidences.append(1 - (cladeCdf - mutCdf) / cladeCdf)
                    else:
                        # confidences.append(1 - (mutCdf - cladeCdf) / (1 - cladeCdf))
                        confidences.append(1 - (mutCdf - cladeCdf) / mutCdf)

                confSum = np.sum(confidences)
                # print('Mean conf = %.3f' % np.mean(confidences))
                for t in range(self.emergenceTimes[l] + 1, T):
                    if confSum / (T - t) >= self.thCollapse and self.traj[t, l] >= self.cladeFreq[t, k] * self.thFixedWithinClade:
                        self.fixedWithinClade[k][t].append(l)
                        break
                    confSum -= confidences[t]
                # plt.plot(range(T), cdf, linestyle="dashed", linewidth=5, color="red")
                # plt.plot(range(T), confidences, linestyle="dashed", linewidth=5, color="pink")

        for k in range(self.numClades):
            fixedMuts = []
            for t in range(T):
                for l in self.fixedWithinClade[k][t]:
                    fixedMuts.append(l)

                if amendCladeFreq:
                    if fixedMuts:
                        concensusFreq = np.mean(traj[t, fixedMuts])
                        for l in fixedMuts:
                            traj[t, l] = concensusFreq
                        self.cladeFreq[t, k] = concensusFreq
                else:
                    for l in fixedMuts:
                        traj[t, l] = self.cladeFreq[t, k]

                if self.verbose and t == T - 1:
                    print(f'Clade {k}, {len(fixedMuts)} mutations are fixed')

        if amendCladeFreq:
            # Now that self.cladeFreq is amended, we need to update self.cladeFreqWithAncestor accordingly.
            self.getCladeFreqWithAncestor()

        self.fixationWithinCladeTimes = {}
        for k in range(self.numClades):
            for t in range(T):
                for l in self.fixedWithinClade[k][t]:
                    self.fixationWithinCladeTimes[l] = t

        if self.debug:
            print(f'{len(self.fixationWithinCladeTimes)} trajectories collapsed (fixed within clade).')
            plt.hist(list(self.fixationWithinCladeTimes.values()))

        if self.timing:
            print('Collapsing... took %.3fs' % (timer() - start))


    def computeCovariance(self, traj, tStart=None, tEnd=None, interpolate=True):
        """
        Estimates covariance matrix MPL.integrated throught the evolution.
        useRecoveredTraj: whether to use the allele-freq-traj recovered after clade reconstruction, or the original allele-freq-traj.
        interpolate: whether or not interpolate allele frequencies linearly beween each two adjacent time points, when integrating covariance matrix.
        """
        T, L, times = self.T, self.L, self.times
        intCov = np.zeros((L, L), dtype=float)
        if tStart is None:
            tStart = 0
        if tEnd is None:
            tEnd = T
        mutToClade = self.mapMutToClade()
        cladeToPeriod = self.mapCladeToPeriod()

        x = self.getCooccurenceTraj(traj, mutToClade, cladeToPeriod)

        if self.timing:
            start = timer()
        if interpolate:
            for t in range(tStart, tEnd - 1):
                dg = times[t + 1] - times[t]
                for i in range(L):
                    intCov[i, i] += MPL.integratedVariance(traj[t, i], traj[t + 1, i], dg)
                    for j in range(i + 1, L):
                        intCov[i, j] += MPL.integratedCovariance(traj[t, i], traj[t, j], x[t, i, j], traj[t+1, i], traj[t+1, j], x[t+1, i, j], dg)
        else:
            for t in range(tStart, tEnd):
                if t == 0:
                    dg = times[1] - times[0]
                elif t == len(times) - 1:
                    dg = times[t] - times[t - 1]
                else:
                    dg = (times[t + 1] - times[t - 1]) / 2
                for i in range(L):
                    intCov[i, i] += dg * (traj[t, i] * (1 - traj[t, i]))
                    for j in range(i + 1, L):
                        intCov[i, j] += dg * (x[t, i, j] - traj[t, i] * traj[t, j])

        for i in range(L):
            for j in range(i + 1, L):
                intCov[j, i] = intCov[i, j]

        if self.timing:
            print('Computing covariance... took %.3fs' % (timer() - start))

        return intCov


    def mapMutToClade(self):
        """
        Maps a mutation to its clade, index 0 is otherMuts
        """
        mutToClade = {}
        for k, clade in enumerate(self.cladeMuts):
            for mut in clade:
                mutToClade[mut] = k + 1
        for mut in self.otherMuts:
            mutToClade[mut] = 0
        self.mutToClade = mutToClade
        return mutToClade


    def mapCladeToPeriod(self):
        cladeToPeriod = {0: 0}
        numExistingClades = 0
        for i, period in enumerate(self.periods):
            for k in range(period.numClades):
                cladeToPeriod[numExistingClades + k + 1] = i
            numExistingClades += period.numClades
        return cladeToPeriod


    def mapAncestorToDescendant(self):
        if self.ancestorIndices is None:
            return None
        ancestorToDescendant = {k: set() for k in self.ancestorIndices}
        numExistingClades = 0
        for i, period in enumerate(self.periods):
            ancestor = self.ancestorIndices[i]
            for k in range(period.numClades):
                ancestorToDescendant[ancestor].add(numExistingClades + k + 1)
            numExistingClades += period.numClades
        for i in range(len(self.ancestorIndices) - 1, 0, -1):
            curAncestor = self.ancestorIndices[i]
            preAncestor = self.ancestorIndices[i - 1]
            ancestorToDescendant[preAncestor] |= ancestorToDescendant[curAncestor]
        return ancestorToDescendant


    def getCooccurenceTraj(self, traj, mutToClade, cladeToPeriod):
        """
        Return cooccurence frequency trajectories of all pairs.
        """
        if self.timing:
            start = timer()
        T, L = self.T, self.L
        x = np.zeros((T, L, L), dtype=float)
        for i in range(L):
            for j in range(i + 1, L):
                if i in self.otherMuts and j in self.otherMuts and self.assumeCooperationAmongSharedMuts:
                    # Correlated
                    for t in range(T):
                        x[t, i, j] = min(traj[t, i], traj[t, j])
                elif i in self.otherMuts or j in self.otherMuts:
                    # Not correlated
                    for t in range(T):
                        x[t, i, j] = traj[t, i] * traj[t, j]
                elif self.isCorrelated(i, j, mutToClade, cladeToPeriod):
                    # Correlated
                    for t in range(T):
                        x[t, i, j] = min(traj[t, i], traj[t, j])
                else:
                    # Anti-correlated
                    for t in range(T):
                        x[t, i, j] = max(0, traj[t, i] + traj[t, j] - 1)
                for t in range(T):
                    x[t, j, i] = x[t, i, j]
        if self.timing:
            print('Computing cooccurence freq... took %.3fs' % (timer() - start))
        return x


    def isCorrelated(self, mut_i, mut_j, mutToClade, cladeToPeriod):
        """
        Return if mutation i and mutation j are correlated (linked), e.g., on the same background.
        """
        clade_i, clade_j = mutToClade[mut_i], mutToClade[mut_j]
        period_i, period_j = cladeToPeriod[clade_i], cladeToPeriod[clade_j]
        # In the same competition period, then both mutations must belong to the same clade.
        if period_i == period_j:
            return clade_i == clade_j
        # In different periods, then the former one must be an ancestor clade.
        if period_i < period_j:
            # return clade_i in self.ancestorIndices and np.mean([self.getCooperationScore(self.cladeFreqWithAncestor[:, clade_j], self.traj[:, l]) for l in self.cladeMuts[clade_i - 1]]) > -1e-6
            return clade_i in self.ancestorIndices and self.getCooperationScore(self.traj[:, mut_i], self.traj[:, mut_j]) > -1e-6
            # return clade_i in self.ancestorIndices
        else:
            # return clade_j in self.ancestorIndices and np.mean([self.getCooperationScore(self.cladeFreqWithAncestor[:, clade_i], self.traj[:, l]) for l in self.cladeMuts[clade_j - 1]]) > -1e-6
            return clade_j in self.ancestorIndices and self.getCooperationScore(self.traj[:, mut_i], self.traj[:, mut_j]) > -1e-6
            # return clade_j in self.ancestorIndices


    def inferSelection(self, intCov, mu=None, regularization=1):
        """
        1. Use reconstructed clade info to estimate covariance matrix.
        2. Apply MPL inference to infer selection coefficients of all mutations.
        """
        if mu is None and self.mu is None:
            print('Mutation rate not provided. Cannot infer selection with MPL.')
            return
        T, L, traj, times = self.T, self.L, self.traj, self.times

        if mu is None:
            if self.useEffectiveMu:
                if self.meanReadDepth is not None:
                    mu = 1 / (self.T * self.meanReadDepth)
                else:
                    meanReadDepth = np.mean(self.readDepths)
                    mu = 1 / (self.T * meanReadDepth)
            else:
                mu = self.mu

        # if useTrueCov:
        #     intCov = self.intCov
        # else:
        #     intCov = self.computeCovariance(self.processedTraj)
        D = MPL.computeD(traj, times, mu)
        selection = MPL.inferSelection(intCov, D, regularization * np.identity(L))
        return selection


    def computeFitness(self, traj, selection, initial_fitness=1, normalizeFitness=False):
        """
        Computes fitness of the population throughout the evolution.
        """
        T, L = traj.shape
        fitness = initial_fitness + np.sum(selection * traj, axis=1)
        if normalizeFitness:
            fitness /= np.max(fitness)
        return fitness


    def plotTraj(self, traj, name='Recovered traj'):
        PLOT.plotTrajComparisonSideBySide(self.traj, traj, times=self.times, annot=True, title=f'{name}, MAE=%.4f' % MAE(self.traj, traj))
        if self.debug and np.sum([MAE(self.traj[:, l], traj[:, l]) >= 0.01 for l in range(self.L)]) < 20:
            for l in range(self.L):
                if MAE(self.traj[:, l], traj[:, l]) >= 0.01:
                    print(f'{bcolors.WARNING}mutation {l} belong to clade {self.mutToClade[l]}')


    def plotRecoveredIntCov(self):
        PLOT.plotCovarianceComparison(self.intCov, self.recoveredIntCov, titles=['true', 'est', f'error, MAE=%.4f' % MAE(self.intCov, self.recoveredIntCov)])


    def plotRecoveredSelection(self, regularization, plotEstCov=True, plotTrueCov=True):
        variance = np.array([self.recoveredIntCov[l, l] for l in range(self.L)])
        PLOT.plotSelectionComparison(self.selection, self.selectionByTrueCov, self.recoveredSelection, 1 / variance, annot=True, ylabel='recovered selection')
        if plotEstCov:
            PLOT.plotSelectionComparison(self.selection, self.selectionByTrueCov, self.selectionByEstCov, 1 / variance, annot=True, ylabel='selection by est_cov')
        if plotTrueCov:
            PLOT.plotSelectionComparison(self.selection, self.selectionByTrueCov, self.selectionByTrueCov, 1 / variance, annot=True, ylabel='selection by true_cov')


    def plotRecoveredFitness(self, anotherFitnessToCompare=None, times_list=None, anotherLabel='by_est_cov'):
        if self.fitness_times is not None:
            times_list = [self.fitness_times, self.times, self.times, self.times]
        PLOT.plotFitnessComparison([self.fitness, self.fitnessByTrueCov, self.recoveredFitness, anotherFitnessToCompare], labels=['true', 'by_true_cov', 'by_recovered_cov', anotherLabel],  times_list=times_list, times=self.times)
