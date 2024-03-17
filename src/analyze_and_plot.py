import sys
import numpy as np
import pandas as pd
import math
from scipy import stats
from scipy import interpolate
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

import MPL
import reconstruct_clades as RC
import estimate_covariance as EC

COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'] * 10

# Standard color scheme

GREY_COLOR_RGB = (0.5, 0.5, 0.5)
GREY_COLOR_HEX = '#808080'

BKCOLOR  = '#252525'
LCOLOR   = '#969696'
C_BEN    = '#EB4025' #'#F16913'
C_BEN_LT = '#F08F78' #'#fdd0a2'
C_NEU    =  LCOLOR   #'#E8E8E8' # LCOLOR
C_NEU_LT = '#E8E8E8' #'#F0F0F0' #'#d9d9d9'
C_DEL    = '#3E8DCF' #'#604A7B'
C_DEL_LT = '#78B4E7' #'#dadaeb'

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

def cm2inch(x): return float(x)/2.54
SINGLE_COLUMN = cm2inch(8.5)
ONE_FIVE_COLUMN = cm2inch(11.4)
DOUBLE_COLUMN = cm2inch(17.4)

GOLDR        = (1.0 + np.sqrt(5)) / 2.0
TICKLENGTH   = 3
TICKPAD      = 3
AXWIDTH      = 0.4
SUBLABELS    = ['A', 'B', 'C', 'D', 'E', 'F']

# paper style

FONTFAMILY   = 'Arial'
SIZESUBLABEL = 8
SIZELABEL    = 6
SIZELEGEND   = 6
SIZETICK     = 6
SMALLSIZEDOT = 6.
SIZELINE     = 0.6

DEF_TICKPROPS_HEATMAP = {
    'length'    : TICKLENGTH,
    'width'     : AXWIDTH/2,
    'pad'       : TICKPAD,
    'axis'      : 'both',
    'which'     : 'both',
    'direction' : 'out',
    'colors'    : BKCOLOR,
    # 'labelsize' : SIZETICK,
    'bottom'    : False,
    'left'      : False,
    'top'       : False,
    'right'     : False,
    'pad'       : 2,
}

DEF_FIGPROPS = {
    'transparent' : True,
    'edgecolor'   : None,
    'dpi'         : 1000,
    # 'bbox_inches' : 'tight',
    'pad_inches'  : 0.05,
    'backend'     : 'PGF',
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

DEF_AXPROPS = {
    'linewidth' : AXWIDTH,
    'linestyle' : '-',
    'color'     : BKCOLOR
}

DEF_DXDX_MARGINPROPS = {
    'left': 0.02, 
    'right': 0.98,
    'top': 0.98,
    'bottom': 0.02,
}

def resetPlottingParams():
    PARAMS = {'text.usetex': False, 'mathtext.fontset': 'stixsans', 'mathtext.default': 'regular', 'pdf.fonttype': 42, 'ps.fonttype': 42}
    plt.rcParams.update(matplotlib.rcParamsDefault)
    plt.rcParams.update(PARAMS)


resetPlottingParams()


def hexToRGB(hex, normalize=True):
    hex = hex.lstrip('#')
    rgb = tuple(int(hex[i:i+2], 16) for i in (0, 2, 4))
    if normalize:
        rgb = tuple([_ / 256 for _ in rgb])
    return rgb


def MAE(a, b):
    return np.mean(np.absolute(a - b))


def getEmergenceTimes(traj, thPresent=0.01):
    """
    Computes emergence time of all mutations.
    """
    T, L = traj.shape
    emergenceTimes = np.full(L, -1, dtype=int)
    for l in range(L):
        for t in range(T):
            if traj[t, l] > thPresent:
                emergenceTimes[l] = t
                break
    return emergenceTimes


def getDominantGenotypeTrajectories(dic, T=1001, threshold=0.01, totalPopulation=1000):
    sVec, nVec = dic['sVec'], dic['nVec']
    pop = []
    for t in range(T):
        pop.append(np.sum(nVec[t]))
    seqOnceEmerged = []
    # pick out dominant genotypes
    for t in range(T):
        for s in range(len(nVec[t])):
            k = 0
            for k in range(len(seqOnceEmerged)):
                if np.array_equal(sVec[t][s], seqOnceEmerged[k]):
                    k = -1
                    break
            if nVec[t][s] / totalPopulation >= threshold and k >= 0:
                seqOnceEmerged.append(sVec[t][s])

    freqOnceEmerged = np.zeros((len(seqOnceEmerged), T), dtype=float)
    for t in range(T):
        for k in range(len(seqOnceEmerged)):
            for s in range(len(nVec[t])):
                if np.array_equal(sVec[t][s], seqOnceEmerged[k]):
                    freqOnceEmerged[k, t] = nVec[t][s] / pop[t]

    return seqOnceEmerged, freqOnceEmerged, pop


def getCladeMuts(seqOnceEmerged, freqOnceEmerged):
    T = len(freqOnceEmerged[0])
    seqToFreq = {seqToString(seqOnceEmerged[i]): freqOnceEmerged[i] for i in range(len(seqOnceEmerged))}
    cladeSeqs = [[seqOnceEmerged[0]]]
    for seq in seqOnceEmerged[1:]:
        assigned = False
        for c in range(len(cladeSeqs)):
            preSeq = cladeSeqs[c][-1]
            if hamming(seq, preSeq) == 1:
                cladeSeqs[c].append(seq)
                assigned = True
                break
        if assigned == False:
            cladeSeqs.append([seq])

    cladeFreq = np.zeros((len(cladeSeqs), T), dtype=float)
    for c in range(len(cladeSeqs)):
        for seq in cladeSeqs[c]:
            cladeFreq[c] += seqToFreq[seqToString(seq)]
    return cladeSeqs, cladeFreq


def seqToString(seq):
    s = ""
    for i in seq:
        s += str(int(i))
    return s


def getCladeMutsWildtypeIsBackground(seqOnceEmerged, freqOnceEmerged):
    T = len(freqOnceEmerged[0])
    seqToFreq = {seqToString(seqOnceEmerged[i]): freqOnceEmerged[i] for i in range(len(seqOnceEmerged))}
    cladeSeqs = [[seqOnceEmerged[0]]]
    for seq in seqOnceEmerged[1:]:
        assigned = False
        for c in range(len(cladeSeqs)):
            preSeq = cladeSeqs[c][-1]
            if hamming(seq, preSeq) == 1:
                cladeSeqs[c].append(seq)
                assigned = True
                break
        if assigned == False:
            cladeSeqs.append([seq])

    cladeFreq = np.zeros((len(cladeSeqs), T), dtype=float)
    for c in range(len(cladeSeqs)):
        for seq in cladeSeqs[c]:
            if np.any(seq):
                cladeFreq[c] += seqToFreq[seqToString(seq)]
    return cladeSeqs, cladeFreq


def getCladeMutsWinnerIsNew(dic, seqOnceEmerged, freqOnceEmerged, verbose=False):
    traj, nVec, sVec = dic['traj'], dic['nVec'], dic['sVec']
    statusChangeTimes = getStatusChangeTimes(traj)
    print(statusChangeTimes.shape)
    seqEmergeTimes = getSequenceEmergenceTimes(nVec, sVec)
    numTimes = len(freqOnceEmerged[0])
    seqToFreq = {seqToString(seqOnceEmerged[i]): freqOnceEmerged[i] for i in range(len(seqOnceEmerged))}
    clade = [[seqOnceEmerged[0]]]
    for seq in seqOnceEmerged[1:]:
        assigned = False
        for c in range(len(clade)):
            preSeq = clade[c][-1]
            if hamming(seq, preSeq) == 1:
                clade[c].append(seq)
                assigned = True
                break
        if assigned == False:
            # a seq has more than two sites different from last emerging seq, a new clade needs to be declared!
            freq = computeCladeFreq(clade, seqToFreq, seqEmergeTimes[seqToString(seq)])
            # Compare new mut with latest genotype of currently dominant clade
            # k = np.argmax(freq)
            # Compare new mut with latest genotype of latest clade
            k = -1
            emgSeq = seqEmergeTimes[seqToString(seq)]
            preSeq = clade[k][-1]
            preMutList = [i for i in range(len(preSeq)) if preSeq[i] == 1 and seq[i] == 0]
            mutList = [i for i in range(len(seq)) if seq[i] == 1 and preSeq[i] == 0]
            if verbose == True:
                print(f'\tsignature mutation for preSeq {preMutList}')
            preMut = preMutList[0]
            if verbose == True:
                print(f'\tsignature mutation for seq {mutList}')
            mut = mutList[0]
            # preEmg, preFix, preExt = statusChangeTimes[preMut][0], statusChangeTimes[preMut][1], statusChangeTimes[preMut][2]
            # emg, fix, ext  = statusChangeTimes[mut][0], statusChangeTimes[mut][1], statusChangeTimes[mut][2]
            preEmg, preFix, preExt = statusChangeTimes[preMut]
            emg, fix, ext  = statusChangeTimes[mut]
            start = np.min([i for i in [preFix, fix, preExt, ext, numTimes] if i > 0])
            end = numTimes
            if start == numTimes:
                start = emgSeq
            preDom, dom = np.sum(traj[start:end, preMutList[-1]]), np.sum(traj[start:end, mutList[-1]])
            if (ext == -1 and preExt > 0) or (fix > 0 and preFix == -1) or (ext > 0 and preExt > 0 and ext > preExt) or (start > emgSeq and dom > preDom) or (start == emgSeq and dom < preDom):
                clade.append([seq])
            elif (preExt == -1 and ext > 0) or (preFix > 0 and fix == -1) or (ext > 0 and preExt > 0 and preExt > ext) or (start > emgSeq and preDom < dom) or (start == emgSeq and preDom > dom):
                clade[k][-1] = seq
                clade.append([preSeq])
            else:
                print(f"Unresolved new mutation at time {seqEmergeTimes[seqToString(seq)]}.")
                print(f'mut    {mut} emerge {emg} fix {fix} extinct {ext}')
                print(f'preMut {preMut} emerge {preEmg} fix {preFix} extinct {preExt}')
            if verbose == True:
                print()

    freqClade = np.zeros((len(clade), numTimes), dtype = float)
    for c in range(len(clade)):
        for seq in clade[c]:
            freqClade[c] += seqToFreq[seqToString(seq)]
    return clade, freqClade


def getStatusChangeTimes(traj, thEmerge=0.001, duration=10):
    """
    return times in the format of [(emerge, fixed, extinct), ...]
    thEmerge and 1-thEmerge are respectively the sign for extinction and fixation.
    Both extinction and fixation are evaluated with mean frequency over a time window of size "duration"
    """
    T, L = traj.shape
    times = np.zeros((L, 3), dtype = int)
    for l in range(L):
        emerge = -1
        for t in range(T):
            if np.min(traj[t:min(t + duration, T), l]) >= thEmerge:
                emerge = t
                break
        extinct, fixed = -1, -1
        # for t in range(emerge, T):
        #     if np.max(traj[t:min(t + duration, T), l]) < thEmerge:
        #         extinct = t
        #         break
        if np.mean(traj[T - 1 - duration : T, l]) < thEmerge:
            for t in range(T - 1, emerge - 1, -1):
                if np.mean(traj[t - duration : t, l]) >= thEmerge:
                    extinct = t
                    break
        elif np.mean(traj[T - 1 - duration : T, l]) > 1 - thEmerge:
            for t in range(emerge, T):
                if np.mean(traj[t:min(t + duration, T), l]) > 1 - thEmerge:
                    fixed = t
                    break
        times[l] = emerge, fixed, extinct
    return times


def getSequenceEmergenceTimes(nVec, sVec):
    T = len(sVec)
    emerge = {}
    for t in range(T):
        for s in range(len(nVec[t])):
            if nVec[t][s] > 0 and seqToString(sVec[t][s]) not in emerge:
                emerge[seqToString(sVec[t][s])] = t
    return emerge


def computeCladeFreq(clade, dic_seq_freq, t):
    freq = [np.sum([dic_seq_freq[seqToString(seq)][t] for seq in clade[k]]) for k in range(len(clade))]
    return freq


def hamming(seq1, seq2):
    # return hamming distance of seq1 and seq2
    hamming = 0
    for i in range(len(seq1)):
        if not seq1[i] == seq2[i]:
            hamming += 1

    return hamming


def plotTraj(traj, times=None, linewidth=None, alphas=None, alpha=1, linestyle='solid', colors=None, labels=None, ylim=(-0.03, 1.03), figsize=(10, 3.3), fontsize=15, title='allele frequency trajectories', annot=False, annotTexts=None, annotXs=None, annotYs=None, plotShow=True, plotFigure=True, returnFig=False, plotLegend=False, scatter=False, scatterOnly=False, s=0.5, marker='.', markersize=7, marginprops=None, save_file=None):
    T, L = traj.shape
    if times is None:
        times = np.arange(0, T)
    if alphas is None:
        alphas = np.full((L), alpha)
    if colors is None:
        colors = COLORS
    if plotFigure:
        fig = plt.figure(figsize=figsize)
    if annot:
        # Avoid annotation overlapping
        bucketToSite = {}
        bucketWidth = (times[-1] - times[0]) / (4 * L)
        if annotTexts is None:
            annotTexts = np.arange(0, L)
    for l in range(L):
        if scatterOnly:
            plt.scatter(times, traj[:, l], color=colors[l%len(colors)], alpha=alphas[l], s=s)
        elif scatter:
            plt.plot(times, traj[:, l], color=colors[l%len(colors)], alpha=alphas[l], marker=marker, markersize=markersize, linestyle=linestyle)
        else:
            plt.plot(times, traj[:, l], color=colors[l%len(colors)], linewidth=linewidth, alpha=alphas[l], label=labels[l] if labels is not None else None, linestyle=linestyle)
        if annot:
            if annotXs is not None:
                x = annotXs[l]
                if annotYs is not None:
                    y = annotYs[l]
                else:
                    i = np.argmin(np.absolute(times - x))
                    y = traj[i, l] + 0.05
            else:
                variance = np.zeros(T, dtype=float)
                for t in range(1, T - 1):
                    variance[t] = np.std(traj[t-1:t+2, l])
                t = np.argmax(variance)
                bucketIndex = int(times[t] / bucketWidth)
                while bucketIndex in bucketToSite:
                    bucketIndex += 1
                time = bucketIndex * bucketWidth
                t = np.argmin(np.absolute(times - time))
                bucketToSite[bucketIndex] = l
                x, y = times[t], traj[t, l] + 0.05
            plt.text(x, y, f'{annotTexts[l]}', fontsize=fontsize, color=colors[l%len(colors)])

    set_ticks_labels_axes()
    if marginprops is not None:
        plt.subplots_adjust(**marginprops)
    plt.ylim(ylim)
    plt.title(title, fontsize=fontsize)
    if labels is not None and plotLegend:
        plt.legend(fontsize=fontsize)
    if plotShow:
        plt.show()
    if plotFigure and save_file:
        fig.savefig(save_file, facecolor=fig.get_facecolor(), **DEF_FIGPROPS)
    if returnFig:
        return fig


def plotMatrix(mat, figsize=(4, 4), center=0, cbar=False, plotShow=True, xticklabels=None, yticklabels=None, rotation=30, title=None):
    plt.figure(figsize=figsize)
    sns.heatmap(mat, center=center, square=True, cbar=cbar,
                cmap=sns.diverging_palette(145, 300, as_cmap=True))
    if xticklabels is not None:
        plt.gca().set_xticklabels(xticklabels, rotation=rotation)
    if yticklabels is not None:
        plt.gca().set_yticklabels(yticklabels, rotation=0)
    if title is not None:
        plt.title(title)
    if plotShow:
        plt.show()


def plotCandidateInterpolations(freqs, times, candidateInterpolations, compareInterpolation, labels=None, colors=None, compareLabel='linear interpolation', compareColor='black', alpha=0.5, figsize=(10, 3.3), fontsize=12, title=None, plotShow=True):
    if plotShow:
        plt.figure(figsize=figsize)
    if labels is None:
        labels = []
        for (emergenceTime, duration), f in candidateInterpolations.items():
            labels.append(f'duration={duration}')
    if colors is None:
        colors = COLORS

    times_intpl = range(times[0], times[-1] + 1)
    i = 0
    for (emergenceTime, duration), f in candidateInterpolations.items():
        freqs_intpl = [f(t) for t in times_intpl]
        plt.plot(times_intpl, freqs_intpl, alpha=alpha, label=labels[i], color=colors[i%len(colors)])
        i += 1

    freqs_intpl = [compareInterpolation(t) for t in times_intpl]
    plt.plot(times_intpl, freqs_intpl, alpha=alpha, label=compareLabel, color=compareColor)

    plt.scatter(times, freqs, color='red')
    plt.legend(fontsize=fontsize * 0.8)
    if title is not None:
        plt.title(title, fontsize=fontsize)
    if plotShow:
        plt.show()


def plotGenotypeTraj(freqOnceEmerged, times=None, alphas=None, figsize=(10, 3.3), fontsize=15, title='genotype frequency trajectories', plotShow=True):
    G, T = freqOnceEmerged.shape
    if times is None:
        times = np.arange(0, T)
    if alphas is None:
        alphas = np.full((G), 1)
    if plotShow:
        plt.figure(figsize=figsize)
    for g in range(G):
        plt.plot(times, freqOnceEmerged[g], color=COLORS[g%len(COLORS)], alpha=alphas[g])

    plt.ylim(-0.03, 1.03)
    plt.title(title, fontsize=fontsize)
    if plotShow:
        plt.show()


def plotTrajComparison(traj, estTraj, times=None, figsize=(10, 3.3), fontsize=15, title='allele frequency trajectories', annot=False, asSubplot=False, xlabel=None, ylabel=None, xticks=True, yticks=True):
    T, L = traj.shape
    if times is None:
        times = np.arange(0, T)
    if not asSubplot:
        plt.figure(figsize=figsize)
    if annot:
        # Avoid annotation overlapping
        bucketToSite = {}
        bucketWidth = (times[-1] - times[0]) / (4 * L)
    for l in range(L):
        if MAE(traj[:, l], estTraj[:, l]) < 0.01:
            alpha = 0.1
        else:
            alpha = 1
        plt.scatter(times, traj[:, l], color=COLORS[l], alpha=alpha, marker='|', s=1)
        plt.plot(times, estTraj[:, l], color=COLORS[l], linestyle='dashed', alpha=alpha)
        if annot:
            variance = np.zeros(T, dtype=float)
            for t in range(1, T - 1):
                variance[t] = np.std(traj[t-1:t+2, l])
            t = np.argmax(variance)
            bucketIndex = int(times[t] / bucketWidth)
            while bucketIndex in bucketToSite:
                bucketIndex += 1
            time = bucketIndex * bucketWidth
            t = np.argmin(np.absolute(times - time))
            plt.text(times[t], traj[t, l] + 0.05, f'{l}', fontsize=fontsize, color=COLORS[l], alpha=alpha)
            bucketToSite[bucketIndex] = l

    plt.ylim(-0.03, 1.03)
    plt.title(title, fontsize=fontsize)
    if not yticks:
        plt.yticks(ticks=[], labels=[])
    if not xticks:
        plt.xticks(ticks=[], labels=[])
    plt.ylabel(ylabel, fontsize=fontsize)
    plt.ylabel(xlabel, fontsize=fontsize)
    if not asSubplot:
        plt.show()


def plotTrajComparisonSideBySide(traj, estTraj, times=None, figsize=(10, 8), fontsize=15, title='allele frequency trajectories', annot=False, xlabel=None, ylabel=None):
    T, L = traj.shape
    if times is None:
        times = np.arange(0, T)
    fig, axes = plt.subplots(2, 1, figsize=figsize)
    if L > 1000:
        annot = False
    if annot:
        # Avoid annotation overlapping
        bucketToSite = {}
        bucketWidth = (times[-1] - times[0]) / (4 * L)
    for i, data in enumerate([traj, estTraj]):
        plt.sca(axes[i])
        for l in range(L):
            if MAE(traj[:, l], estTraj[:, l]) < min(0.1 * np.mean(traj[:, l]), 0.01):
                alpha = 0.1
            else:
                alpha = 1
            plt.plot(times, data[:, l], color=COLORS[l % len(COLORS)], alpha=alpha)
            if annot:
                variance = np.zeros(T, dtype=float)
                for t in range(1, T - 1):
                    variance[t] = np.std(traj[t-1:t+2, l])
                t = np.argmax(variance)
                bucketIndex = int(times[t] / bucketWidth)
                while bucketIndex in bucketToSite:
                    bucketIndex += 1
                time = bucketIndex * bucketWidth
                t = np.argmin(np.absolute(times - time))
                plt.text(times[t], traj[t, l] + 0.05, f'{l}', fontsize=fontsize, color=COLORS[l % len(COLORS)], alpha=alpha)
                bucketToSite[bucketIndex] = l
        plt.ylim(-0.03, 1.03)
        if i == 1:
            plt.title(title, fontsize=fontsize)
        else:
            plt.title('true allele freq traj', fontsize=fontsize)
        plt.ylabel(ylabel, fontsize=fontsize)
        plt.ylabel(xlabel, fontsize=fontsize)
    plt.show()


def plotCovarianceComparison(trueCov, estCov, cbar=False, as_subplot=False, figsize=(10, 3.3), fontsize=15, titles=['true', 'est', 'error']):
    """Plots a figure comparing true and estimated integrated covariance matrix, as well as their difference."""

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    cov_list = [trueCov, estCov, estCov - trueCov]
    vmin = min(np.min(cov_list[0]), np.min(cov_list[1]))
    vmax = max(np.max(cov_list[0]), np.max(cov_list[1]))
    cbar_ax = fig.add_axes(rect=[.128, .175, .773, .02])  # color bar on the bottom, rect=[left, bottom, width, height]
    cbar_ticks = np.arange(int(vmin/5)*5, int(vmax/5)*5, 50)
    cbar_ticks -= cbar_ticks[np.argmin(np.abs(cbar_ticks))]
    for i, cov in enumerate(cov_list):
        plt.sca(axes[i])
        ax = plt.gca()
        plot_cbar = (i == 0) and cbar
        sns.heatmap(cov, center=0, vmin=vmin - 5, vmax=vmax + 5, cmap=sns.diverging_palette(145, 300, as_cmap=True), square=True, cbar=plot_cbar, cbar_ax=cbar_ax if plot_cbar else None, cbar_kws=dict(ticks=cbar_ticks, orientation="horizontal"))
        if titles[i] == 'error':
            title = titles[i] + f' MAE=%.2f' % MAE(trueCov, estCov)
        else:
            title = titles[i]
        plt.title(title, fontsize=fontsize)
        if plot_cbar and cbar_ax:
            cbar_ax.tick_params(labelsize=SIZELABEL)
        ax.tick_params(**DEF_TICKPROPS_HEATMAP, labelsize=10)

    plt.subplots_adjust(bottom=0.25)
    plt.show()


def plotSelectionComparison(trueSelection, selectionByTrueCov, estSelection, error, figsize=(10, 3.3), fontsize=15, title='', ylabel='inferred selection', annot=False):
    L = len(trueSelection)
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    minS, maxS = min(np.min(trueSelection), np.min(estSelection)), max(np.max(trueSelection), np.max(estSelection))
    annot_offset = (maxS - minS) / 60
    xlim = ylim = (minS - 0.01, maxS + 0.01)
    error /= np.max(error)
    error *= 0.3 * (maxS - minS)
    # sortedIndices = np.argsort(trueSelection)
    xlabels = ['true selection', 'selection inferred by true cov']
    xs = [trueSelection, selectionByTrueCov]
    for i, x in enumerate(xs):
        plt.sca(axes[i])
        at_left = i == 0
        for l in range(L):
            plt.scatter([x[l]], [estSelection[l]], color=COLORS[l%len(COLORS)])
            plt.errorbar(x[l], estSelection[l], yerr=error[l], color=COLORS[l%len(COLORS)], ecolor='grey', elinewidth=1)
            if annot:
                plt.text(x[l], estSelection[l] + annot_offset, f'{l}', fontsize=fontsize, color=COLORS[l%len(COLORS)])
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.plot([minS, maxS], [minS, maxS], linestyle='dashed', color='grey')
        plt.xlabel(xlabels[i], fontsize=fontsize)
        if at_left:
            plt.ylabel(ylabel, fontsize=fontsize)
        spearmanr, _ = stats.spearmanr(x, estSelection)
        plt.title(f'MAE=%.4f, spearmanr=%.2f' % (MAE(x, estSelection), spearmanr))
    plt.show()


def plotCovarianceAndSelectionComparison(trueCov, estCov, trueSelection, selectionByTrueCov, estSelection, error=None, evaluate_true_cov=False, compare_eigenvalues=False, plot_hist=False, cbar=False, ylabel='inferred selection', annot=False, figsize=(20, 3.3), fontsize=12, suptitle_y=1.025, titles=['true', 'est', 'error'], suptitle=""):
    fig, axes = plt.subplots(1, 5 + compare_eigenvalues, figsize=figsize)
    L = len(trueSelection)
    cov_list = [trueCov, estCov, estCov - trueCov]
    vmin = min(np.min(cov_list[0]), np.min(cov_list[1]))
    vmax = max(np.max(cov_list[0]), np.max(cov_list[1]))
    cbar_ax = fig.add_axes(rect=[.128, .175, .773, .02])  # color bar on the bottom, rect=[left, bottom, width, height]
    cbar_ticks = np.arange(int(vmin/5)*5, int(vmax/5)*5, 50)
    cbar_ticks -= cbar_ticks[np.argmin(np.abs(cbar_ticks))]
    for i, cov in enumerate(cov_list):
        plt.sca(axes[i])
        ax = plt.gca()
        plot_cbar = (i == 0) and cbar
        sns.heatmap(cov, center=0, vmin=vmin - 5, vmax=vmax + 5, cmap=sns.diverging_palette(145, 300, as_cmap=True), square=True, cbar=plot_cbar, cbar_ax=cbar_ax if plot_cbar else None, cbar_kws=dict(ticks=cbar_ticks, orientation="horizontal"))
        if titles[i] == 'error':
            title = titles[i] + f' MAE=%.2f max_abs_diff=%.2f' % (MAE(trueCov, estCov), np.max(np.abs(trueCov - estCov)))
        else:
            title = titles[i]
        plt.title(title, fontsize=fontsize)
        if plot_cbar and cbar_ax:
            cbar_ax.tick_params(labelsize=SIZELABEL)
        ax.tick_params(**DEF_TICKPROPS_HEATMAP, labelsize=10)

    if compare_eigenvalues:
        plt.sca(axes[len(cov_list)])
        eigenvalues_true, eigenvectors_true = np.linalg.eig(trueCov)
        eigenvalues_est, eigenvectors_est = np.linalg.eig(estCov)
        num_bins = len(trueCov)
        vmin = min(np.min(eigenvalues_true), np.min(eigenvalues_est))
        vmax = max(np.max(eigenvalues_true), np.max(eigenvalues_est))
        if plot_hist:
            interval = (vmax - vmin) / num_bins
            bins = np.arange(vmin, vmax + interval, interval)
            plt.hist(eigenvalues_true, bins=bins, label='true', alpha=0.6)
            plt.hist(eigenvalues_est, bins=bins, label='est', alpha=0.6)
            plt.legend()
            plt.ylabel('Count')
            plt.xlabel('Eigenvalues')
        else:
            xlim = ylim = (vmin / 2, vmax * 2)
            plt.scatter(eigenvalues_true, eigenvalues_est)
            plt.xlabel(f'Eigenvalue of trueCov')
            plt.ylabel(f'Eigenvalue of estCov')
            plt.plot([vmin, vmax], [vmin, vmax], linestyle='dashed', color='grey')
            plt.yscale('log')
            plt.xscale('log')
            plt.xlim(xlim)
            plt.ylim(ylim)


    minS, maxS = min(np.min(trueSelection), np.min(estSelection)), max(np.max(trueSelection), np.max(estSelection))
    annot_offset = (maxS - minS) / 60
    xlim = ylim = (minS - 0.01, maxS + 0.01)

    if evaluate_true_cov:
        xlabels = ['true selection', 'true selection']
        ylabels = [ylabel, 'selection inferred by true cov']
        xs = [trueSelection, trueSelection]
        ys = [estSelection, selectionByTrueCov]
    else:
        xlabels = ['true selection', 'selection inferred by true cov']
        ylabels = [ylabel, ylabel]
        xs = [trueSelection, selectionByTrueCov]
        ys = [estSelection, estSelection]
    for i, (x, y) in enumerate(zip(xs, ys)):
        plt.sca(axes[i + len(cov_list) + compare_eigenvalues])
        at_left = i == 0
        for l in range(L):
            plt.scatter([x[l]], [y[l]], color=COLORS[l%len(COLORS)])
            if annot:
                plt.text(x[l], y[l] + annot_offset, f'{l}', fontsize=fontsize, color=COLORS[l%len(COLORS)])
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.plot([minS, maxS], [minS, maxS], linestyle='dashed', color='grey')
        plt.xlabel(xlabels[i], fontsize=fontsize)
        if at_left or evaluate_true_cov:
            plt.ylabel(ylabels[i], fontsize=fontsize)
        spearmanr, _ = stats.spearmanr(x, y)
        plt.title(f'MAE=%.4f, spearmanr=%.2f' % (MAE(x, y), spearmanr), fontsize=fontsize)

    if suptitle:
        plt.suptitle(suptitle, fontsize=fontsize, y=suptitle_y)
    plt.subplots_adjust(wspace=0.3)
    plt.show()


def plotFitnessComparison(fitness_list, times_list=None, times=None, figsize=(10, 3.3), fontsize=15, title='fitness trajectories', asSubplot=False, labels=['true', 'by_true_cov', 'by_recovered_cov', 'by_est_cov'], colors=[3, 0, 1, 4, 2], plotLegend=True, plotXticks=True, plotYticks=True, ylim=None):
    T = len(fitness_list[0])
    if times is None:
        times = np.arange(0, T)
    if not asSubplot:
        plt.figure(figsize=figsize)
    for i, fitness in enumerate(fitness_list):
        if times_list is not None:
            times = times_list[i]
        if fitness is not None:
            if len(fitness_list) <= len(colors):
                color = COLORS[colors[i]]
            else:
                color = COLORS[i % len(COLORS)]
            plt.plot(times, fitness, label=labels[i], color=color)
    if ylim is not None:
        plt.ylim(ylim)
    if not plotXticks:
        plt.xticks(ticks=[], labels=[])
    if not plotYticks:
        plt.yticks(ticks=[], labels=[])
    if plotLegend:
        plt.legend(fontsize=fontsize*0.6)
    plt.title(title, fontsize=fontsize)
    if not asSubplot:
        plt.show()


def plotRegularizationComparison(regToRes, figsize=(10, 2.5), fontsize=15, alpha=0.5):
    regList = sorted(list(regToRes.keys()))
    resList = [regToRes[reg] for reg in regList]
    metricsList = []
    for res in resList:
        spearmanrOfSelectionByTrueCov, _ = stats.spearmanr(res[0], res[1])
        spearmanrOfSelectionByEstCov, _ = stats.spearmanr(res[0], res[2])
        MAEOfSelectionByTrueCov = MAE(res[0], res[1])
        MAEOfSelectionByEstCov = MAE(res[0], res[2])
        MAEOfFitnessByTrueCov = MAE(res[3], res[4])
        MAEOfFitnessByEstCov = MAE(res[3], res[5])
        metricsList.append((spearmanrOfSelectionByTrueCov, spearmanrOfSelectionByEstCov, MAEOfSelectionByTrueCov, MAEOfSelectionByEstCov, MAEOfFitnessByTrueCov, MAEOfFitnessByEstCov))
    metricsList = np.array(metricsList)
    labels = ['true_cov', 'recovered']
    markers = ['*', 'v']
    colors = [COLORS[0], COLORS[1]]
    titles = ['Spearmanr_selection', 'MAE_selection', 'MAE_fitness']

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    for i in range(len(metricsList[0])):
        plt.sca(axes[i // 2])
        plt.plot(regList, metricsList[:, i], label=labels[i % 2], marker=markers[i % 2], color=colors[i % 2], alpha=alpha)
        plt.legend(fontsize=fontsize / 1.5)
        if i % 2 == 0:
            plt.title(titles[i // 2], fontsize=fontsize)
            plt.xlabel('regularization', fontsize=fontsize)
            plt.xscale('log')
    plt.show()


def plotWindowComparison(winToRes, figsize=(10, 2.5), fontsize=15, alpha=0.5):
    winList = sorted(list(winToRes.keys()))
    resList = [winToRes[reg] for reg in winList]
    metricsList = []
    for res in resList:
        spearmanrOfSelectionByTrueCov, _ = stats.spearmanr(res[0], res[1])
        spearmanrOfSelectionByEstCov, _ = stats.spearmanr(res[0], res[2])
        MAEOfSelectionByTrueCov = MAE(res[0], res[1])
        MAEOfSelectionByEstCov = MAE(res[0], res[2])
        MAEOfFitnessByTrueCov = MAE(res[3], res[4])
        MAEOfFitnessByEstCov = MAE(res[3], res[5])
        metricsList.append((spearmanrOfSelectionByTrueCov, spearmanrOfSelectionByEstCov, MAEOfSelectionByTrueCov, MAEOfSelectionByEstCov, MAEOfFitnessByTrueCov, MAEOfFitnessByEstCov))
    metricsList = np.array(metricsList)
    labels = ['true_cov', 'est_cov']
    markers = ['*', 'v']
    colors = [COLORS[0], COLORS[2]]
    titles = ['Spearmanr_selection', 'MAE_selection', 'MAE_fitness']

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    for i in range(len(metricsList[0])):
        plt.sca(axes[i // 2])
        plt.plot(winList, metricsList[:, i], label=labels[i % 2], marker=markers[i % 2], color=colors[i % 2], alpha=alpha)
        plt.legend(fontsize=fontsize / 1.5)
        if i % 2 == 0:
            plt.title(titles[i // 2], fontsize=fontsize)
            plt.xlabel('window', fontsize=fontsize)
    plt.show()


def plotCladeFreq(cladeFreq, plotLegend=True, plotShow=True, fontsize=15, alpha=1, start_from_clade_1=True, title='Clade frequencey'):
    for c in range(len(cladeFreq)):
        plt.scatter(range(len(cladeFreq[c])), cladeFreq[c], s=3, label=f"clade {c + 1}" if start_from_clade_1 else f"clade {c}", alpha=alpha)
    plt.ylim(-0.03, 1.03)
    plt.title(title, fontsize=fontsize)
    if plotLegend:
        plt.legend()
    if plotShow:
        plt.show()


def plotTotalCladeFreq(totalCladeFreq, times=None, cladeMuts=None, otherMuts=None, someMuts=None, colorAncestorFixedRed=False, thExtinct=0.01, traj=None, figsize=(10, 3.3), ylim=(-0.03, 1.03), fontsize=12, colors=['grey'] + COLORS, legendsize=10, alpha=0.3, linewidth=1, title=None, plotFigure=True, plotClade=True, plotLegend=True, plotShow=True):
    T, K = totalCladeFreq.shape
    if times is None:
        times = np.arange(0, T)
    if plotFigure:
        fig = plt.figure(figsize=figsize)
    if plotClade:
        plt.plot(times, totalCladeFreq[:, 0], label=f'Ancestor', color=colors[0], linestyle='solid', linewidth=linewidth)
    if otherMuts is not None:
        for l in otherMuts:
            if colorAncestorFixedRed:
                if traj[-1, l] < thExtinct:
                    plt.plot(times, traj[:, l], color=colors[0], linewidth=0.5 * linewidth, alpha=alpha)
                else:
                    plt.plot(times, traj[:, l], color=LTEE_ANCESTOR_FIXED_COLOR, linewidth=0.5 * linewidth, alpha=alpha)
            else:
                plt.plot(times, traj[:, l], color=colors[0], linewidth=0.5 * linewidth, alpha=alpha)
    # plt.scatter(times, totalCladeFreq[:, 0], s=3, label=f"Ancestor clade", color='grey')
    for c in range(1, K):
        if plotClade:
            plt.plot(times, totalCladeFreq[:, c], label=f'Clade {c}', color=colors[c], linestyle='solid', linewidth=linewidth)
        # plt.scatter(times, totalCladeFreq[:, c], s=3, label=f"clade {c}", color=COLORS[c - 1])
        if cladeMuts is not None and c - 1 < len(cladeMuts):
            for l in cladeMuts[c - 1]:
                if someMuts is None or l in someMuts:
                    plt.plot(times, traj[:, l], color=colors[c], linewidth=0.5 * linewidth, alpha=alpha)

    plt.ylim(ylim)
    if title is not None:
        plt.title(title, fontsize=fontsize)
    if plotLegend:
        plt.legend(fontsize=legendsize)
    if plotShow:
        plt.show()


def plotConstructedClades(cladeMuts, figsize=None):
    plt.figure(figsize=figsize or (len(cladeMuts) * 1.5, 1.5))
    cladeSizes = [len(clade) for clade in cladeMuts]
    plt.scatter(range(len(cladeSizes)), cladeSizes, alpha=0.5)
    plt.xticks([int(i) for i in range(len(cladeSizes))])
    plt.xlabel("clade NO. (clade 0 is shared)")
    plt.ylabel("clade size")
    plt.show()


def plotCladeAndMutFreq(majorCladePolyMuts, cladeFreq, traj, times=None, otherMuts=None, figsize=(10, 3.3), fontsize=15, alpha=0.3, linewidth=1, plotFigure=True, plotLegend=True, save_file=None):

    T, L = traj.shape
    T, numClades = cladeFreq.shape
    times = times if times is not None else np.arange(0, T)
    if plotFigure:
        fig = plt.figure(figsize=figsize)
    # cladeTraj = cladeFreq.T
    # combined_list = [(np.copy(clades_select[c]), np.copy(cladeTraj[c])) for c in range(numClades)]
    # sorted_list = sorted(combined_list, key = lambda x : - x[1][-1])
    # for c in range(numClades):
    #     clades_select[c], cladeTraj[c] = sorted_list[c]
    #     clades_select[c] = list(clades_select[c]) # list is more convenient to append, remove
    # cladeFreq = cladeTraj.T
    for c in range(numClades):
        plt.plot(times, cladeFreq[:, c], label=f'clade {c + 1}', color=COLORS[c], linestyle='dashed', linewidth=1.5 * linewidth)
        for l in majorCladePolyMuts[c]:
            plt.plot(times, traj[:, l], color=COLORS[c], linewidth=0.5 * linewidth, alpha=alpha)
    plt.plot(times, 1 - np.sum(cladeFreq, axis=1), label=f'Ancestor clade', color='gray', linestyle='dashed', linewidth=1.5 * linewidth)
    if otherMuts is not None:
        for l in otherMuts:
            # Intentionally made them hard to spot on screen.
            plt.plot(times, traj[:, l], color='yellow', linewidth=0.5 * linewidth, alpha=alpha)
    if plotLegend:
        plt.legend(fontsize=fontsize / 1.5)
    plt.ylim(-0.03, 1.03)
    title = ';  '.join([f'clade {k + 1}: {len(majorCladePolyMuts[k])}' for k in range(numClades)])
    plt.title("infer  " + title + f"; Other: {L - np.sum([len(_) for _ in majorCladePolyMuts])}", fontsize=fontsize)
    plt.show()
    if plotFigure and save_file:
        fig.savefig(save_file, facecolor=fig.get_facecolor(), **DEF_FIGPROPS)


def plotCladeAndMutFreq_overview(period, colors=ALLELE_COLORS, ylim=(-0.03, 1.03), clade_index_offset=0, ancestor_clade_index=-1, otherMuts=None, figsize=(SINGLE_COLUMN / 3, SINGLE_COLUMN / 2), fontsize=SIZELABEL, legendsize=SIZELABEL / 1.5, alleleFreqAlpha=0.5, cladeFreqLinestyle='dashed', linewidth=SIZELINE, cladeFreqLinewidth=2*SIZELINE, title=None, plotFigure=True, returnFig=False, plotLegend=True, plotShow=True, marginprops=None, save_file=None):
    traj, times, cladeFreq, majorCladePolyMuts = period.traj, period.times, period.cladeFreq, period.cladeMuts
    T, L = traj.shape
    T, numClades = cladeFreq.shape
    if times is None:
        times = np.arange(0, T)

    if plotFigure:
        fig = plt.figure(figsize=figsize)

    for c in range(numClades):
        plt.plot(times, cladeFreq[:, c], color=colors[c + clade_index_offset], linestyle=cladeFreqLinestyle, linewidth=cladeFreqLinewidth)
        for l in majorCladePolyMuts[c]:
            plt.plot(times, traj[:, l], color=colors[c + clade_index_offset], linewidth=linewidth, alpha=alleleFreqAlpha)
    plt.plot(times, 1 - np.sum(cladeFreq, axis=1), color=GREY_COLOR_RGB if ancestor_clade_index==-1 else colors[ancestor_clade_index], linestyle=cladeFreqLinestyle, linewidth=cladeFreqLinewidth)

    if otherMuts is not None:
        for l in otherMuts:
            # Intentionally made them hard to spot on screen.
            plt.plot(times, traj[:, l], color='yellow', linewidth=linewidth, alpha=alleleFreqAlpha)

    plt.plot([times[1], times[1] + 0.0001], [-1, -1], linewidth=cladeFreqLinewidth, linestyle=cladeFreqLinestyle, color=GREY_COLOR_RGB, label='Clade')
    plt.plot([times[1], times[1] + 0.0001], [-1, -1], linewidth=linewidth, color=GREY_COLOR_RGB, alpha=alleleFreqAlpha, label='Allele')

    set_ticks_labels_axes()
    if marginprops is not None:
        plt.subplots_adjust(**marginprops)
    if plotLegend:
        plt.legend(fontsize=legendsize)
    plt.ylim(ylim)
    if title is not None:
        plt.title(title, fontsize=SIZELABEL)
    if plotShow:
        plt.show()
    if plotFigure and save_file:
        fig.savefig(save_file, facecolor=fig.get_facecolor(), **DEF_FIGPROPS)
    if returnFig:
        return fig


def plotCladeSitesTraj(traj, cladeMuts, times=None, numClades=2, thFixed=0.75, thExtinct=0.1, alpha=0.3):
    T, L = traj.shape
    times = times if times is not None else np.arange(0, T)
    numClades = min(numClades, len(cladeMuts) - 1)
    shared = cladeMuts[0]
    other = []
    for k in range(numClades + 1, len(cladeMuts)):
        other += cladeMuts[k]

    fig, axes = plt.subplots(1, numClades + 1, figsize=((numClades + 1) * 5, 2))
    plt.sca(axes[0])
    for l in shared:
        plt.plot(times, traj[:, l], color='yellow')
    plt.ylim(-0.03, 1.03)
    plotThreeLevelTraj(traj, other, ['red', 'grey', 'black'], times=times, alpha=alpha, thFixed=thFixed, thExtinct=thExtinct)
    plt.title(f'other sites')

    for n in range(1, numClades + 1):
        plt.sca(axes[n])
        plotThreeLevelTraj(traj, cladeMuts[n], ['red', 'grey', COLORS[n - 1]], times=times, alpha=alpha, thFixed=thFixed, thExtinct=thExtinct)
        plt.title(f'clade {n}')
    plt.show()


def plotThreeLevelTraj(traj, muts, colors, times=None, alpha=0.3, thFixed=0.75, thExtinct=0.1):
    T, L = traj.shape
    times = times if times is not None else np.arange(0, T)
    for l in muts:
        fixed = np.min([T] + [t for t in range(T) if traj[t, l] > thFixed])
        meanAfterFixed = np.mean(traj[fixed:T, l]) if fixed < T else 0
        extinct = np.max([0] + [t for t in range(T) if traj[t, l] > thExtinct])
        meanAfterExtinct = np.mean(traj[extinct:T, l])
        if fixed < min(T * 0.9, T - 1) and meanAfterFixed >= thFixed:
            plt.plot(times, traj[:, l], color=colors[0], alpha=0.3)
        elif meanAfterExtinct < thExtinct:
            plt.plot(times, traj[:, l], color=colors[1], alpha=0.3)
        else:
            plt.plot(times, traj[:, l], color=colors[2], alpha=0.3)
    plt.ylim(-0.03, 1.03)


def plotIncorporated(traj, cladeFreq, majorCladePolyMuts, incorporated, excludedAfterIncorporated, times=None, figsize=None, fontsize=15):
    """
    plotIncorporated
    """
    T, numClades = cladeFreq.shape
    T, L = traj.shape
    times = times if times is not None else np.arange(0, T)
    cladesToPlot = set([owner for i, owner in incorporated])
    cladesHasExcluded = set([owner for i, owner in excludedAfterIncorporated])
    cladesHasKept = set([owner for i, owner in incorporated if (i, owner) not in excludedAfterIncorporated])
    # for i, owner in incorporated:
    #     cladesToPlot.add(owner)
    numCladesToPlot = len(cladesHasExcluded) + len(cladesHasKept)
    if figsize is None:
        figsize = (10, 4 * numCladesToPlot)
    fig, axes = plt.subplots(numCladesToPlot, 1, figsize=figsize)
    n = 0
    for k in cladesToPlot:
        if k in cladesHasExcluded:
            if numCladesToPlot > 1:
                plt.sca(axes[n])
            n += 1
            plotCladeFreqAsBackground(cladeFreq, times=times)
            for i, owner in excludedAfterIncorporated:
                if owner == k:
                    plt.plot(times, traj[:, i], alpha=0.5, color=COLORS[k])
            plt.ylim(-0.03, 1.03)
            plt.title(f'clade {k + 1} incorporated but then excluded', fontsize=fontsize)

        if k in cladesHasKept:
            if numCladesToPlot > 1:
                plt.sca(axes[n])
            n += 1
            plotCladeFreqAsBackground(cladeFreq, times=times)
            for i, owner in incorporated:
                if owner == k:
                    if (i, owner) not in excludedAfterIncorporated:
                        plt.plot(times, traj[:, i], alpha=0.5, color=COLORS[k])
            plt.ylim(-0.03, 1.03)
            plt.title(f'clade {k + 1} incorporated and kept', fontsize=fontsize)
    plt.show()


def plotCladeFreqAsBackground(cladeFreq, times=None, onlyOneClade=None, linewidth=5, linestyle='dashed', alpha=0.5):
    T, numClades = cladeFreq.shape
    times = times if times is not None else np.arange(0, T)
    if onlyOneClade is None:
        for k in range(numClades):
            plt.plot(times, cladeFreq[:, k], color=COLORS[k], linewidth=linewidth, linestyle=linestyle, alpha=alpha)
    else:
        k = onlyOneClade
        plt.plot(times, cladeFreq[:, k], color=COLORS[k], linewidth=linewidth, linestyle=linestyle, alpha=alpha)


def plotMullerCladeFreq(mullerCladeFreq, mullerColors, times, figsize=(10, 3.3), plotFigure=True, plotShow=True, save_file=None):
    if plotFigure:
        fig = plt.figure(figsize=figsize)
    plt.stackplot(times, mullerCladeFreq.T, colors=mullerColors)
    if plotShow:
        plt.show()
    if plotFigure and save_file:
        fig.savefig(save_file, facecolor=fig.get_facecolor(), **DEF_FIGPROPS)



#########################################
# Plotting functions for test_clade_reconstruction_with_simulated_data.ipynb
#########################################

def plotTrajForSimulations(summaries, start_n=0, figsize=(20, 15), nCol=5, fontsize=15):
    nRow = len(summaries) // nCol + (len(summaries) % nCol > 0)
    fig, axes = plt.subplots(nRow, nCol, figsize=figsize)
    for n, res in enumerate(summaries):
        if res is None:
            continue
        plt.sca(axes[n//nCol, n%nCol])
        plotTrajComparison(res['true'][0], res['recovered'][0], asSubplot=True, annot=True, title=f'n={n+start_n}, MAE=%.3f'%(MAE(res['true'][0], res['recovered'][0])), xticks=n//nCol == nRow-1, yticks=n%nCol==0)
    plt.show()


def plotCovForSimulations(summaries, start_n=0, figsize=(20, 15), nCol=5, fontsize=15):
    nRow = len(summaries) // nCol + (len(summaries) % nCol > 0)
    fig, axes = plt.subplots(nRow, nCol, figsize=figsize)
    for n, res in enumerate(summaries):
        if res is None:
            continue
        plt.sca(axes[n//nCol, n%nCol])
        sns.heatmap(res['recovered'][1]-res['true'][1], center=0, square=True, cbar=False,
                    cmap=sns.diverging_palette(145, 300, as_cmap=True))
        plt.title(f"n={n+start_n}, reg={res['reg']}", fontsize=fontsize)
    plt.show()


def plotCov(cov, figsize=(4, 4), plotShow=True):
    plt.figure(figsize=figsize)
    sns.heatmap(cov, center=0, square=True, cbar=False,
                cmap=sns.diverging_palette(145, 300, as_cmap=True))
    if plotShow:
        plt.show()


def plotSegmentedIntCov(reconstruction, ylabel_x=-0.21, alpha=0.5, grid_line_offset=0, cbar_shrink=1, xtick_distance=40, plot_cbar=True, plot_yticks=True, plot_xticks=True, vmin=None, vmax=None, figsize=(SINGLE_COLUMN/2, SINGLE_COLUMN/2), plotShow=True, save_file=None):

    fig = plt.figure(figsize=figsize)
    ax = plt.gca()

    int_cov = reconstruction.recoveredIntCov
    clades = [reconstruction.otherMuts] + reconstruction.cladeMuts
    segmented_int_cov = segmentMatrix(int_cov, clades)[0]
    L = len(segmented_int_cov)

    heatmap = sns.heatmap(segmented_int_cov, center=0, vmin=vmin, vmax=vmax, cmap=CMAP, square=True, cbar=plot_cbar, cbar_kws={"shrink": cbar_shrink})

    if plot_cbar:
        cbar = heatmap.collections[0].colorbar
        cbar.ax.tick_params(**DEF_TICKPROPS_COLORBAR)

    if plot_yticks:
        ticks, ylabels, group_sizes = get_ticks_and_labels_for_clusterization(clades, name='Clade', note_size=True)
        plot_ticks_and_labels_for_clusterization(ticks, ylabels, group_sizes, ylabel_x=ylabel_x)
        ax.hlines([_ + grid_line_offset for _ in ticks], *ax.get_xlim(), color=GREY_COLOR_HEX, alpha=alpha, linewidth=SIZELINE * 1.2)
        ax.vlines([_ + grid_line_offset for _ in ticks], *ax.get_ylim(), color=GREY_COLOR_HEX, alpha=alpha, linewidth=SIZELINE * 1.2)

    if plot_xticks:
        xticklabels = np.arange(0, L + xtick_distance // 2, xtick_distance)
        xticks = [l for l in xticklabels]
        plt.xlabel('Locus index', fontsize=SIZELABEL)
    else:
        xticks, xticklabels = [], []

    set_ticks_labels_axes(yticks=[], yticklabels=[], xticks=xticks, xticklabels=xticklabels)

    if plotShow:
        plt.show()
    if save_file:
        fig.savefig(save_file, facecolor=fig.get_facecolor(), **DEF_FIGPROPS)


def plotSegmentedIntDxdx(reconstruction, normalize=False, segment_line_color=GREY_COLOR_HEX, ylabel_x=-0.21, alpha=0.5, grid_line_offset=0, linewidth=2 * SIZELINE, cbar_shrink=1, xtick_distance=4, plot_cbar=True, plot_yticks=True, plot_xticks=True, vmin=None, vmax=None, figsize=(SINGLE_COLUMN/2, SINGLE_COLUMN/2), plotFigure=True, returnFig=False, plotShow=True, save_file=None):

    if plotFigure:
        fig = plt.figure(figsize=figsize)
    ax = plt.gca()

    segmentedIntDxdx, groups = reconstruction.segmentedIntDxdx
    L = len(segmentedIntDxdx)
    for l in range(L):
        segmentedIntDxdx[l, l] = 0
    if normalize:
        segmentedIntDxdx = normalize_segmentedIntDxdx(segmentedIntDxdx)

    heatmap = sns.heatmap(segmentedIntDxdx, center=0, vmin=vmin, vmax=vmax, cmap=CMAP, square=True, cbar=plot_cbar, cbar_kws={"shrink": cbar_shrink})
    
    if plot_cbar:
        cbar = heatmap.collections[0].colorbar
        cbar.ax.tick_params(**DEF_TICKPROPS_COLORBAR)
    ticks, ylabels, group_sizes = get_ticks_and_labels_for_clusterization(groups, name='Group', note_size=True)
    
    if plot_yticks:
        plot_ticks_and_labels_for_clusterization(ticks, ylabels, group_sizes, ylabel_x=ylabel_x)
    ax.hlines([_ + grid_line_offset for _ in ticks], *ax.get_xlim(), color=segment_line_color, alpha=alpha, linewidth=linewidth)
    ax.vlines([_ + grid_line_offset for _ in ticks], *ax.get_ylim(), color=segment_line_color, alpha=alpha, linewidth=linewidth)

    if plot_xticks:
        xticklabels = np.arange(0, L + xtick_distance // 2, xtick_distance)
        xticks = [l for l in xticklabels]
        plt.xlabel('Locus index', fontsize=SIZELABEL)
    else:
        xticks, xticklabels = [], []

    set_ticks_labels_axes(yticks=[], yticklabels=[], xticks=xticks, xticklabels=xticklabels)
    plt.subplots_adjust(**DEF_DXDX_MARGINPROPS)

    if plotShow:
        plt.show()
    if plotFigure and save_file:
        fig.savefig(save_file, facecolor=fig.get_facecolor(), **DEF_FIGPROPS)
    if plotFigure and returnFig:
        return fig


def normalize_segmentedIntDxdx(segmentedIntDxdx_, scale=0.3):
    segmentedIntDxdx = np.copy(segmentedIntDxdx_)
    L = len(segmentedIntDxdx)
    offDiagonalTerms = EC.get_off_diagonal_terms(segmentedIntDxdx)
    vmin, vmax = np.min(offDiagonalTerms), np.max(offDiagonalTerms)

    # Normalize all magnitudes between [0, 1]
    for l1 in range(L):
        segmentedIntDxdx[l1, l1] = 0
        for l2 in range(l1):
            if segmentedIntDxdx[l1, l2] < 0:
                segmentedIntDxdx[l1, l2] /= -vmin
                segmentedIntDxdx[l2, l1] = segmentedIntDxdx[l1, l2]
            elif segmentedIntDxdx[l1, l2] > 0:
                segmentedIntDxdx[l1, l2] /= vmax
                segmentedIntDxdx[l2, l1] = segmentedIntDxdx[l1, l2]

    # Push magnitudes toward 1
    # print(segmentedIntDxdx[0, :3])
    for l1 in range(L):
        for l2 in range(l1):
            if segmentedIntDxdx[l1, l2] < 0:
                segmentedIntDxdx[l1, l2] = -(-segmentedIntDxdx[l1, l2]) ** scale
                segmentedIntDxdx[l2, l1] = segmentedIntDxdx[l1, l2]
            elif segmentedIntDxdx[l1, l2] > 0:
                segmentedIntDxdx[l1, l2] = segmentedIntDxdx[l1, l2] ** scale
                segmentedIntDxdx[l2, l1] = segmentedIntDxdx[l1, l2]
    # print(segmentedIntDxdx[0, :3])
    return segmentedIntDxdx


def plotIntDxdx(intWeightedDxdx, ylabel_x=-0.21, alpha=0.5, grid_line_offset=0, cbar_shrink=1, xtick_distance=4, plot_cbar=True, plot_xticks=True, vmin=None, vmax=None, figsize=(SINGLE_COLUMN/2, SINGLE_COLUMN/2), plotFigure=True, plotShow=True, save_file=None):

    if plotFigure:
        fig = plt.figure(figsize=figsize)
    ax = plt.gca()

    L = len(intWeightedDxdx)
    for l in range(L):
        intWeightedDxdx[l, l] = 0

    heatmap = sns.heatmap(intWeightedDxdx, center=0, vmin=vmin, vmax=vmax, cmap=CMAP, square=True, cbar=plot_cbar, cbar_kws={"shrink": cbar_shrink})
    
    if plot_cbar:
        cbar = heatmap.collections[0].colorbar
        cbar.ax.tick_params(**DEF_TICKPROPS_COLORBAR)
    
    if plot_xticks:
        xticklabels = np.arange(0, L + xtick_distance // 2, xtick_distance)
        xticks = [l for l in xticklabels]
        plt.xlabel('Locus index', fontsize=SIZELABEL)
    else:
        xticks, xticklabels = [], []

    set_ticks_labels_axes(yticks=[], yticklabels=[], xticks=xticks, xticklabels=xticklabels)
    plt.subplots_adjust(**DEF_DXDX_MARGINPROPS)

    if plotShow:
        plt.show()
    if save_file:
        fig.savefig(save_file, facecolor=fig.get_facecolor(), **DEF_FIGPROPS)


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


def plotCovList(cov_list, figsize=(20, 15), nCol=5, fontsize=15, plotTitle=True, titles=None, cbar=False, vmin=None, vmax=None):
    nRow = len(cov_list) // nCol + (len(cov_list) % nCol > 0)
    fig, axes = plt.subplots(nRow, nCol, figsize=figsize)
    for n, cov in enumerate(cov_list):
        if nRow > 1:
            plt.sca(axes[n//nCol, n%nCol])
        else:
            plt.sca(axes[n])
        sns.heatmap(cov, center=0, square=True, cbar=cbar, vmin=vmin, vmax=vmax,
                    cmap=sns.diverging_palette(145, 300, as_cmap=True))
        if plotTitle:
            if titles is None:
                plt.title(f"n={n}", fontsize=fontsize)
            else:
                plt.title(titles[n], fontsize=fontsize)
    plt.show()


def plotSelection(compareS, recoveredS, error, n, title=None, annot=True, fontsize=15, plotXticks=True, plotYticks=True):
    L = len(compareS)
    minS, maxS = min(np.min(compareS), np.min(recoveredS)), max(np.max(compareS), np.max(recoveredS))
    annot_offset = (maxS - minS) / 60
    xlim = ylim = (minS - 0.01, maxS + 0.01)
    if error is not None:
        error /= np.max(error)
        error *= 0.3 * (maxS - minS)
    for l in range(L):
        plt.scatter([compareS[l]], [recoveredS[l]], color=COLORS[l//len(COLORS)])
        if error is not None:
            plt.errorbar(compareS[l], recoveredS[l], yerr=error[l], color=COLORS[l//len(COLORS)], ecolor='grey', elinewidth=1)
        if annot:
            plt.text(compareS[l], recoveredS[l] + annot_offset, f'{l}', fontsize=fontsize, color=COLORS[l//len(COLORS)])
    plt.plot([minS, maxS], [minS, maxS], linestyle='dashed', color='grey')
    plt.xlim(xlim)
    plt.ylim(ylim)
    if not plotXticks:
        plt.xticks(ticks=[], labels=[])
    if not plotYticks:
        plt.yticks(ticks=[], labels=[])
    if title is not None:
        plt.title(title, fontsize=fontsize)


def plotSelectionForSimulations(summaries, start_n=0, checkPerfOfTrueCov=False, compareWithTrueCov=False, figsize=(20, 15), nCol=5, fontsize=15, hspace=0.3, wspace=0.3):
    nRow = len(summaries) // nCol + (len(summaries) % nCol > 0)
    fig, axes = plt.subplots(nRow, nCol, figsize=figsize)
    for n, res in enumerate(summaries):
        if res is None:
            continue
        plt.sca(axes[n//nCol, n%nCol])
        at_left = n%nCol == 0
        at_bottom = n//nCol == nRow - 1
        if checkPerfOfTrueCov:
            variance = np.array([res['true_cov'][1][l, l] for l in range(len(res['true_cov'][2]))])
            error = 1 / (variance + res['reg'])
            recoveredS = res['true_cov'][2]
        else:
            variance = np.array([res['recovered'][1][l, l] for l in range(len(res['recovered'][2]))])
            error = 1 / (variance + res['reg'])
            recoveredS = res['recovered'][2]
        if compareWithTrueCov:
            compareS = res['true_cov'][2]
        else:
            compareS = res['true'][2]
        spearmanr, _ = stats.spearmanr(compareS, recoveredS)
        title = f'n={n}, MAE=%.4f\nspearmanr=%.2f' % (MAE(compareS, recoveredS), spearmanr)
        plotSelection(compareS, recoveredS, error, n + start_n, title=title, fontsize=fontsize, plotXticks=at_bottom, plotYticks=at_left)
    plt.subplots_adjust(hspace=hspace, wspace=wspace)
    plt.show()


def checkSelectionByTrueCov(summaries, start_n=0, figsize=(10, 9), nCol=3, fontsize=8, alpha=0.3, hspace=0.45, wspace=0.45):
    nRow = 3
    fig, axes = plt.subplots(nRow, nCol, figsize=figsize)
    s_trueCov, s_true, emergenceTime_list = [], [], []
    fitness_traj_list = []
    fitness_at_emergence = []
    s_trueCov_normalized_by_true_fitness, s_trueCov_normalized_by_inferred_fitness = [], []
    for n, res in enumerate(summaries):
        if res is None:
            continue
        traj = res['true'][0]
        emergenceTimes = getEmergenceTimes(traj)
        s_trueCov_ = res['true_cov'][2]
        s_true_ = res['true'][2]
        fitness_traj, fitness_traj_trueCov = res['true'][3], res['true_cov'][3]
        fitness_traj_list.append(fitness_traj)
        for i in range(len(traj[0])):
            if np.max(traj[:, i]) < 0.3:
                continue
            s_trueCov.append(s_trueCov_[i])
            s_true.append(s_true_[i])
            emergenceTime_list.append(emergenceTimes[i])
            fitness_at_emergence.append(fitness_traj[emergenceTimes[i]])
        for i, s in enumerate(list(s_trueCov_)):
            if np.max(traj[:, i]) < 0.3:
                continue
            if np.max(traj[:, i]) < 0.9:
                s_trueCov_normalized_by_true_fitness.append(s)
                s_trueCov_normalized_by_inferred_fitness.append(s)
                continue
            # time = emergenceTimes[i]
            # peakTime = np.argmax(traj[:, i])
            dx = np.array([traj[t + 1] - traj[t] for t in range(len(traj) - 1)])
            time = np.argmax(np.absolute(dx[:, i]))
            s_trueCov_normalized_by_true_fitness.append(s * fitness_traj[time])
            s_trueCov_normalized_by_inferred_fitness.append(s * fitness_traj_trueCov[time])
            # s_trueCov_normalized_by_true_fitness.append((1 + s) * fitness_traj[time] - 1)
            # s_trueCov_normalized_by_inferred_fitness.append((1 + s) * fitness_traj_trueCov[time] - 1)



    plt.sca(axes[0, 0])
    x, y = np.array(s_true), np.array(s_trueCov)
    title = f'MAE=%.4f\nspearmanr=%.2f' % (MAE(x, y), stats.spearmanr(x, y)[0])
    plt.scatter(x, y, alpha=alpha)
    plt.xlabel('True selection', fontsize=fontsize)
    plt.ylabel('Selection inferred by true-cov', fontsize=fontsize)
    plt.title(title, fontsize=fontsize)

    plt.sca(axes[0, 1])
    x, y = np.array(emergenceTime_list), np.array(s_trueCov) - np.array(s_true)
    title = f'spearmanr=%.2f' % (stats.spearmanr(x, y)[0])
    plt.scatter(x, y, alpha=alpha)
    plt.xlabel('Emergence time', fontsize=fontsize)
    plt.ylabel('Error of selection inferred by true-cov', fontsize=fontsize)
    plt.title(title, fontsize=fontsize)

    plt.sca(axes[0, 2])
    x, y = np.array(emergenceTime_list), np.array(s_trueCov) / np.array(s_true)
    title = f'spearmanr=%.2f' % (stats.spearmanr(x, y)[0])
    plt.scatter(x, y, alpha=alpha)
    plt.xlabel('Emergence time', fontsize=fontsize)
    plt.ylabel('Ratio between selection inferred \nby true-cov and true selection', fontsize=fontsize)
    plt.title(title, fontsize=fontsize)

    plt.sca(axes[1, 0])
    for fitness_traj in fitness_traj_list:
        plt.plot(range(len(fitness_traj)), fitness_traj, alpha=0.1)
    plt.title('All fitness trajectories', fontsize=fontsize)

    plt.sca(axes[1, 1])
    x, y = np.array(s_true), np.array(s_trueCov_normalized_by_true_fitness)
    title = f'MAE=%.4f\nspearmanr=%.2f' % (MAE(x, y), stats.spearmanr(x, y)[0])
    plt.scatter(x, y, alpha=alpha)
    plt.xlabel('True selection', fontsize=fontsize)
    plt.ylabel('Selection inferred by true-cov \nnormalized by true fitness', fontsize=fontsize)
    plt.title(title, fontsize=fontsize)

    plt.sca(axes[1, 2])
    x, y = np.array(s_true), np.array(s_trueCov_normalized_by_inferred_fitness)
    title = f'MAE=%.4f\nspearmanr=%.2f' % (MAE(x, y), stats.spearmanr(x, y)[0])
    plt.scatter(x, y, alpha=alpha)
    plt.xlabel('True selection', fontsize=fontsize)
    plt.ylabel('Selection inferred by true-cov \nnormalized by inferred fitness', fontsize=fontsize)
    plt.title(title, fontsize=fontsize)


    plt.sca(axes[2, 0])
    x, y = np.array(s_trueCov), np.array(s_trueCov_normalized_by_inferred_fitness)
    title = f'MAE=%.4f\nspearmanr=%.2f' % (MAE(x, y), stats.spearmanr(x, y)[0])
    plt.scatter(x, y, alpha=alpha)
    plt.xlabel('Selection inferred by true-cov', fontsize=fontsize)
    plt.ylabel('Selection inferred by true-cov \nnormalized by inferred fitness', fontsize=fontsize)
    plt.title(title, fontsize=fontsize)

    plt.sca(axes[2, 1])
    x, y = np.array(fitness_at_emergence), np.array(s_true) / np.array(s_trueCov)
    title = f'spearmanr=%.2f' % (stats.spearmanr(x, y)[0])
    plt.scatter(x, y, alpha=alpha)
    plt.xlabel('Population fitness at emergence time', fontsize=fontsize)
    plt.ylabel('Ratio between true selection and\n selection inferred by true-cov', fontsize=fontsize)
    plt.title(title, fontsize=fontsize)
    plt.xlim(0.9, 1.4)
    plt.ylim(0, 5)

    plt.subplots_adjust(hspace=hspace, wspace=wspace)
    plt.show()


def plotFitnessForSimulations(summaries, index=3, times=None, titles=None, compareFitnessList=None, compareLabel=None, compareEstCov=True, start_n=0, figsize=(20, 15), nCol=5, fontsize=15, plotSL=False, ylims=None, ylim=None, ommitXticks=True, ommitYticks=True):
    nRow = len(summaries) // nCol + (len(summaries) % nCol > 0)
    fig, axes = plt.subplots(nRow, nCol, figsize=figsize)
    for n, res in enumerate(summaries):
        if res is None:
            continue
        plt.sca(axes[n//nCol, n%nCol])
        at_left = n%nCol == 0
        at_bottom = n//nCol == nRow - 1
        fitness_list = []
        labels = []
        if 'true' in res and res['true'][index] is not None:
            fitness_list.append(res['true'][index])
            labels.append('true')
        if 'true_cov' in res and res['true_cov'][index] is not None:
            fitness_list.append(res['true_cov'][index])
            labels.append('true_cov')
        if 'recovered' in res and res['recovered'][index] is not None:
            fitness_list.append(res['recovered'][index])
            labels.append('recovered')
        if plotSL and 'SL' in res and res['SL'][index] is not None:
            fitness_list.append(res['SL'][index])
            labels.append('SL')
        anotherFitnessToCompare, anotherLabel = None, None
        if compareEstCov and 'est_cov' in res and res['est_cov'][index] is not None:
            fitness_list.append(res['est_cov'][index])
            labels.append('by_est_cov')
        if compareFitnessList is not None and compareLabel is not None:
            fitness_list.append(compareFitnessList[n])
            labels.append(compareLabel)
        title = f'n={n+start_n}' if titles is None else titles[n]
        ylim_ = ylims[n] if ylims is not None else ylim
        plotXticks = at_bottom if ommitXticks else True
        plotYticks = at_left if ommitYticks else True
        plotFitnessComparison(fitness_list, times=times, labels=labels, asSubplot=True, title=title, plotLegend=(n==0), plotXticks=plotXticks, plotYticks=plotYticks, ylim=ylim_)
    plt.show()


def plotFitnessForSimulationsFromTwoResults(summaries1, summaries2, label1, label2, plotTrueCov=True, start_n=0, figsize=(20, 15), nCol=5, fontsize=15):
    numPlots = len(summaries1)
    nRow = numPlots // nCol + (numPlots % nCol > 0)
    fig, axes = plt.subplots(nRow, nCol, figsize=figsize)
    for n, (res1, res2) in enumerate(zip(summaries1, summaries2)):
        plt.sca(axes[n//nCol, n%nCol])
        if res1 is None:
            continue
        if plotTrueCov:
            labels = ['true', 'by_true_cov | ' + r'$\gamma=$' + str(res1['reg']) + ' | ' + label1, 'by_true_cov | ' + r'$\gamma=$' + str(res2['reg']) + ' | ' + label2]
            colors = [3, 2, 0]
            plotFitnessComparison(res1['true'][3], res1['true_cov'][3], res2['true_cov'][3], asSubplot=True, title=f' n={n+start_n}', labels=labels, colors=colors)
        else:
            labels = ['true', 'by_est_cov | ' + r'$\gamma=$' + str(res1['reg_est']) + ' | ' + label1, 'by_est_cov | ' + r'$\gamma=$' + str(res2['reg_est']) + ' | ' + label2]
            colors = [3, 4, 1]
            plotFitnessComparison(res1['true'][3], res1['recovered'][3], res2['recovered'][3], asSubplot=True, title=f' n={n+start_n}', labels=labels, colors=colors)
        if n//nCol < nRow - 1:
            plt.xticks(ticks=[], labels=[])
    plt.show()


def plotFitnessForSimulationsTwoResults(summaries1, summaries2, label1, label2, start_n=0, figsize=(20, 25), nCol=5, fontsize=15):
    numPlots = len(summaries1) + len(summaries2)
    nRow = numPlots // nCol + (numPlots % nCol > 0)
    fig, axes = plt.subplots(nRow, nCol, figsize=figsize)
    for n, (res1, res2) in enumerate(zip(summaries1, summaries2)):
        plt.sca(axes[2 * (n//nCol), n%nCol])
        if res1 is None:
            continue
        labels = ['true', 'by_true_cov | ' + r'$\gamma=$' + str(res1['reg']), 'by_est_cov | ' + r'$\gamma=$' + str(res1['reg_est'])]
        plotFitnessComparison(res1['true'][3], res1['true_cov'][3], res1['recovered'][3], asSubplot=True, title=label1 + f' n={n+start_n}', labels=labels)
        plt.xticks(ticks=[], labels=[])
        plt.sca(axes[2 * (n//nCol) + 1, n%nCol])
        if res2 is None:
            continue
        labels = ['true', 'by_true_cov | ' + r'$\gamma=$' + str(res2['reg']), 'by_est_cov | ' + r'$\gamma=$' + str(res2['reg_est'])]
        plotFitnessComparison(res2['true'][3], res2['true_cov'][3], res2['recovered'][3], asSubplot=True, title=label2 + f' n={n+start_n}', labels=labels)
        if 2 * (n//nCol) + 1 < nRow - 1:
            plt.xticks(ticks=[], labels=[])
    plt.show()


def plotAllForSimulations(summaries, start_n=0, figsize=(20, 75), nCol=5, fontsize=15):
    nRow = len(summaries)
    fig, axes = plt.subplots(nRow, nCol, figsize=figsize)
    for n, res in enumerate(summaries):
        if res is None:
            continue
        plt.sca(axes[n, 0])
        plotTrajComparison(res['true'][0], res['recovered'][0], asSubplot=True,
                              annot=True, title=f'n={n}, MAE=%.3f'%(MAE(res['true'][0], res['recovered'][0])),
                              xticks=n//nCol == nRow-1, yticks=n%nCol==0)
        plt.sca(axes[n, 1])
        sns.heatmap(res['recovered'][1]-res['true'][1], center=0, square=True, cbar=False,
                    cmap=sns.diverging_palette(145, 300, as_cmap=True))
        plt.title(f'n={n+start_n}', fontsize=fontsize)
        plt.sca(axes[n, 2])
        variance = np.array([res['recovered'][1][l, l] for l in range(len(res['recovered'][2]))])
        plotSelection(res['true'][2], res['recovered'][2], 1 / (variance + res['reg']), n)
        plt.sca(axes[n, 3])
        plotSelection(res['true_cov'][2], res['recovered'][2], 1 / (variance + res['reg']), n)
        plt.sca(axes[n, 4])
        plotFitnessComparison(res['true'][3], res['true_cov'][3], res['recovered'][3], asSubplot=True, title=f'n={n}')

    plt.show()


def plotReconstructionEvaluationForSimulations(summaries, plotProcessed=False, figsize=(10, 6), fontsize=12):
    for n, summary in enumerate(summaries):
        for key, val in summary.items():
            if 'MAE' in key and '<' in key:
                if not val:
                    print(f'Simulation {n}, {key} is False.')
    logP_recovered_normalized_list = [_['logP_recovered_normalized'] for _ in summaries]
    if plotProcessed:
        logP_processed_normalized_list = [_['logP_processed_normalized'] for _ in summaries]
    MAE_recoveredTraj_list = [_['MAE_recoveredTraj'] for _ in summaries]
    # MAE_processedTraj_list = [_['MAE_processedTraj'] for _ in summaries]
    fig, axes = plt.subplots(2, 1, figsize=figsize)
    plt.sca(axes[0])
    plt.scatter(np.arange(len(summaries)), logP_recovered_normalized_list, label='logP_recovered_normalized')
    if plotProcessed:
        plt.scatter(np.arange(len(summaries)), logP_processed_normalized_list, label='logP_processed_normalized')
    bot, top = plt.gca().get_ylim()
    for n in range(len(summaries)):
        y = logP_recovered_normalized_list[n]
        if plotProcessed:
            y = max(y, logP_processed_normalized_list[n])
        plt.text(n, y, f'{n}')
    # plt.title('Normalized logP per simulation', fontsize=fontsize)
    plt.ylabel('Normalized logP', fontsize=fontsize)
    plt.legend(fontsize=fontsize)
    plt.sca(axes[1])
    plt.scatter(np.arange(len(summaries)), MAE_recoveredTraj_list, label='MAE_recoveredTraj')
    # plt.scatter(np.arange(len(summaries)), MAE_processedTraj_list, label='MAE_processedTraj')
    bot, top = plt.gca().get_ylim()
    for n in range(len(summaries)):
        plt.text(n, MAE_recoveredTraj_list[n], f'{n}')
    # plt.title('MAE of traj per simulation', fontsize=fontsize)
    plt.legend(fontsize=fontsize)
    plt.ylabel(f'MAE of traj', fontsize=fontsize)
    plt.xlabel(f'Simulation', fontsize=fontsize)
    plt.show()


def plotReconstructionMetricsWithInferenceMetrics(summaries_reconstruction, summaries_inference, exclude_by_logP=False, exclude_by_MAE=False, th_logP=-2.5, th_MAE=0.001, figsize=(15, 8), wspace=0.4, hspace=0.4):
    rm_names = ['MAE_recoveredTraj', 'logP_recovered_normalized']
    reconstruction_metrics = [
        [_[name] for _ in summaries_reconstruction] for name in rm_names
    ]
    MAE_cov, spearmanr_cov = getPerf(summaries_inference, index=1)
    MAE_selection, spearmanr_selection = getPerf(summaries_inference, index=2)
    MAE_fitness, spearmanr_fitness = getPerf(summaries_inference, index=3)
    # im_names = ['MAE_cov', 'Spearmanr_cov', 'MAE_selection', 'Spearmanr_selection', 'MAE_fitness', 'Spearmanr_fitness']
    # inference_metrics = [MAE_cov, spearmanr_cov, MAE_selection, spearmanr_selection, MAE_fitness, spearmanr_fitness]
    im_names = ['MAE_cov', 'Spearmanr_cov', 'MAE_selection', 'Spearmanr_selection']
    inference_metrics = [MAE_cov, spearmanr_cov, MAE_selection, spearmanr_selection]

    nRow, nCol = len(reconstruction_metrics), len(inference_metrics)
    fig, axes = plt.subplots(nRow, nCol, figsize=figsize)
    for i, rm in enumerate(reconstruction_metrics):
        for j, im in enumerate(inference_metrics):
            plt.sca(axes[i, j])
            at_left = (j == 0)
            at_bottom = (i == nRow - 1)
            prefix = ''
            if 'logP' in rm_names[i] and exclude_by_logP:
                indices = np.array([_ for _ in range(len(rm)) if rm[_] < th_logP])
                im, rm = np.array(im)[indices], np.array(rm)[indices]
                prefix = f'filtered_by_logP {th_logP}\n'
            elif 'MAE' in rm_names[i] and exclude_by_MAE:
                indices = [_ for _ in range(len(rm)) if rm[_] > th_MAE]
                im, rm = np.array(im)[indices], np.array(rm)[indices]
                prefix = f'filtered_by_MAE {th_MAE}\n'
            plt.scatter(im, rm)
            if at_left:
                plt.ylabel(prefix + rm_names[i])
            else:
                plt.yticks(ticks=[], labels=[])
            if at_bottom:
                plt.xlabel(im_names[j])
            else:
                plt.xticks(ticks=[], labels=[])
            plt.title('Spearmanr=%.2f' % stats.spearmanr(rm, im)[0])
    plt.subplots_adjust(wspace=wspace, hspace=hspace)
    plt.show()



def getPerf(summaries, index=3, name='recovered'):
    MAE = np.zeros(len(summaries))
    spearmanr = np.zeros(len(summaries))
    for n, summary in enumerate(summaries):
        trueCov, recovered = summary['true_cov'][index], summary[name][index]
        MAE[n] = RC.MAE(trueCov, recovered)
        spearmanr[n], _ = stats.spearmanr(np.ndarray.flatten(trueCov), np.ndarray.flatten(recovered))
    return MAE, spearmanr


def getPerfFromList(summaries, perfList, index=3):
    MAE = np.zeros(len(perfList))
    spearmanr = np.zeros(len(perfList))
    for n, perf in enumerate(perfList):
        summary = summaries[n]
        trueCov = summary['true_cov'][index]
        MAE[n] = RC.MAE(trueCov, perf)
        spearmanr[n], _ = stats.spearmanr(np.ndarray.flatten(trueCov), np.ndarray.flatten(perf))
    return MAE, spearmanr


def plotPerf(summaries, index=3, compareName='est_cov', figsize=(10, 4)):
    if index == 2:
        postfix = ' for selections'
    elif index ==3:
        postfix = ' for fitness'
    elif index == 1:
        postfix = ' for covariance'
    MAE, spearmanr = getPerf(summaries, index=index, compareName=compareName)
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    plt.sca(axes[0])
    plt.hist(MAE)
    plt.title('MAE' + postfix)
    plt.sca(axes[1])
    plt.hist(spearmanr, bins=np.arange(0, 1.1, 0.1))
#     yticks = np.arange(0, len(summaries) + 5, 5)
#     plt.yticks(ticks=yticks, labels=yticks)
    plt.title('Spearmanr' + postfix)
    plt.show()


def plotPerfComparison(summaries, index=3, compareName='lolipop', compareList=None, figsize=(10, 4), alpha=0.6):
    if index == 2:
        postfix = ' for selections'
    elif index ==3:
        postfix = ' for fitness'
    elif index == 1:
        postfix = ' for covariance'
    MAE, spearmanr = getPerf(summaries, index=index, name='recovered')
    if compareList is not None:
        MAE_compare, spearmanr_compare = getPerfFromList(summaries, compareList, index=index)
    else:
        MAE_compare, spearmanr_compare = getPerf(summaries, index=index, name=compareName)
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    plt.sca(axes[0])
    bins = np.arange(0, max(np.max(MAE), np.max(MAE_compare)) + 0.01, 0.01)
    plt.hist(MAE, bins=bins, label='recovered', alpha=alpha)
    plt.hist(MAE_compare, bins=bins, label=compareName, alpha=alpha)
    plt.title('MAE' + postfix)
    plt.legend()
    plt.sca(axes[1])
    plt.hist(spearmanr, bins=np.arange(0, 1.1, 0.1), label='recovered', alpha=alpha)
    plt.hist(spearmanr_compare, bins=np.arange(0, 1.1, 0.1), label=compareName, alpha=alpha)
#     yticks = np.arange(0, len(summaries) + 5, 5)
#     plt.yticks(ticks=yticks, labels=yticks)
    plt.title('Spearmanr' + postfix)
    plt.legend()
    plt.show()


def checkPerfStats(summaries, index=3, compareName='est_cov', closeStandard='MAE<0.01'):
    if index == 3:
        print(f'For fitness trajectories')
    elif index == 2:
        print(f'For selections')
    count_close_to_true = 0
    count_better_than_compare = 0
    for n, summary in enumerate(summaries_99):
        trueCov, recovered, compare = summary['true_cov'][index], summary['recovered'][index], summary[compareName][index]
        MAE_recovered = RC.MAE(trueCov, recovered)
        MAE_compare = RC.MAE(trueCov, compare)
        spearmanr_recovered, _ = stats.spearmanr(trueCov, recovered)
        if MAE_recovered < MAE_compare:
            count_better_than_compare += 1
        if closeStandard[:3] == 'MAE':
            th = float(closeStandard[4:])
            if MAE_recovered < th:
                count_close_to_true += 1
        elif closeStandard[:9] == 'spearmanr':
            th = float(closeStandard[10:])
            if spearmanr_recovered > th:
                count_close_to_true += 1
    print(f'Better than {compareName} (in terms of MAE): ', count_better_than_compare, '/ 40')
    print(f'Close to trueCov ({closeStandard}): ', count_close_to_true, '/ 40')


#########################################
# Plotting functions for interpolation*.ipynb
#########################################


def plotSimulation(simulation, true_selection=None, scatter=True, markersize=7, annot=True, reg=0, plotGeno=False, title=None, plotShow=True):
    traj, times, mu = simulation['traj'], simulation['times'], simulation['mu']
    T, L = traj.shape
    N = np.sum(simulation['nVec'][0])
    if true_selection is not None:
        D = MPL.computeD(traj, times, mu)
        s_inferred = MPL.inferSelection(simulation['cov'], D, reg * np.identity(L))
        printSelection(s_inferred, true_selection)
    if title is None:
        title = 'Allele traj'
    plotTraj(traj, times=times, scatter=scatter, markersize=markersize, title=title, annot=annot, plotShow=plotShow)
    if plotGeno:
        seqOnceEmerged, freqOnceEmerged, pop = getDominantGenotypeTrajectories(simulation, T=T, threshold=0, totalPopulation=N)
        plotTraj(freqOnceEmerged.T, times=times, scatter=scatter, markersize=markersize, title='genotype traj', annot=annot)


def printSelection(inferred_selection, true_selection, printMAE=True, prefix=''):
    true_text = ' '.join(['%.3f' % _ for _ in true_selection])
    print(prefix + 'true : ', true_text)
    inferred_text = ' '.join(['%.3f' % _ for _ in inferred_selection])
    print(prefix + 'infer: ', inferred_text)
    if printMAE:
        print(prefix + 'MAE  :  %.4f ' % RC.MAE(inferred_selection, true_selection))


def plotInterpolation(traj_intpl, traj_sampled, times_sampled, traj_original=None, title='Interpolated', annot=True, plotLegend=True):
    placeholder_xs = [0, 1e-6]
    placeholder_ys = [-0.1, -0.1]
    placeholder_color = 'grey'
    plotTraj(traj_sampled, times=times_sampled, scatterOnly=True, s=10, plotShow=False, marker='o')
    plt.scatter(placeholder_xs, placeholder_ys, color=placeholder_color, label='Sampled')
    plotTraj(traj_intpl, annot=annot, scatter=False, plotFigure=False, plotShow=False, alpha=0.5, title=title)
    plt.plot(placeholder_xs, placeholder_ys, color=placeholder_color, linestyle='solid', label=title)
    if traj_original is not None:
        plotTraj(traj_original, scatter=False, plotFigure=False, plotShow=False, alpha=0.3, linestyle='dashed', title=title)
        plt.plot(placeholder_xs, placeholder_ys, color=placeholder_color, linestyle='dashed', label='True')
    if plotLegend:
        plt.legend()
    plt.show()


def compareTrajVertically(traj_list, times_list=None, titles=None, annot=True):
    for i, traj in enumerate(traj_list):
        times = times_list[i] if times_list is not None else np.arange(len(traj))
        title = titles[i] if titles is not None else ''
        plotTraj(traj, times=times, title=title, annot=annot)
