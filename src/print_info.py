import sys
import numpy as np
import pandas as pd
import math
from scipy import stats
from scipy import interpolate
import matplotlib.pyplot as plt
from tabulate import tabulate


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


def printGroups(groups):
    table = [[len(group) for group in groups]]
    headers = [f'{bcolors.BLACK}shared'] + [f'group {i+1}' for i in range(len(groups) - 1)]
    print(tabulate(table, headers, tablefmt='plain'))


def printPeriods(periods):
    print(f'\nIdentifying competition periods... {periods}')


def printCompetingClades(traj, competingClades):
    (T, L), K = traj.shape, len(competingClades) + 1

    print("\nprinting clades composition...")
    print('index ', end = "     ")
    for i in range(len(traj[0])):
        print(i // 10, end = ' ')
    print()
    print('index ', end = "     ")
    for i in range(len(traj[0])):
        print(i % 10, end = ' ')
    print()

    print(f'clade 1')
    for i in range(0, len(competingClades)):
        print(f' end  with {seqToSparseString(mutToSeq(competingClades[i][0], L), compact=True)}')
        print(f'clade {i + 2}')
        print(f'start with {seqToSparseString(mutToSeq(competingClades[i][1], L), compact=True)}')


def seqToSparseString(seq, compact = False):
    s = ""
    if compact:
        for i in range(len(seq)):
            if int(seq[len(seq) - 1 - i]) != 0:
                break
        for j in range(0, len(seq) - i):
            s += str(int(seq[j])) + ' '
    else:
        for i in seq:
            s += str(int(i)) + ' '
    return s


def mutToSeq(muts, L):
    seq = np.zeros(L, dtype = int)
    for l in muts:
        seq[l] = 1
    return seq


def printFinalCladesInfo(traj, majorCladePolyMuts, otherMutations):
    T, L = traj.shape
    numClades = len(majorCladePolyMuts)
    totalMuts = np.sum([len(k) for k in majorCladePolyMuts]) + len(otherMutations)

    names = [f'{bcolors.BLACK}clade {k+1}' for k in range(numClades)] + ['other', 'total']
    numMuts = [len(clade) for clade in majorCladePolyMuts] + [len(otherMutations), totalMuts]
    print(tabulate([numMuts], names, tablefmt='plain'))
    if not L == totalMuts:
        print(f'{bcolors.FAIL}> Counts do not match {L} != {totalMuts}')
    cladeMutsAreUnique = np.all([len(np.unique(k)) == len(k) for k in majorCladePolyMuts])
    otherMutsAreUnique = len(np.unique(otherMutations)) == len(otherMutations)
    if not cladeMutsAreUnique:
        print(f'{bcolors.FAIL}> Some clade(s) contain duplicate mutations')
    if not otherMutsAreUnique:
        print(f'{bcolors.FAIL}> Non-clade group contains duplicate mutations')


def printClassifications(classifications):
    keys = ['all', 'fixed', 'extinct', 'fixing', 'extincting', 'intermediate',
            'laterFixed', 'laterExtinct', 'laterIntermediate']

    print("\nprinting classifications in each period...")
    for p in range(len(classifications)):
        print(f'period {p + 1}')
        first = True
        for k in keys:
            print(f'\t{k}', end = ": ")
            print(classifications[p][k])


def printWinnerLoser(winner, loser):
    print(f'\nprinting winner and loser in each period...')
    print(f'period', end = '\t')
    for i in range(len(winner)):
        print(f'{i + 1}', end = '\t')
    print(f'\nwinner', end = '\t')
    for i in range(len(winner)):
        print(f'{winner[i] + 1}', end = '\t')
    print(f'\nloser', end = '\t')
    for i in range(len(winner)):
        print(f'{loser[i] + 1}', end = '\t')
    print()


def printUnique(unique):
    print("\nprinting unique mutations of each clade...")
    for k in range(len(unique)):
        print(f'\tUnique muts of clade {k + 1}: {unique[k]}')
