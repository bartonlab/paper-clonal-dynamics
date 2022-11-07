#!/usr/bin/env python
# A simple Wright-Fisher simulation with an additive fitness model

import sys
import argparse
import numpy as np                          # numerical tools
import pickle
from timeit import default_timer as timer   # timer for performance
from copy import deepcopy

import MPL

import warnings
warnings.filterwarnings("ignore")


def main(verbose=False):
    """ Simulate Wright-Fisher evolution of a population and save the results. """

    # Read in simulation parameters from command line

    parser = argparse.ArgumentParser(description='Wright-Fisher evolutionary simulation.')
    parser.add_argument('-o',   type=str,   default='trajectory_covariance', help='output destination')
    parser.add_argument('-N',   type=int,   default=1000,              help='population size')
    parser.add_argument('-T',   type=int,   default=1000,              help='number of generations in simulation')
    parser.add_argument('--mu', type=float, default=1.0e-3,            help='mutation rate')
    parser.add_argument('--minS', type=float, default=0.01,            help='min selection for a single mutation, when drawing from a uniform distribution')
    parser.add_argument('--maxS', type=float, default=0.04,            help='max of selection for a single mutation, when drawing from a uniform distribution')
    parser.add_argument('--meanS', type=float, default=0.03,            help='mean selection for a single mutation')
    parser.add_argument('--stdS', type=float, default=0.01,            help='std of selection for a single mutation')
    parser.add_argument('--threshold', type=float, default=0.1,        help="threshold that a mutation's max frequency has to reach in order to be preserved in outputs")

    parser.add_argument('--uniform', action='store_true', default=False, help='whether or not draw selection coefficients from a uniform distribution')
    parser.add_argument('--controlled_genotype_fitness', action='store_true', default=False, help='whether or not control the increase of genotype fitness')
    parser.add_argument('--genotype_fitness_increase_rate', type=float, default=1.0e-4, help='genotype_fitness_increase_rate')

    parser.add_argument('--covariance', action='store_true', default=False, help='whether or not compute integrated covariacne matrix after simulation')
    parser.add_argument('--covAtEachTime', action='store_true', default=False, help='whether or not record covariance at each generation')

    parser.add_argument('--recombination', action='store_true', default=False, help='whether or not enable recombination')
    parser.add_argument('--recombination_rate', type=float, default=1.0e-8, help='recombination rate (probability for any two sequence to recombine at a generation)')

    parser.add_argument('--cooccurence', action='store_true', default=False, help='whether or not enable cooccurence')
    parser.add_argument('--max_cooccuring_mutations', type=int, default=1, help='max number of mutations that cooccur')

    parser.add_argument('--schedule', action='store_true', default=False, help='whether or not mutate according to schedules specified by a dict{time: selection}')
    parser.add_argument('--scheduled_mutation_selections',   type=str,   default=None, help='file containing dict specifying times and selections of scheduled mutations')

    parser.add_argument('--clade', action='store_true', default=False, help='whether or not enforce emergence of major and minor clades')
    parser.add_argument('--cladeInitialFitness', type=float, default=0.15, help='initial fitness of major and minor clades')

    parser.add_argument('--noMultinomialSampling', action='store_true', default=False, help='whether or not do a multinomial sampling when redistributing frequencies')
    parser.add_argument('--saveCompleteResults', action='store_true', default=False, help='whether or not also save outpur with all mutations. ')
    parser.add_argument('--verbose', action='store_true', default=False, help='whether or not print verbose info for debugging')


    arg_list = parser.parse_args(sys.argv[1:])

    out_str = arg_list.o
    N = arg_list.N
    T = arg_list.T
    mu = arg_list.mu
    recombination_rate = arg_list.recombination_rate
    max_cooccuring_mutations = arg_list.max_cooccuring_mutations
    genotype_fitness_increase_rate = arg_list.genotype_fitness_increase_rate
    L = int(1.2 * N * T * mu)
    meanS = arg_list.meanS
    stdS = arg_list.stdS
    minS = arg_list.minS
    maxS = arg_list.maxS


    if arg_list.scheduled_mutation_selections is not None:
        with open(arg_list.scheduled_mutation_selections, 'rb') as f:
            scheduled_mutation_selections = pickle.load(f)

    verbose = arg_list.verbose

    # _ SPECIES CLASS _ #

    class Species:

        positions = np.random.permutation(L)
        mid_to_pos = {i: pos for i, pos in enumerate(positions)} # record a mapping from mutation index to its position in the sequence
        selections = {} # record selections of all mutations
        clades = {} # record clade of all mutations
        mid = 0 # id of mutations
        majorMutations = []
        minorMutations = []

        """A group of sequences with the same genotype.

        Attributes:
            n: Number of members.
            f: Fitness value of the species.
            mutations: mutations acquired by the species.
            clade: clade of the species
        """
        def __init__(self, n = 1, f = 1., **kwargs):
            """ Initialize clone/provirus-specific variables. """
            self.n = n   # number of members
            self.f = f
            if 'mutations' in kwargs:
                self.mutations = kwargs['mutations']
            else:
                self.mutations = []
            if 'clade' in kwargs:
                self.clade = kwargs['clade']
            else:
                self.clade = 'ancestral'

        def compute_fitness_from_mutations(self):
            self.f = 1 + np.sum([Species.selections[m] for m in self.mutations])

        @classmethod
        def clone(cls, s):
            return cls(n = 1, f = s.f, mutations = [k for k in s.mutations], clade = s.clade) # Return a new copy of the input Species

        def mutate(self):
            """ Mutate and return self + new sequences."""

            newSpecies = []
            nMut = 0

            if self.n > 0:
                nMut = np.random.binomial(self.n, mu) # get number of individuals that mutate
                self.n -= nMut # subtract number mutated from size of current clone

                # process mutations
                for i in range(nMut):
                    s = Species.clone(self) # create a new copy sequence
                    num_cooccuring_muts = np.random.randint(1, max_cooccuring_mutations + 1)
                    for j in range(num_cooccuring_muts):
                        s.mutations.append(Species.mid)
                        randomS = self.sample_selection()
                        randomS /= num_cooccuring_muts  # normalize the fitness effect
                        s.f += randomS
                        Species.clades[Species.mid] = s.clade;
                        Species.selections[Species.mid] = randomS
                        Species.mid += 1
                    newSpecies.append(s)

            # return the result
            if self.n > 0:
                newSpecies.append(self)
            return newSpecies

        def sample_selection(self):
            if arg_list.controlled_genotype_fitness:
                meanS_, stdS_ = 1 + genotype_fitness_increase_rate * tCur - self.f, 0.003
                randomS = np.random.normal(meanS_, stdS_)
            elif arg_list.uniform:
                if minS <= -0.06:
                    negativePortion = -minS / (maxS - minS)
                    flag = np.random.random()
                    if flag < negativePortion:
                        randomS = np.random.uniform(-0.03, 0)
                    else:
                        randomS = np.random.uniform(0, maxS)
                else:
                    randomS = np.random.uniform(minS, maxS)
            else:
                randomS = np.random.normal(meanS, stdS)
            return randomS

        def mutate_as_scheduled(self, scheduled_selection):
            """ Mutate and return self + new sequences. """
            newSpecies = []
            nMut = min(10, max(1, self.n // 2))
            # nMut = self.n
            # nMut = max(1, self.n // 2)

            if self.n > 0:
                self.n -= nMut # subtract number mutated from size of current clone

                s = Species.clone(self) # create a new copy sequence
                s.mutations.append(Species.mid)
                Species.clades[Species.mid] = s.clade;
                Species.selections[Species.mid] = scheduled_selection
                s.f += scheduled_selection
                Species.mid += 1
                newSpecies.append(s)

            # return the result
            if self.n > 0:
                newSpecies.append(self)
            return newSpecies

        def recombine(self, other):
            # recombination with another Species
            positions = []
            newSpecies = []
            nRecomb = 0
            nPairs = self.n * other.n
            if nPairs > 0:
                # print(f'nPairs={nPairs}, recombination_rate={recombination_rate}')
                nRecomb = np.random.binomial(nPairs, recombination_rate)  # This is a good approximation when recombination_rate is small.
                nRecomb = min(nRecomb, min(self.n, other.n))  # In case nRecomb > one of the population size
                for i in range(nRecomb):
                    pos = np.random.randint(0, L - 1)
                    this_left, this_right = self.split_mutations_at(pos)
                    other_left, other_right = other.split_mutations_at(pos)
                    s1 = Species(n = 1, mutations = list(this_left.union(other_right)))
                    s1.compute_fitness_from_mutations()
                    s2 = Species(n = 1, mutations = list(other_left.union(this_right)))
                    s2.compute_fitness_from_mutations()
                    self.n -= 1
                    other.n -= 1
                    newSpecies.append(s1)
                    newSpecies.append(s2)
                    positions.append((pos, tuple(sorted(s1.mutations)), tuple(sorted(s2.mutations))))
            # if self.n > 0:
            #     newSpecies.append(self)
            # if other.n > 0:
            #     newSpecies.append(other)
            return newSpecies, positions

        def split_mutations_at(self, pos):
            left, right = set(), set()
            for m in self.mutations:
                p = Species.mid_to_pos[m]
                if p <= pos:
                    left.add(m)
                else:
                    right.add(m)
            return left, right


    # ^ SPECIES CLASS ^

    tStart = 0       # start generation
    tCur   = 0
    tEnd   = T       # end generation
    start  = timer() # track running time

    pop, mVec, sVec, nVec = [], [], [], []

    if arg_list.clade:
        s = arg_list.sMutationsClades
        majorMutations = [0]
        minorMutations = [1]
        Species.clades[0] = 'major'
        Species.clades[1] = 'minor'
        for i in range(2):
            Species.selections[i] = s
        pop = [Species(n=N // 2, f=1 + s, mutations=majorMutations, clade='major'),
               Species(n = N - N // 2, f = 1 + s, mutations=minorMutations, clade='minor')]
        mVec = [[majorMutations, minorMutations]]
        nVec = [[N // 2, N - N // 2]]
        Species.mid = 2
        cVec = [['major', 'minor']]
    else:
        # Start with all sequences being wild-type
        pop  = [Species(n = N)]             # current population; list of Species
        mVec = [[[]]]                       # list of mutations at each time point
        nVec = [[N]]   # list of clone size at each time point
        recombinations = []  # list of recombination events, which records (t, s1, s2, pos)

    # Evolve the population

    for t in range(tStart, tEnd):

        tCur = t

        # printUpdate(t, tEnd)    # status check

        # Select species to replicate
        r = np.array([s.n * s.f for s in pop])
        p = r / np.sum(r) # probability of selection for each species (sequence)
        if arg_list.noMultinomialSampling:
            n = N * p
        else:
            n = np.random.multinomial(N, pvals=p) # selected number of each species

        # Update population size and mutate
        newPop = []

        if arg_list.schedule:
            # When a mutation is scheduled at current time; Select a species that will acquire the mutation
            if t in scheduled_mutation_selections:
                selected_i = np.random.choice(np.arange(len(pop)), 1, p=n / np.sum(n))
                newPop = [species for i, species in enumerate(pop) if i != selected_i] + pop[selected_i].mutate_as_scheduled(scheduled_mutation_selections[t])
        else:
            # Mutate
            for i in range(len(pop)):
                pop[i].n = n[i] # set new number of each species
                mutatedPop = pop[i].mutate()
                newPop += mutatedPop

        if arg_list.recombination:
            for i, s1 in enumerate(pop):
                for j, s2 in enumerate(pop[i+1:]):
                    newSpecies_list, positions = s1.recombine(s2)
                    for pos in positions:
                        recombinations.append((t, tuple(s1.mutations), tuple(s2.mutations), pos))
                    for newSpecies in newSpecies_list:
                        foundDuplicate = False
                        for s in newPop:
                            if set(newSpecies.mutations) == set(s.mutations):
                                s.n += newSpecies.n
                                foundDuplicate = True
                        if not foundDuplicate:
                            newPop.append(newSpecies)

        pop = newPop

        # Update measurements

        nVec.append([s.n for s in pop])
        mVec.append([s.mutations for s in pop])
        if arg_list.clade:
            cVec.append([s.clade for s in pop])

    numMutations = len(Species.selections.keys())
    T = len(mVec)
    if verbose:
        print(f'# of timepoints {T}')
        print(f'{numMutations} mutations emerged')
        print(f'at t = 0, {mVec[0]}, {nVec[0]}')

    # calculate frequencies of all occured mutations
    fMut = np.zeros((T, numMutations), dtype=float)
    for t in range(T):
        for k in range(len(mVec[t])):
            for mid in mVec[t][k]:
                fMut[t, mid] += nVec[t][k] / N

    # preserve mutations that once exceed a threshold frequency
    preservedMutations = [mid for mid in range(numMutations) if np.max(fMut[:, mid]) > arg_list.threshold]
    cladesOfPreservedMutations = [Species.clades[mid] for mid in preservedMutations]
    L = len(preservedMutations)
    for t in range(T):
        for k in range(len(mVec[t])):
            mVec[t][k] = [mid for mid in mVec[t][k] if mid in preservedMutations]
    if verbose:
        print(f'{L} out of {numMutations} mutations are preserved by threshold {arg_list.threshold}')
        print(f'preserved mid: {preservedMutations}')

    # recombination events
    if arg_list.recombination:
        preservedRecombinations = []
        for (t, muts1, muts2, pos) in recombinations:
            preservedMuts1 = [m for m in muts1 if m in preservedMutations]
            preservedMuts2 = [m for m in muts2 if m in preservedMutations]
            preservedRecombinations.append((t, preservedMuts1, preservedMuts2, pos))

    # selections of preserved sites
    preservedSelections = [Species.selections[mid] for mid in preservedMutations]

    # merge mVec that are same and save to sVec
    midToSite = {preservedMutations[i]: i for i in range(L)}
    sVec = [[] for t in range(T)]
    nVec_merged = [[] for t in range(T)]
    if arg_list.clade:
        cVec_merged = [[] for t in range(T)]
    for t in range(T):
        seqToIndex = {}
        for k in range(len(mVec[t])):
            # seq = np.zeros(L, dtype=int)
            # for mid in mVec[t][k]:
            #     seq[midToSite[mid]] = 1
            sites = [midToSite[mid] for mid in mVec[t][k]]
            seq = sitesToSeq(sites, L)
            key = seq.tobytes()
            if key in seqToIndex:
                index = seqToIndex[key]
                nVec_merged[t][index] += nVec[t][k]
            else:
                seqToIndex[key] = len(sVec[t])
                sVec[t].append(seq)
                nVec_merged[t].append(nVec[t][k])
                if arg_list.clade:
                    cVec_merged[t].append(cVec[t][k])

    if verbose and arg_list.clade:
        print(f"{np.sum([nVec_merged[-1][k] for k in range(len(nVec_merged[-1])) if cVec_merged[-1][k] == 'major'])} out of {np.sum(nVec_merged[-1])} is major")
        print(f"{np.sum([nVec_merged[-1][k] for k in range(len(nVec_merged[-1])) if cVec_merged[-1][k] == 'minor'])} is minor")

    times = np.arange(T)
    traj = computeTrajectories(sVec, nVec_merged)
    D = MPL.computeD(traj, times, mu)

    # calculate the integrated covariance matrix
    if arg_list.covariance:
        q = 2
        totalCov = np.zeros((L, L), dtype=float)
        fVec_merged = [[count / N for count in nVec_merged[t]] for t in range(T)]
        if arg_list.covAtEachTime:
            covAtEachTime = MPL.processStandard(sVec, fVec_merged, times, q, totalCov, covAtEachTime=True)
        else:
            MPL.processStandard(sVec, fVec_merged, times, q, totalCov)

    # get complete sVec, with all mutations.
    if arg_list.saveCompleteResults:
        D_complete = MPL.computeD(fMut, times, mu)
        sVec_complete = [[sitesToSeq(mutations, numMutations) for mutations in mVec[t]] for t in range(T)]
        if arg_list.covariance:
            q = 2
            totalCov_complete = np.zeros((numMutations, numMutations), dtype=float)
            fVec_complete = [[count / N for count in nVec[t]] for t in range(T)]
            if arg_list.covAtEachTime:
                covAtEachTime_complete = MPL.processStandard(sVec_complete, fVec_complete, times, q, totalCov_complete, covAtEachTime=True)
            else:
                MPL.processStandard(sVec_complete, fVec_complete, times, q, totalCov_complete)

        f_complete = open(out_str + '_complete.npz', 'wb')
        data = {'mu': mu, 'nVec': nVec, 'sVec': sVec_complete, 'times': times, 'traj': fMut, 'D': D_complete, 'selections': Species.selections}
        if arg_list.covariance:
            data['cov'] = totalCov_complete
        if arg_list.covAtEachTime:
            data['covAtEachTime'] = covAtEachTime_complete
        if arg_list.clade:
            data['cladesMut'] = Species.clades
        np.savez_compressed(f_complete, **data)
        f_complete.close()

    # End and output total time
    if verbose:
        print(f'# of timepoints {len(sVec)}, {len(nVec_merged)}, {len(traj)}, {len(covAtEachTime) if arg_list.covAtEachTime else None}')
        print(f'# of sites {len(sVec[0][0])}, {len(traj[0])}, {len(preservedSelections)}, {len(totalCov) if arg_list.covariance else None}')
        if arg_list.recombination:
            print(f'# of recombinations {len(preservedRecombinations)}')
            if len(preservedRecombinations) < 20:
                print(f'\t they are: ')
                for t, muts1, muts2, pos in preservedRecombinations:
                    print('\t', t, muts1, muts2, pos[0], pos[1], pos[2])

    f = open(out_str + '.npz', 'wb')

    data = {'mu': mu, 'nVec': nVec_merged, 'sVec': sVec, 'times': times, 'traj': traj, 'D': D, 'selections': preservedSelections}
    if arg_list.covariance:
        data['cov'] = totalCov
    if arg_list.covAtEachTime:
        data['covAtEachTime'] = covAtEachTime
    if arg_list.clade:
        data['cladesMut'] = cladesOfPreservedMutations
    if arg_list.recombination:
        data['recombinations'] = preservedRecombinations
        data['mid_to_pos'] = Species.mid_to_pos

    np.savez_compressed(f, **data)
    f.close()

    end = timer()
    print('\nTotal time: %lfs, average per generation %lfs' % ((end - start), (end - start) / float(tEnd)))


def printUpdate(current, end, bar_length=20):
    """ Print an update of the simulation status. h/t Aravind Voggu on StackOverflow. """
    percent = float(current) / end
    dash    = ''.join(['-' for k in range(int(round(percent * bar_length)-1))]) + '>'
    space   = ''.join([' ' for k in range(bar_length - len(dash))])

    sys.stdout.write("\rSimulating: [{0}] {1}%".format(dash + space, int(round(percent * 100))))
    sys.stdout.write("\n")
    sys.stdout.flush()


def computeTrajectories(sVec, nVec):
    """ Computes allele frequency trajectories. """

    N = np.sum(nVec[0])
    L = len(sVec[0][0])
    traj = np.zeros((len(nVec), L), dtype=float)
    for t in range(len(nVec)):
        for i in range(L):
            traj[t, i] = np.sum([sVec[t][k][i] * nVec[t][k] for k in range(len(sVec[t]))]) / N
    return traj


def sitesToSeq(sites, length):
    """ Returns a sequence representation of a list of mutations. """
    seq = np.zeros(length, dtype=int)
    for site in sites:
        seq[site] = 1
    return seq

if __name__ == '__main__': main()
