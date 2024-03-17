#!/usr/bin/env python
# A simple Wright-Fisher simulation with an additive fitness model

import sys
import argparse
import numpy as np                          # numerical tools
import pickle
from timeit import default_timer as timer   # timer for performance
import random
import copy
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

    parser.add_argument('--num_init_mutations', type=int, default=30, help='number of initial mutations when random_init is enabled')
    parser.add_argument('--random_init', action='store_true', default=False, help='whether or not to start with random species of random frequencies')

    parser.add_argument('--diploid', action='store_true', default=False, help='whether or not simulate a diploid population (default is halploid)')

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
    if arg_list.random_init:
        num_init_mutations = arg_list.num_init_mutations
        L += num_init_mutations
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
        ploidy = 1
        positions = np.random.permutation(L)
        mid_to_pos = {i: pos for i, pos in enumerate(positions)} # record a mapping from mutation index to its position in the sequence
        selections = {} # record selections of all mutations; A selection for a mutation denotes the fitness contribution if all sets of chromosome contain that mutation. 
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
        def __init__(self, n=1, f=1., **kwargs):
            """ Initialize clone/provirus-specific variables. """
            self.n = n   # number of members
            self.f = f
            if 'mutations' in kwargs:
                self.mutations = kwargs['mutations']
            else:
                self.mutations = [[] for i in range(Species.ploidy)]
            if 'clade' in kwargs:
                self.clade = kwargs['clade']
            else:
                self.clade = 'ancestral'

        def __str__(self):
            rpr = ""
            rpr += f"A species n={self.n}, f={self.f}, chromosomes:\n"
            for chromosome in self.mutations:
                rpr += str(chromosome) + '\n'
            return rpr

        def get_chromosome_set(self):
            return set([tuple(sorted(chromosome)) for chromosome in self.mutations])

        def same_as(self, other):
            assert len(self.mutations) == len(other.mutations), f"Ploidy not the same! this {len(self.mutations)} that {len(other.mutations)}"
            return self.get_chromosome_set() == other.get_chromosome_set()
            # for this_muts, other_muts in zip(self.mutations, other.mutations):
            #     if set(this_muts) != set(other_muts):
            #         return False
            # return True

        def compute_fitness_from_mutations(self):
            fitness = 1
            for genome in self.mutations:
                for mid in genome:
                    # fitness += Species.selections[mid] / Species.ploidy
                    fitness += Species.selections[mid]
            self.f = fitness

        @classmethod
        def clone(cls, s):
            return cls(n=1, f=s.f, mutations=copy.deepcopy(s.mutations), clade=s.clade) # Return a new copy of the input Species

        def mate(self, others):
            """ Mate with other species and return new species. The total population size of others should equal to self.n"""
            assert self.n == np.sum([other.n for other in others])
            newSpecies = []
            for other in others:
                type_1_mating = np.random.binomial(other.n, 0.5)
                type_2_mating = other.n - type_1_mating
                if type_1_mating > 0:
                    newSpecies.append(Species(n=type_1_mating, mutations=[self.mutations[0], other.mutations[0]]))
                    newSpecies.append(Species(n=type_1_mating, mutations=[self.mutations[1], other.mutations[1]]))
                if type_2_mating > 0:
                    newSpecies.append(Species(n=type_2_mating, mutations=[self.mutations[0], other.mutations[1]]))
                    newSpecies.append(Species(n=type_2_mating, mutations=[self.mutations[1], other.mutations[0]]))
            newSpecies_dedup = []
            for candidate in newSpecies:
                found_duplicate = False
                for existing in newSpecies_dedup:
                    if existing.same_as(candidate):
                        existing.n += candidate.n
                        found_duplicate = True
                if not found_duplicate:
                    newSpecies_dedup.append(candidate)

            for s in newSpecies_dedup:
                s.compute_fitness_from_mutations()
            # if np.random.random() > 0.99:
            #     print(f"len(newSpecies)={len(newSpecies)}\tlen(newSpecies_dedup)={len(newSpecies_dedup)}")
            return newSpecies_dedup

        def mutate(self):
            """ Mutate and return self + new species."""

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
                        ploidy_index = np.random.randint(Species.ploidy)  # Randomly select a copy of genome to introduce the mutation
                        s.mutations[ploidy_index].append(Species.mid)
                        randomS = self.sample_selection()
                        randomS /= num_cooccuring_muts  # In case the collective fitness effect of cooccuring mutations gets too large
                        s.f += randomS / self.ploidy
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
                ploidy_index = np.random.randint(Species.ploidy)
                s.mutations[ploidy_index].append(Species.mid)
                Species.clades[Species.mid] = s.clade
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
                    pos_list = np.random.randint(0, L - 1, size=Species.ploidy)
                    this_left, this_right = self.split_mutations_at(pos_list)
                    other.permute_homolog_chromosome()
                    other_left, other_right = other.split_mutations_at(pos_list)

                    s1 = Species(n=1, mutations=[list(left.union(right)) for left, right in zip(this_left, other_right)])
                    s1.compute_fitness_from_mutations()
                    s2 = Species(n=1, mutations=[list(left.union(right)) for left, right in zip(other_left, this_right)])
                    s2.compute_fitness_from_mutations()
                    self.n -= 1
                    other.n -= 1
                    newSpecies.append(s1)
                    newSpecies.append(s2)
                    positions.append(pos_list)
            # if self.n > 0:
            #     newSpecies.append(self)
            # if other.n > 0:
            #     newSpecies.append(other)
            return newSpecies, positions

        def permute_homolog_chromosome(self):
            self.mutations = random.sample(self.mutations, len(self.mutations))

        def split_mutations_at(self, pos_list):
            left_list, right_list = [], []
            for genome, pos in zip(self.mutations, pos_list):
                left, right = set(), set()
                for m in genome:
                    p = Species.mid_to_pos[m]
                    if p <= pos:
                        left.add(m)
                    else:
                        right.add(m)
                left_list.append(left)
                right_list.append(right)
            return left_list, right_list

    # ^ SPECIES CLASS ^

    def contains_duplicate_species(pop):
        for i in range(len(pop)):
            for j in range(i):
                if pop[i].same_as(pop[j]):
                    print(pop[i])
                    print(pop[j])
                    return True
        return False

    if arg_list.diploid:
        Species.ploidy = 2

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
    elif arg_list.random_init:
        num_init_species = np.random.randint(3, 11)
        init_frequencies = np.random.random(size=num_init_species)
        init_frequencies /= np.sum(init_frequencies)
        for mid in range(num_init_mutations):
            Species.clades[mid] = 'ancestral'
            Species.selections[mid] = 0
        Species.mid = num_init_mutations

        nVec = [[]]
        cum_num = 0
        for i, freq in enumerate(init_frequencies):
            num = int(N * freq)
            if i == num_init_species - 1:
                num = N - cum_num
            nVec[0].append(num)
            cum_num += num
        assert cum_num == N

        mVec = [[[[] for _ in range(Species.ploidy)] for i in range(num_init_species)]]
        for i in range(num_init_mutations):
            identity = np.random.randint(num_init_species)
            ploidy_index = np.random.randint(Species.ploidy)
            mVec[0][identity][ploidy_index].append(i)

        pop  = [Species(n=num, mutations=mutations) for num, mutations in zip(nVec[0], mVec[0])]  # current population; list of Species
        recombinations = []  # list of recombination events, which records (t, s1, s2, pos)
    else:
        # Start with all sequences being wild-type
        pop  = [Species(n=N)]  # current population; list of Species
        mVec = [[[[] for _ in range(Species.ploidy)]]]  # list of mutations at each time point. mVec[t][i][j] is the jth mutation of ith species at time t
        nVec = [[N]]  # list of clone size at each time point
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
        elif arg_list.diploid:
            n = np.random.multinomial(N//2, pvals=p) # number of mating opportunities for each species
        else:
            n = np.random.multinomial(N, pvals=p) # selected number of each species

        # Update population size and mutate
        newPop = []

        if arg_list.schedule:
            # When a mutation is scheduled at current time; Select a species that will acquire the mutation
            if t in scheduled_mutation_selections:
                selected_i = np.random.choice(np.arange(len(pop)), 1, p=n / np.sum(n))
                newPop = [species for i, species in enumerate(pop) if i != selected_i] + pop[selected_i].mutate_as_scheduled(scheduled_mutation_selections[t])
        elif arg_list.diploid:
            # Sexual reproduction
            individuals = []
            for s in pop:
                for i in range(s.n):
                    individuals.append(Species.clone(s))
            matees = random.sample(individuals, N//2)
            random.shuffle(matees)  # Individuals to mate
            index = 0
            mated_pop = []
            for i, s in enumerate(pop):
                s.n = n[i] # set new number of mating opportunities for each species
                if s.n > 0:
                    for candidate in s.mate(matees[index:index + s.n]):
                        found_duplicate = False
                        for existing in mated_pop:
                            if existing.same_as(candidate):
                                found_duplicate = True
                                existing.n += candidate.n
                        if not found_duplicate:
                            mated_pop.append(candidate)
                    index += s.n
            # Mutate
            for s in mated_pop:
                mutatedPop = s.mutate()
                newPop += mutatedPop
        else:
            # Mutate
            for i in range(len(pop)):
                pop[i].n = n[i] # set new number of each species
                mutatedPop = pop[i].mutate()
                newPop += mutatedPop

        if contains_duplicate_species(newPop):
            print("After mutations, found duplicate species!")
            return

        if arg_list.recombination:
            for i, s1 in enumerate(pop):
                for j, s2 in enumerate(pop[i+1:]):
                    newSpecies_list, positions = s1.recombine(s2)
                    if len(newSpecies_list) == 0:
                        continue
                    for pos_list in positions:
                        recombinations.append((t, [tuple(_) for _ in s1.mutations], [tuple(_) for _ in s2.mutations], pos_list))
                    if Species.mid < 30:
                        # print("Before recombination: two species")
                        # print(s1)
                        # print(s2)
                        # print(f"After recombination: {len(newSpecies_list)} new species")
                        # for s in newSpecies_list:
                        #     print(s)
                        pass
                    for newSpecies in newSpecies_list:
                        # print(newSpecies)
                        # print(s)
                        # print("Same?", newSpecies.same_as(s))
                        foundDuplicate = False
                        for s in newPop:
                            if newSpecies.same_as(s):
                                s.n += newSpecies.n
                                foundDuplicate = True
                        if not foundDuplicate:
                            newPop.append(newSpecies)

        if contains_duplicate_species(newPop):
            print("After recombinations, found duplicate species!")
            return

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
        for t in [0, T - 1]:
            print(f'at t = {t}\tmVec[{t}] = {mVec[t]}\tnVec[{t}] = {nVec[t]}')

    # calculate frequencies of all occured mutations
    fMut = np.zeros((T, numMutations), dtype=float)
    for t in range(T):
        for k in range(len(mVec[t])):
            for chromosome in mVec[t][k]:
                for mid in chromosome:
                    fMut[t, mid] += nVec[t][k] / N / Species.ploidy

    # preserve mutations that once exceed a threshold frequency
    preservedMutations = [mid for mid in range(numMutations) if np.max(fMut[:, mid]) > arg_list.threshold]
    cladesOfPreservedMutations = [Species.clades[mid] for mid in preservedMutations]
    L = len(preservedMutations)
    for t in range(T):
        for k in range(len(mVec[t])):
            for p in range(Species.ploidy):
                mVec[t][k][p] = [mid for mid in mVec[t][k][p] if mid in preservedMutations]
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
            for p in range(Species.ploidy):
                sites = [midToSite[mid] for mid in mVec[t][k][p]]
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

    if verbose:
        print(f'np.sum(nVec_merged[0])={np.sum(nVec_merged[0])}')
    times = np.arange(T)
    traj = computeTrajectories(sVec, nVec_merged)
    D = MPL.computeD(traj, times, mu)

    # calculate the integrated covariance matrix
    if arg_list.covariance:
        q = 2
        totalCov = np.zeros((L, L), dtype=float)
        total_population_size = np.sum(nVec_merged[0])
        fVec_merged = [[count / total_population_size for count in nVec_merged[t]] for t in range(T)]
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
