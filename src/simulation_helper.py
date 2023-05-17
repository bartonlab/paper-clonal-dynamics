
#############  PACKAGES  #############

import numpy as np

from copy import deepcopy

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import seaborn as sns

import pickle

from scipy import stats

from tabulate import tabulate

import MPL as MPL
import reconstruct_clades as RC
import analyze_and_plot as AP
import figures as FIG
import lolipop_helper

DATA_DIR = './data'
FIG_DIR = './figures'
LOCAL_JOBS_DIR = './data/local_jobs'
SIMULATION_DIR = './data/simulation'
SIMULATION_RECONSTRUCTION_DIR = './data/simulation_reconstruction'
SELECTION_DIR = './data/selection'
JOB_DIR = './jobs'

# relative directories looking from shell scripts in JOB_DIR
DATA_DIR_REL = '../data'
SRC_DIR_REL = f'../src'
SIMULATION_DIR_REL = f'{DATA_DIR_REL}/simulation'
SELECTION_DIR_REL = f'{DATA_DIR_REL}/selection'

# Lolipop
LOLIPOP_JOBS_DIR = './data/lolipop/jobs'
LOLIPOP_INPUT_DIR = './data/lolipop/input'
LOLIPOP_OUTPUT_DIR = './data/lolipop/output'
LOLIPOP_PARSED_OUTPUT_DIR = './data/lolipop/parsed_output'
LOLIPOP_INFERENCE_DIR = './data/lolipop/inference'
# Lolipop directories looking from LOLIPOP_JOBS_DIR
LOLIPOP_INPUT_DIR_REL = '../input'
LOLIPOP_OUTPUT_DIR_REL = '../output'

# Evoracle
EVORACLE_SIMULATION_PARSED_OUTPUT_DIR = f'{DATA_DIR}/evoracle/simulation_parsed_output'
EVORACLE_DIR_SIMULATION_REL = f'{DATA_DIR_REL}/evoracle/simulation'
EVORACLE_SIMULATION_PARSED_OUTPUT_DIR_REL = f'{DATA_DIR_REL}/evoracle/simulation_parsed_output'

# haploSep
HAPLOSEP_INPUT_DIR = './data/haploSep/input'
HAPLOSEP_OUTPUT_DIR = './data/haploSep/output'

# LTEE
LTEE_JOB_DIR = './data/cluster_jobs'
CLUSTER_JOBS_DIR = './data/cluster_jobs'
LTEE_TRAJ_DIR = './data/LTEE_trajectories'
CLUSTERIZATION_OUTPUT_DIR = './data/clusterization_output'
RECONSTRUCTION_OUTPUT_DIR = './data/reconstruction_output'

# relative directories looking from shell scripts in LTEE_JOB_DIR
LTEE_TRAJ_DIR_REL = '../LTEE_trajectories'
CLUSTERIZATION_OUTPUT_DIR_REL = '../clusterization_output'

METHODS = FIG.METHODS

CMAP = FIG.CMAP

SIMULATION_PARAMS = {
    'mutation_rate': 0,
    'linear_reg': 1,
    'times': None,
}

############################################
#
# Parameters
#
############################################

class Params_WF:
    def __init__(self, s, i=None, N=1000, L=20, T=1000, mu=1e-3, start_n=0, num_trials=40):
        self.start_n, self.num_trials, self.s, self.i, self.N, self.L, self.T, self.mu = start_n, num_trials, s, i, N, L, T, mu

    def parse_all_parameters(self):
        return self.start_n, self.num_trials, self.s, self.i, self.N, self.L, self.T, self.mu


class Params:
    def __init__(self, start_n=0, num_trials=40, N=1000, T=1000, mu=2e-4, meanS=0.03, stdS=0.01, minS=0.01, maxS=0.04, threshold=0.05, uniform=False, recombination=False, recombination_rate=1e-8, controlled_genotype_fitness=False, genotype_fitness_increase_rate=1e-4, cooccurence=False, max_cooccuring_mutations=1, covariance=True, covAtEachTime=True, saveCompleteResults=False, verbose=True):

        self.start_n, self.num_trials, self.N, self.T, self.mu, self.meanS, self.stdS, self.minS, self.maxS, self.threshold, self.uniform, self.recombination, self.recombination_rate, self.cooccurence, self.max_cooccuring_mutations, self.controlled_genotype_fitness, self.genotype_fitness_increase_rate, self.covariance, self.covAtEachTime, self.saveCompleteResults, self.verbose = start_n, num_trials, N, T, mu, meanS, stdS, minS, maxS, threshold, uniform, recombination, recombination_rate, cooccurence, max_cooccuring_mutations, controlled_genotype_fitness, genotype_fitness_increase_rate, covariance, covAtEachTime, saveCompleteResults, verbose


    def parse_all_parameters(self):

        return self.start_n, self.num_trials, self.N, self.T, self.mu, self.meanS, self.stdS, self.minS, self.maxS, self.threshold, self.uniform, self.recombination, self.recombination_rate, self.cooccurence, self.max_cooccuring_mutations, self.controlled_genotype_fitness, self.genotype_fitness_increase_rate, self.covariance, self.covAtEachTime, self.saveCompleteResults, self.verbose


############################################
#
# Utility functions
#
############################################


def compute_allele_traj_from_geno_traj(geno_traj, genotypes):
    L = len(genotypes[0])
    T = len(geno_traj)
    allele_traj = np.zeros((T, L))
    for t in range(T):
        for k, genotype in enumerate(genotypes):
            for i, locus in enumerate(genotype):
                allele_traj[t, i] += geno_traj[t, k] * locus
    return allele_traj


def parse_filename_postfix(p, n=None):
    start_n, num_trials, N, T, mu, meanS, stdS, minS, maxS, threshold, uniform, recombination, recombination_rate, cooccurence, max_cooccuring_mutations, controlled_genotype_fitness, genotype_fitness_increase_rate, covariance, covAtEachTime, saveCompleteResults, verbose = p.parse_all_parameters()

    if uniform:
        if controlled_genotype_fitness:
            postfix = f'mu={mu}_gfir={genotype_fitness_increase_rate}_th={threshold}'
        elif recombination:
            postfix = f'min={minS}_max={maxS}_mu={mu}_r={recombination_rate}_th={threshold}'
        elif cooccurence:
            postfix = f'min={minS}_max={maxS}_mu={mu}_c={max_cooccuring_mutations}_th={threshold}'
        else:
            postfix = f'min={minS}_max={maxS}_mu={mu}_th={threshold}'
    else:
        if controlled_genotype_fitness:
            postfix = f'mu={mu}_gfir={genotype_fitness_increase_rate}_th={threshold}'
        elif recombination:
            postfix = f'mean={meanS}_std={stdS}_mu={mu}_r={recombination_rate}_th={threshold}'
        elif cooccurence:
            postfix = f'mean={meanS}_std={stdS}_mu={mu}_c={max_cooccuring_mutations}_th={threshold}'
        else:
            postfix = f'mean={meanS}_std={stdS}_mu={mu}_th={threshold}'
    if n is not None:
        postfix += f'_n={n}'

    return postfix


def parse_filename_postfix_for_WF_simulation(p, n):
    start_n, num_trials, s, i, N, L, T, mu = p.parse_all_parameters()
    postfix = f'L={L}_mu={mu}_n={n}'

    return postfix


############################################
#
# Scripts
#
############################################


def generate_local_shell_script(p, path2job=LOCAL_JOBS_DIR, jobname='simulation_local_submission'):

    start_n, num_trials, N, T, mu, meanS, stdS, minS, maxS, threshold, uniform, recombination, recombination_rate, cooccurence, max_cooccuring_mutations, controlled_genotype_fitness, genotype_fitness_increase_rate, covariance, covAtEachTime, saveCompleteResults, verbose = p.parse_all_parameters()

    command = generate_command(p)
    # jobname = f'simulation_mean={meanS}_std={stdS}_mu={mu}_th={threshold}_local_submission'
    fp = open(path2job + '/' + jobname + '.sh', 'w')
    fp.write('#!/bin/sh\n')
    fp.write(f'# {jobname}\n')
    fp.write(command)
    fp.close()


def generate_local_shell_script_general(jobname, command, path2job=LOCAL_JOBS_DIR):

    fp = open(path2job + '/' + jobname + '.sh', 'w')
    fp.write('#!/bin/sh\n')
    fp.write(f'# {jobname}\n')
    fp.write(command)
    fp.close()


def generate_local_shell_script_for_jobs(jobnames, path2job=LOCAL_JOBS_DIR, jobname='simulation_local_submission'):
    fp = open(path2job + '/' + jobname + '.sh', 'w')
    fp.write('#!/bin/sh\n')
    for job in jobnames:
        fp.write(f'sh {job}.sh\n')
    fp.close()


def generate_local_shell_script_for_WF_simulation(p, path2job=LOCAL_JOBS_DIR, jobname='simulation_local_WF_submission'):

    command = generate_command_for_WF_simulation(p)
    fp = open(path2job + '/' + jobname + '.sh', 'w')
    fp.write('#!/bin/sh\n')
    fp.write(f'# {jobname}\n')
    fp.write(command)
    fp.close()


def generate_command_for_WF_simulation(p):

    start_n, num_trials, s, i, N, L, T, mu = p.parse_all_parameters()
    command = ''
    for n in np.arange(start_n, start_n + num_trials):
        output_file = f'{SIMULATION_DIR_REL}/simulation_output_{parse_filename_postfix_for_WF_simulation(p, n)}'
        job_pars = {'-o': output_file,
                    '-N': N,
                    '-L': L,
                    '-T': T,
                    '-s': s,
                    '--mu': mu,
                    }
        if i is not None:
            job_pars['-i'] = i
        command += 'python ../../Wright-Fisher.py '
        command += ' '.join([k + ' ' + str(v) for k, v in job_pars.items()])
        command += '\n'
    return command


def generate_command(p):
    start_n, num_trials, N, T, mu, meanS, stdS, minS, maxS, threshold, uniform, recombination, recombination_rate, cooccurence, max_cooccuring_mutations, controlled_genotype_fitness, genotype_fitness_increase_rate, covariance, covAtEachTime, saveCompleteResults, verbose = p.parse_all_parameters()

    command = ''
    for n in np.arange(start_n, start_n + num_trials):
        output_file = f'{SIMULATION_DIR_REL}/simulation_output_{parse_filename_postfix(p, n)}'
        job_pars = {'-o': output_file,
                    '-N': N,
                    '-T': T,
                    '--mu': mu,
                    '--recombination_rate': recombination_rate,
                    '--max_cooccuring_mutations': max_cooccuring_mutations,
                    '--genotype_fitness_increase_rate': genotype_fitness_increase_rate,
                    '--meanS': meanS,
                    '--stdS': stdS,
                    '--minS': minS,
                    '--maxS': maxS,
                    '--threshold': threshold,
                    }
        if uniform:
            job_pars['--uniform'] = ''
        if recombination:
            job_pars['--recombination'] = ''
        if cooccurence:
            job_pars['--cooccurence'] = ''
        if controlled_genotype_fitness:
            job_pars['--controlled_genotype_fitness'] = ''
        if covariance:
            job_pars['--covariance'] = ''
        if covAtEachTime:
            job_pars['--covAtEachTime'] = ''
        if saveCompleteResults:
            job_pars['--saveCompleteResults'] = ''
        if verbose:
            job_pars['--verbose'] = ''
        command += 'python ../../simulate_evolution.py '
        command += ' '.join([k + ' ' + str(v) for k, v in job_pars.items()])
        command += '\n'
    return command


def generate_shell_script(path2job, jobname, command, hours=4, mem=16, ncpus=1, env='cov_est', local=False):
    """Generates a shell script to run on HPCC cluster, or locally."""

    if local:
        fp = open(path2job + '/' + jobname + '_local.sh', 'w')
        fp.write('#!/bin/sh\n')
        fp.write(command)
        fp.close()
    else:
        fp = open(path2job + '/' + jobname + '.sh', 'w')
        fp.write('#!/bin/bash -l\n')
        fp.write(f'#SBATCH --nodes=1\n#SBATCH --ntasks=1\n')
        fp.write(f'#SBATCH --cpus-per-task={ncpus}\n')
        fp.write(f'#SBATCH --mem={mem}G\n#SBATCH --time={hours}:00:00\n')
        fp.write(f'#SBATCH --output={jobname}.stdout\n')
        fp.write(f'#SBATCH --job-name="{jobname}"\n')
        fp.write(f'date\nsource activate {env}\n')
        fp.write(command)
        fp.write('\ndate')
        fp.close()


def generate_evoracle_scripts(p, env='evoracle', save_geno_traj=False, partition='intel', overwrite=False):
    start_n, num_trials, N, T, mu, meanS, stdS, minS, maxS, threshold, uniform, recombination, recombination_rate, cooccurence, max_cooccuring_mutations, controlled_genotype_fitness, genotype_fitness_increase_rate, covariance, covAtEachTime, saveCompleteResults, verbose = p.parse_all_parameters()
    postfix = parse_filename_postfix(p, 0)
    postfix = '_'.join(postfix.split('_')[:-1])
    job_prefix = f'evoracle_{postfix}'

    for n in range(num_trials):
        # if print_process:
        #     print(f'n={n}', end='\t')
        simulation = load_simulation(p, n)
        T, L = simulation['traj'].shape
        N = np.sum(simulation['nVec'][0])
        jobname = f'{job_prefix}_n={n}'
        command = ''
        job_pars = {'-n': n,
                    }
        if save_geno_traj:
            job_pars['--save_geno_traj'] = ''
        command += f'python {SRC_DIR_REL}/evoracle_batch_job.py '
        command += ' '.join([k + ' ' + str(v) for k, v in job_pars.items()])
        command += '\n'
        generate_shell_script(JOB_DIR, jobname, command, hours=12, env=env)


    jobname = f'{job_prefix}_mkdir'
    command = ''
    for n in range(num_trials):
        command += f'mkdir {EVORACLE_DIR_SIMULATION_REL}/n={n}\n'
    generate_shell_script(JOB_DIR, jobname, command, env=env)

    jobname = f'{job_prefix}_submission'
    command = ''
    for n in range(num_trials):
        if overwrite or not test_single_evoracle(n):
            command += f'sbatch -p {partition} {job_prefix}_n={n}.sh\n'
    generate_shell_script(JOB_DIR, jobname, command, env=env)


def test_single_evoracle(n):
    try:
        res = load_evoracle(n)
        for key in list(res.keys()):
            res[key]
        return True
    except:
        return False


############################################
#
# Simulation
#
############################################


def load_simulation(p, n, directory=SIMULATION_DIR):
    filename = f'{directory}/simulation_output_{parse_filename_postfix(p, n)}.npz'
    dic = np.load(filename, allow_pickle=True)
    return dic


def load_WF_simulation(p, n, directory=SIMULATION_DIR):
    filename = f'{directory}/simulation_output_{parse_filename_postfix_for_WF_simulation(p, n)}.npz'
    dic = np.load(filename, allow_pickle=True)
    return dic


def parse_geno_traj_for_a_simulation(p, n):

    simulation = load_simulation(p, n)
    T, L = simulation['traj'].shape
    N = np.sum(simulation['nVec'][0])
    genotypes, genotype_frequencies, pop = AP.getDominantGenotypeTrajectories(simulation, T=T, threshold=0, totalPopulation=N)
    return genotype_frequencies.T


def get_num_genotypes_and_num_alleles(p):

    start_n, num_trials, N, T, mu, meanS, stdS, minS, maxS, threshold, uniform, recombination, recombination_rate, cooccurence, max_cooccuring_mutations, controlled_genotype_fitness, genotype_fitness_increase_rate, covariance, covAtEachTime, saveCompleteResults, verbose = p.parse_all_parameters()

    num_genotypes, num_alleles = [], []
    for n in range(num_trials):
        simulation = load_simulation(p, n)
        T, L = simulation['traj'].shape
        N = np.sum(simulation['nVec'][0])
        genotypes, _, _ = AP.getDominantGenotypeTrajectories(simulation, threshold=0, T=T, totalPopulation=N)
        num_alleles.append(L)
        num_genotypes.append(len(genotypes))

    return num_genotypes, num_alleles


def get_num_recombinations(p):

    start_n, num_trials, N, T, mu, meanS, stdS, minS, maxS, threshold, uniform, recombination, recombination_rate, cooccurence, max_cooccuring_mutations, controlled_genotype_fitness, genotype_fitness_increase_rate, covariance, covAtEachTime, saveCompleteResults, verbose = p.parse_all_parameters()

    num_recombination = []
    for n in range(num_trials):
        simulation = load_simulation(p, n)
        if 'recombinations' in simulation:
            num_recombination.append(len(simulation['recombinations']))
        else:
            num_recombination.append(0)
    return num_recombination


def plot_a_simulation(simulation, alpha=0.6):

    traj, cov = simulation['traj'], simulation['cov']
    T, L = traj.shape
    N = np.sum(simulation['nVec'][0])
    fig, axes = plt.subplots(3, 1, figsize=(10, 8))
    seqOnceEmerged, freqOnceEmerged, pop = AP.getDominantGenotypeTrajectories(simulation, T=T, threshold=0, totalPopulation=N)

    plt.sca(axes[0])
    AP.plotTraj(traj, plotFigure=False, plotShow=False, alpha=alpha)
    plt.sca(axes[1])
    AP.plotTraj(np.array(freqOnceEmerged).T, plotFigure=False, plotShow=False, title='genotype frequency trajectories', alpha=alpha)
    plt.sca(axes[2])
    sns.heatmap(cov, center=0, cmap=CMAP, square=True)

    plt.show()



def plot_simulation(p, nCol=5, hspace=0.25, highlight=[], plot_clade_traj=True, winnerIsNew=False, wildtypeIsBackground=False, figsize=(20, 15), fontsize=15):

    start_n, num_trials, N, T, mu, meanS, stdS, minS, maxS, threshold, uniform, recombination, recombination_rate, cooccurence, max_cooccuring_mutations, controlled_genotype_fitness, genotype_fitness_increase_rate, covariance, covAtEachTime, saveCompleteResults, verbose = p.parse_all_parameters()

    if uniform:
        print(f'minS={minS}, maxS={maxS}')
    else:
        print(f'meanS={meanS}, stdS={stdS}')
    nRow = num_trials // nCol + (1 if num_trials % nCol != 0 else 0)
    fig, axes = plt.subplots(nRow, nCol, figsize=figsize)
    for n in np.arange(start_n, start_n + num_trials):
        plt.sca(axes[(n-start_n)//5, (n-start_n)%5])
        at_left = ((n-start_n) % 5 == 0)
        at_bottom = ((n-start_n) // 5 == nRow - 1)

        dic = load_simulation(p, n)

        if plot_clade_traj:
            seqOnceEmerged, freqOnceEmerged, pop = AP.getDominantGenotypeTrajectories(dic, T=T+1, threshold=0, totalPopulation=N)
            if winnerIsNew:
                clade, cladeFreq = AP.getCladeMutsWinnerIsNew(dic, seqOnceEmerged, freqOnceEmerged)
            elif wildtypeIsBackground:
                clade, cladeFreq = AP.getCladeMutsWildtypeIsBackground(seqOnceEmerged, freqOnceEmerged)
            else:
                clade, cladeFreq = AP.getCladeMuts(seqOnceEmerged, freqOnceEmerged)
            AP.plotCladeFreq(cladeFreq, plotLegend=False, plotShow=False)
            plt.ylabel('clade frequency' if at_left else '', fontsize=fontsize/1.3, labelpad=5)
            plt.title(f"n={n}; {len(cladeFreq)} clades; {len(dic['traj'][0])} muts", fontsize=fontsize)
        else:
            AP.plotTraj(dic['traj'], plotFigure=False, plotShow=False)
            plt.ylabel('Allele frequency' if at_left else '', fontsize=fontsize/1.3, labelpad=5)
            plt.title(f"n={n}; {len(dic['traj'][0])} muts", fontsize=fontsize)

        if not at_left:
            plt.yticks(ticks=[], labels=[])
        plt.xlabel('generation' if at_bottom else '', fontsize=fontsize/1.3, labelpad=5)
        if not at_bottom:
            plt.xticks(ticks=[], labels=[])
        if n in highlight:
            ax = plt.gca().axis()
            rec = Rectangle((ax[0]-0.7, ax[2]), ax[1]-ax[0], ax[3]-ax[2], fill=False, lw=3, color='red')
            rec = plt.gca().add_patch(rec)
            rec.set_clip_on(False)
        else:
            plt.gca().tick_params(color='gray', labelcolor='black')
            for spine in plt.gca().spines.values():
                spine.set_edgecolor('gray')
    plt.subplots_adjust(hspace=hspace)
    plt.show()


def plot_WF_simulation(p, nCol=5, hspace=0.25, highlight=[], figsize=(20, 18), fontsize=15):

    start_n, num_trials, s, i, N, L, T, mu = p.parse_all_parameters()

    nRow = num_trials // nCol + (1 if num_trials % nCol != 0 else 0)
    fig, axes = plt.subplots(nRow, nCol, figsize=figsize)
    for n in np.arange(start_n, start_n + num_trials):
        plt.sca(axes[(n-start_n)//5, (n-start_n)%5])
        at_left = ((n-start_n) % 5 == 0)
        at_bottom = ((n-start_n) // 5 == nRow - 1)

        dic = load_WF_simulation(p, n)

        AP.plotTraj(dic['traj'], plotFigure=False, plotShow=False)
        plt.ylabel('Allele frequency' if at_left else '', fontsize=fontsize/1.3, labelpad=5)
        plt.title(f"n={n}; {len(dic['traj'][0])} muts", fontsize=fontsize)

        if not at_left:
            plt.yticks(ticks=[], labels=[])
        plt.xlabel('generation' if at_bottom else '', fontsize=fontsize/1.3, labelpad=5)
        if not at_bottom:
            plt.xticks(ticks=[], labels=[])
        if n in highlight:
            ax = plt.gca().axis()
            rec = Rectangle((ax[0]-0.7, ax[2]), ax[1]-ax[0], ax[3]-ax[2], fill=False, lw=3, color='red')
            rec = plt.gca().add_patch(rec)
            rec.set_clip_on(False)
        else:
            plt.gca().tick_params(color='gray', labelcolor='black')
            for spine in plt.gca().spines.values():
                spine.set_edgecolor('gray')
    plt.subplots_adjust(hspace=hspace)
    plt.show()


############################################
#
# Lolipop
#
############################################


def create_tables_for_Lolipop(p):

    start_n, num_trials, N, T, mu, meanS, stdS, minS, maxS, threshold, uniform, recombination, recombination_rate, cooccurence, max_cooccuring_mutations, controlled_genotype_fitness, genotype_fitness_increase_rate, covariance, covAtEachTime, saveCompleteResults, verbose = p.parse_all_parameters()

    for n in range(num_trials):
        sim = load_simulation(p, n)
        filename = f'simulation_traj_{parse_filename_postfix(p, n)}.tsv'
        lolipop_helper.saveTrajectoriesToTables(sim['traj'], LOLIPOP_INPUT_DIR + f'/{filename}', sep='\t')


def generate_lolipop_command_for_simulated_data(p, n):
    filename = f'simulation_traj_{parse_filename_postfix(p, n)}.tsv'
    output_directory = f'{LOLIPOP_OUTPUT_DIR_REL}/n={n}'
    command = f"lolipop lineage --input {LOLIPOP_INPUT_DIR_REL}/{filename} --output {output_directory}"
    return command


def create_job_file_for_Lolipop(p, jobname='lolipop_simulated_data.sh'):

    start_n, num_trials, N, T, mu, meanS, stdS, minS, maxS, threshold, uniform, recombination, recombination_rate, cooccurence, max_cooccuring_mutations, controlled_genotype_fitness, genotype_fitness_increase_rate, covariance, covAtEachTime, saveCompleteResults, verbose = p.parse_all_parameters()

    with open(f'{LOLIPOP_JOBS_DIR}/{jobname}', 'w') as fp:
        fp.write('#!/bin/sh\n')
        for n in range(num_trials):
            fp.write(generate_lolipop_command_for_simulated_data(p, n) + '\n')


def save_parsed_output_for_Lolipop(p, overwrite=False):

    start_n, num_trials, N, T, mu, meanS, stdS, minS, maxS, threshold, uniform, recombination, recombination_rate, cooccurence, max_cooccuring_mutations, controlled_genotype_fitness, genotype_fitness_increase_rate, covariance, covAtEachTime, saveCompleteResults, verbose = p.parse_all_parameters()

    for n in range(num_trials):
        if not overwrite and test_lolipop_parsed_output(p, n):
            continue
        filename = f'simulation_traj_{parse_filename_postfix(p, n)}'
        raw_output_directory = f'{LOLIPOP_OUTPUT_DIR}/n={n}'
        saveFile = f'{LOLIPOP_PARSED_OUTPUT_DIR}/{filename}'
        lolipop_helper.parseOutput(raw_output_directory, filename, saveFile=saveFile)
        print(f'n={n}', end='\t')


def load_Lolipop_inference_on_simulated_data(p, directory=LOLIPOP_INFERENCE_DIR):
    output = f'simulation_Lolipop_inference_{parse_filename_postfix(p)}'
    file = f'{directory}/{output}.npz'
    return np.load(file, allow_pickle=True)


def test_lolipop_parsed_output(p, n):
    try:
        res = load_lolipop_parsed_output(p, n)
        for key in list(res.keys()):
            res[key]
        return True
    except:
        return False


def load_lolipop_parsed_output(p, n):
    filename = f'simulation_traj_{parse_filename_postfix(p, n)}'
    file = f'{LOLIPOP_PARSED_OUTPUT_DIR}/{filename}.npz'
    return np.load(file, allow_pickle=True)


def save_Lolipop_inference_on_simulated_date(p, muForInference=0):
    start_n, num_trials, N, T, mu, meanS, stdS, minS, maxS, threshold, uniform, recombination, recombination_rate, cooccurence, max_cooccuring_mutations, controlled_genotype_fitness, genotype_fitness_increase_rate, covariance, covAtEachTime, saveCompleteResults, verbose = p.parse_all_parameters()
    (cov_Lolipop, selection_Lolipop, fitness_Lolipop, covError_Lolipop) = lolipop_helper.inferForSimulations(uniform=uniform, muForInference=muForInference, mu=mu, minS=minS, maxS=maxS, meanS=meanS, stdS=stdS, lolipop_parsed_output_dir=LOLIPOP_PARSED_OUTPUT_DIR, simulation_output_dir=SIMULATION_DIR)
    results = {
        'int_cov': cov_Lolipop,
        'selection': selection_Lolipop,
        'fitness': fitness_Lolipop,
        'cov_error': covError_Lolipop,
    }
    filename = f'simulation_Lolipop_inference_{parse_filename_postfix(p)}'
    np.savez_compressed(f'{LOLIPOP_INFERENCE_DIR}/{filename}', **results)


############################################
#
# Evoracle
#
############################################


def load_evoracle(n, directory=EVORACLE_SIMULATION_PARSED_OUTPUT_DIR):
    file = f'{directory}/evoracle_parsed_output_n={n}.npz'
    return np.load(file, allow_pickle=True)


def load_simulation_results_for_evoracle(p):
    results_evoracle = []
    for n in range(p.num_trials):
        res = load_evoracle(n)
        results_evoracle.append(res)
    return results_evoracle


def check_simulation_results_for_evoracle(p, results_evoracle=None):
    if results_evoracle is None:
        results_evoracle = load_simulation_results_for_evoracle()
    columns = ['n', 'MAE_traj', '# geno', 'Spearmanr_s', 'Spearmanr_s_by_geno_traj']
    rows = []
    for n in range(p.num_trials):
        res = results_evoracle[n]
        if res['traj'] is None:
            row = [n, None, None, None, None]
        else:
            selection = load_simulation(p, n)['selections']
            a, b = res['traj'], res['traj_from_geno_traj']
            num_genotypes = len(res['geno_traj'][0])
            spearmanr = stats.spearmanr(res['selection'], selection)[0]
            spearmanr_from_geno_traj = stats.spearmanr(res['selection_from_geno_traj'], selection)[0]
            row = [n, RC.MAE(a, b), num_genotypes, spearmanr, spearmanr_from_geno_traj]
        rows.append(row)
    print(tabulate(rows, columns, tablefmt='plain'))


############################################
#
# haploSep
#
############################################


def save_simulation_traj_for_haplosep(p):
    for n in range(p.num_trials):
        simulation = load_simulation(p, n)
        traj = simulation['traj']
        filename = f'simulation_n={n}.npy'
        save_traj_for_haplosep(traj, filename)


def save_traj_for_haplosep(traj, filename):
    np.save(f"{HAPLOSEP_INPUT_DIR}/{filename}", traj.T)


def parse_simulation_results_for_haplosep(p, print_process=True, save=True):
    results_haplosep = []
    for n in range(p.num_trials):
        if print_process:
            print(f'n={n}', end='\t')
        output = f'haplosep_output_n={n}.npz'
        traj = load_simulation(p, n)['traj']
        res = infer_with_haplosep_output(output, traj)
        results_haplosep.append(res)
    if save:
        np.savez_compressed(f"{HAPLOSEP_OUTPUT_DIR}/haplosep_results_for_simulation.npz", results=results_haplosep)
    return results_haplosep


def parse_haplosep_output(filename):

    res = np.load(f"{HAPLOSEP_OUTPUT_DIR}/{filename}", allow_pickle=True)
    dic = res['arr_0'].item()
    genotypes = dic['haploStr'].T.astype(int)
    geno_traj = dic['haploFrq'].T
    return genotypes, geno_traj


def load_simulation_results_for_haplosep():

    return np.load(f"{HAPLOSEP_OUTPUT_DIR}/haplosep_results_for_simulation.npz", allow_pickle=True)['results']


def infer_with_haplosep_output(output, traj, params=SIMULATION_PARAMS, compute_population_fitness=False):

    mu = params['mutation_rate']
    reg = params['linear_reg']
    times = params['times']

    T, L = traj.shape
    result = {}

    try:
        genotypes, geno_traj = parse_haplosep_output(output)
    except:
        results = {
            'genotypes': None,
            'geno_traj': None,
            'traj': None,
            'traj_from_geno_traj': None,
            'int_cov': None,
            'selection': None,
            'selection_from_geno_traj': None,
        }
        return results
    if times is None:
        times = np.arange(0, len(geno_traj))

    int_cov = MPL.integrateCovarianceFromStableGenotypes(genotypes, geno_traj, times)
    traj_from_geno_traj = compute_allele_traj_from_geno_traj(geno_traj, genotypes)
    D = MPL.computeD(traj, times, mu)
    D_from_geno_traj = MPL.computeD(traj_from_geno_traj, times, mu)
    selection = MPL.inferSelection(int_cov, D, reg * np.identity(L))
    selection_from_geno_traj = MPL.inferSelection(int_cov, D_from_geno_traj, reg * np.identity(L))

    result = {
        'genotypes': genotypes,
        'geno_traj': geno_traj,
        'traj': traj,
        'traj_from_geno_traj': traj_from_geno_traj,
        'int_cov': int_cov,
        'selection': selection,
        'selection_from_geno_traj': selection_from_geno_traj,
    }

    if compute_population_fitness:
        fitness = MPL.computePopulationFitnessFromSelection(traj, selection)
        fitness_from_geno_traj = MPL.computePopulationFitnessFromSelection(traj_from_geno_traj, selection_from_geno_traj)
        result['fitness'] = fitness
        result['fitness_from_geno_traj'] = fitness_from_geno_traj

    return result


def check_simulation_results_for_haplosep(p, results_haplosep=None):
    if results_haplosep is None:
        results_haplosep = load_simulation_results_for_haplosep()
    columns = ['n', 'MAE_traj', '# geno', 'Spearmanr_s', 'Spearmanr_s_by_geno_traj']
    rows = []
    for n in range(p.num_trials):
        res = results_haplosep[n]
        if res['traj'] is None:
            row = [n, None, None, None, None]
        else:
            selection = load_simulation(p, n)['selections']
            a, b = res['traj'], res['traj_from_geno_traj']
            num_genotypes = len(res['geno_traj'][0])
            spearmanr = stats.spearmanr(res['selection'], selection)[0]
            spearmanr_from_geno_traj = stats.spearmanr(res['selection_from_geno_traj'], selection)[0]
            row = [n, RC.MAE(a, b), num_genotypes, spearmanr, spearmanr_from_geno_traj]
        rows.append(row)
    print(tabulate(rows, columns, tablefmt='plain'))


############################################
#
# Parsing performance results
#
############################################


def get_reconstruction_of_simulation(simulation, mu=2e-4, useEffectiveMu=True, meanReadDepth=1000, verbose=False, debug=False, plot=False, thFixed=0.99):
    reconstruction = RC.CladeReconstruction(simulation['traj'], mu=mu, useEffectiveMu=useEffectiveMu, meanReadDepth=meanReadDepth, verbose=verbose, plot=plot, debug=debug)

    reconstruction.setParamsForClusterization(weightByBothVariance=False, weightBySmallerVariance=False, weightBySmallerInterpolatedVariance=True)
    reconstruction.clusterMutations()
    reconstruction.setParamsForReconstruction(thFixed=thFixed, thExtinct=0, numClades=None, thLogProbPerTime=10)
    reconstruction.checkForSeparablePeriodAndReconstruct()
    reconstruction.setParamsForEvaluation(intCov=simulation['cov'], selection=simulation['selections'], fitness=np.sum(simulation['selections'] * simulation['traj'], axis=1))
    evaluation = reconstruction.evaluate()
    return reconstruction, evaluation


def get_reconstruction_from_traj(traj, mu=2e-4, useEffectiveMu=True, meanReadDepth=1000, verbose=False, plot=False, thFixed=0.99):
    reconstruction = RC.CladeReconstruction(traj, mu=mu, useEffectiveMu=useEffectiveMu, meanReadDepth=meanReadDepth, verbose=verbose, plot=plot)

    reconstruction.setParamsForClusterization(weightByBothVariance=False, weightBySmallerVariance=True)
    reconstruction.clusterMutations()
    reconstruction.setParamsForReconstruction(thFixed=thFixed, thExtinct=0, numClades=None, percentMutsToIncludeInMajorClades=95, thLogProbPerTime=10)
    reconstruction.checkForSeparablePeriodAndReconstruct()
    return reconstruction


def get_inference_of_true_cov_and_SL_for_simulation(simulation, reg=1, mu=0, useEffectiveMu=False, meanReadDepth=1000, verbose=False, plot=False):
    reconstruction = RC.CladeReconstruction(simulation['traj'], mu=mu, useEffectiveMu=useEffectiveMu,
                                            meanReadDepth=meanReadDepth, verbose=verbose, plot=plot)

    reconstruction.setParamsForEvaluation(intCov=simulation['cov'], selection=simulation['selections'], fitness=np.sum(simulation['selections'] * simulation['traj'], axis=1))

    return reconstruction.evaluateInferenceForTrueCovAndSL(reg=reg)


def print_performance_for_a_simulation(simulation, methods=['SL', 'true_cov']):

    (MAE_cov, Spearmanr_cov, Pearsonr_cov, MAE_selection, Spearmanr_selection, Pearsonr_selection,   MAE_fitness, Spearmanr_fitness, Pearsonr_fitness) = parse_performance_for_a_simulation(simulation, methods=methods, computePerfOnGenotypeFitness=True)

    for method in methods:
        print(method)
        print('MAE of fitness : %.4f' % MAE_fitness[method][0])
        print('Spearmanr of fitness : %.4f' % Spearmanr_fitness[method][0])
        print('Pearsonr of fitness : %.4f' % Pearsonr_fitness[method][0])


def parse_performance_for_a_simulation(simulation, reg=1, methods=METHODS, muForInference=0, reference='true', include_Lolipop=True, computePerfOnGenotypeFitness=False):

    MAE_cov, Spearmanr_cov, MAE_selection, Spearmanr_selection = {m: [] for m in methods}, {m: [] for m in methods}, {m: [] for m in methods}, {m: [] for m in methods}
    Pearsonr_cov, Pearsonr_selection = {m: [] for m in methods}, {m: [] for m in methods}
    MAE_fitness, Spearmanr_fitness, Pearsonr_fitness = {m: [] for m in methods}, {m: [] for m in methods}, {m: [] for m in methods}

    T, L = simulation['traj'].shape
    N = np.sum(simulation['nVec'][0])

    if 'recovered' in methods:
        rec, evaluation = get_reconstruction_of_simulation(simulation, mu=muForInference, useEffectiveMu=False)
        perf = rec.summary
    else:
        perf = get_inference_of_true_cov_and_SL_for_simulation(simulation, mu=muForInference, reg=reg, useEffectiveMu=False)

    if computePerfOnGenotypeFitness:
        genotypes, _, _ = AP.getDominantGenotypeTrajectories(simulation, threshold=0, T=T, totalPopulation=N)
        genotype_fitness = RC.computeFitnessOfGenotypes(genotypes, perf[reference][2])

    for method in methods:
        if method == 'Lolipop':
            continue

        MAE_cov[method].append(RC.MAE(perf[reference][1], perf[method][1]))
        Spearmanr_cov[method].append(RC.Spearmanr(perf[reference][1], perf[method][1]))
        Pearsonr_cov[method].append(RC.Pearsonr(perf[reference][1], perf[method][1]))
        MAE_selection[method].append(RC.MAE(perf[reference][2], perf[method][2]))
        Spearmanr_selection[method].append(RC.Spearmanr(perf[reference][2], perf[method][2]))
        Pearsonr_selection[method].append(RC.Pearsonr(perf[reference][2], perf[method][2]))

        if computePerfOnGenotypeFitness:
            fitness = RC.computeFitnessOfGenotypes(genotypes, perf[method][2])
            MAE_fitness[method].append(RC.MAE(genotype_fitness, fitness))
            Spearmanr_fitness[method].append(RC.Spearmanr(genotype_fitness, fitness))
            Pearsonr_fitness[method].append(RC.Pearsonr(genotype_fitness, fitness))

    if computePerfOnGenotypeFitness:
        return MAE_cov, Spearmanr_cov, Pearsonr_cov, MAE_selection, Spearmanr_selection, Pearsonr_selection, MAE_fitness, Spearmanr_fitness, Pearsonr_fitness
    else:
        return MAE_cov, Spearmanr_cov, Pearsonr_cov, MAE_selection, Spearmanr_selection, Pearsonr_selection


def parse_performance_for_ps(ps, reg=1, methods=['true_cov', 'recovered', 'SL']):

    MAE_selection_dic = {_: [] for _ in methods}
    Spearmanr_selection_dic = {_: [] for _ in methods}
    Pearsonr_selection_dic = {_: [] for _ in methods}
    MAE_fitness_dic = {_: [] for _ in methods}
    Spearmanr_fitness_dic = {_: [] for _ in methods}
    Pearsonr_fitness_dic = {_: [] for _ in methods}

    for i, p in enumerate(ps):
        MAE_cov, Spearmanr_cov, Pearsonr_cov, MAE_selection, Spearmanr_selection, Pearsonr_selection, MAE_fitness, Spearmanr_fitness, Pearsonr_fitness = parse_performance_on_simulated_data(p, reg=reg, methods=methods, muForInference=0, reference='true', include_Lolipop=False, computePerfOnGenotypeFitness=True, print_process=False)
        for method in methods:
            MAE_selection_dic[method].append(np.mean(MAE_selection[method]))
            MAE_fitness_dic[method].append(np.mean(MAE_fitness[method]))
            Spearmanr_selection_dic[method].append(np.mean(Spearmanr_selection[method]))
            Spearmanr_fitness_dic[method].append(np.mean(Spearmanr_fitness[method]))
            Pearsonr_selection_dic[method].append(np.mean(Pearsonr_selection[method]))
            Pearsonr_fitness_dic[method].append(np.mean(Pearsonr_fitness[method]))

    return MAE_selection_dic, Spearmanr_selection_dic, Pearsonr_selection_dic, MAE_fitness_dic, Spearmanr_fitness_dic, Pearsonr_fitness_dic


def save_reconstructions_on_simulated_data(p, reg=1, methods=METHODS, muForInference=0, reference='true', overwrite=False):

    start_n, num_trials, N, T, mu, meanS, stdS, minS, maxS, threshold, uniform, recombination, recombination_rate, cooccurence, max_cooccuring_mutations, controlled_genotype_fitness, genotype_fitness_increase_rate, covariance, covAtEachTime, saveCompleteResults, verbose = p.parse_all_parameters()

    for n in range(num_trials):
        if not overwrite:
            try:
                load_reconstruction_for_a_simulation(p, n)
                continue
            except:
                pass
        simulation = load_simulation(p, n)
        T, L = simulation['traj'].shape
        N = np.sum(simulation['nVec'][0])
        reconstruction, evaluation = get_reconstruction_of_simulation(simulation, mu=muForInference, useEffectiveMu=False)
        output = f'{SIMULATION_RECONSTRUCTION_DIR}/simulation_reconstruction_{parse_filename_postfix(p, n)}.obj'
        fp = open(output, 'wb')
        pickle.dump(reconstruction, fp, protocol=4)
        fp.close()


def load_reconstruction_for_a_simulation(p, n, directory=SIMULATION_RECONSTRUCTION_DIR):
    file = f'{directory}/simulation_reconstruction_{parse_filename_postfix(p, n)}.obj'
    with open(file, 'rb') as fp:
        return pickle.load(fp)


def parse_performance_on_simulated_data(p, directory=SIMULATION_RECONSTRUCTION_DIR, reg=1, methods=METHODS, muForInference=0, reference='true', include_Lolipop=True, include_Evoracle=True, include_haploSep=False, use_inferred_geno_traj_for_Evoracle=False, use_inferred_geno_traj_for_haploSep=False, computePerfOnGenotypeFitness=False, print_process=True):

    start_n, num_trials, N, T, mu, meanS, stdS, minS, maxS, threshold, uniform, recombination, recombination_rate, cooccurence, max_cooccuring_mutations, controlled_genotype_fitness, genotype_fitness_increase_rate, covariance, covAtEachTime, saveCompleteResults, verbose = p.parse_all_parameters()

    MAE_cov, Spearmanr_cov, Pearsonr_cov, MAE_selection, Spearmanr_selection, Pearsonr_selection, MAE_fitness, Spearmanr_fitness, Pearsonr_fitness = {m: [] for m in methods}, {m: [] for m in methods}, {m: [] for m in methods}, {m: [] for m in methods}, {m: [] for m in methods}, {m: [] for m in methods}, {m: [] for m in methods}, {m: [] for m in methods}, {m: [] for m in methods}

    if include_Lolipop:
        if print_process:
            print(f'Parsing for Lolipop...', end='\t')
        dic = load_Lolipop_inference_on_simulated_data(p)
        cov_Lolipop, selection_Lolipop, fitness_Lolipop = dic['int_cov'], dic['selection'], dic['fitness']
        # (cov_Lolipop, selection_Lolipop, fitness_Lolipop,
        #  covError_Lolipop) = lolipop_helper.inferForSimulations(uniform=uniform,
        #     muForInference=muForInference, mu=mu, minS=minS, maxS=maxS, meanS=meanS, stdS=stdS,
        #     lolipop_parsed_output_dir=LOLIPOP_PARSED_OUTPUT_DIR,
        #     simulation_output_dir=SIMULATION_DIR)

    if include_Evoracle:
        if print_process:
            print(f'Parsing for Evoracle...', end='\t')
        results_Evoracle = load_simulation_results_for_evoracle(p)
        cov_Evoracle = [res['int_cov'] for res in results_Evoracle]
        if use_inferred_geno_traj_for_Evoracle:
            selection_Evoracle = [res['selection_from_geno_traj'] for res in results_Evoracle]
        else:
            selection_Evoracle = [res['selection'] for res in results_Evoracle]

    if include_haploSep:
        if print_process:
            print(f'Parsing for haploSep...', end='\t')
        results_haploSep = load_simulation_results_for_haplosep()
        cov_haploSep = [res['int_cov'] for res in results_haploSep]
        if use_inferred_geno_traj_for_haploSep:
            selection_haploSep = [res['selection_from_geno_traj'] for res in results_haploSep]
        else:
            selection_haploSep = [res['selection'] for res in results_haploSep]

    print(f'Parsing for others...', end='\t')
    for n in range(num_trials):
        if print_process:
            print(f'n={n}', end='\t')
        simulation = load_simulation(p, n)
        T, L = simulation['traj'].shape
        N = np.sum(simulation['nVec'][0])

        if 'recovered' in methods:
            reconstruction = load_reconstruction_for_a_simulation(p, n, directory=directory)
            # rec, evaluation = get_reconstruction_of_simulation(simulation, mu=muForInference, useEffectiveMu=False)
            perf = reconstruction.summary
        else:
            perf = get_inference_of_true_cov_and_SL_for_simulation(simulation, mu=muForInference, reg=reg, useEffectiveMu=False)

        if computePerfOnGenotypeFitness:
            genotypes, _, _ = AP.getDominantGenotypeTrajectories(simulation, threshold=0, T=T, totalPopulation=N)
            genotype_fitness = RC.computeFitnessOfGenotypes(genotypes, perf[reference][2])

        for method in methods:
            if method in ['Lolipop', 'Evoracle', 'haploSep']:
                continue

            MAE_cov[method].append(RC.MAE(perf[reference][1], perf[method][1]))
            Spearmanr_cov[method].append(RC.Spearmanr(perf[reference][1], perf[method][1]))
            Pearsonr_cov[method].append(RC.Pearsonr(perf[reference][1], perf[method][1]))

            MAE_selection[method].append(RC.MAE(perf[reference][2], perf[method][2]))
            Spearmanr_selection[method].append(RC.Spearmanr(perf[reference][2], perf[method][2]))
            Pearsonr_selection[method].append(RC.Pearsonr(perf[reference][2], perf[method][2]))

            if computePerfOnGenotypeFitness:
                fitness = RC.computeFitnessOfGenotypes(genotypes, perf[method][2])
                MAE_fitness[method].append(RC.MAE(genotype_fitness, fitness))
                Spearmanr_fitness[method].append(RC.Spearmanr(genotype_fitness, fitness))
                Pearsonr_fitness[method].append(RC.Pearsonr(genotype_fitness, fitness))

        method = 'Lolipop'
        if include_Lolipop:
            MAE_cov[method].append(RC.MAE(perf[reference][1], cov_Lolipop[n]))
            Spearmanr_cov[method].append(RC.Spearmanr(perf[reference][1], cov_Lolipop[n]))
            Pearsonr_cov[method].append(RC.Pearsonr(perf[reference][1], cov_Lolipop[n]))
            MAE_selection[method].append(RC.MAE(perf[reference][2], selection_Lolipop[n]))
            Spearmanr_selection[method].append(RC.Spearmanr(perf[reference][2], selection_Lolipop[n]))
            Pearsonr_selection[method].append(RC.Pearsonr(perf[reference][2], selection_Lolipop[n]))
            if computePerfOnGenotypeFitness:
                fitness = RC.computeFitnessOfGenotypes(genotypes, selection_Lolipop[n])
                MAE_fitness[method].append(RC.MAE(genotype_fitness, fitness))
                Spearmanr_fitness[method].append(RC.Spearmanr(genotype_fitness, fitness))
                Pearsonr_fitness[method].append(RC.Pearsonr(genotype_fitness, fitness))
        elif 'Lolipop' in methods:
            MAE_cov[method].append(0)
            Spearmanr_cov[method].append(0)
            Pearsonr_cov[method].append(0)
            MAE_selection[method].append(0)
            Spearmanr_selection[method].append(0)
            Pearsonr_selection[method].append(0)
            if computePerfOnGenotypeFitness:
                MAE_fitness[method].append(0)
                Spearmanr_fitness[method].append(0)
                Pearsonr_fitness[method].append(0)

        method = 'Evoracle'
        if include_Evoracle:
            MAE_cov[method].append(RC.MAE(perf[reference][1], cov_Evoracle[n]))
            Spearmanr_cov[method].append(RC.Spearmanr(perf[reference][1], cov_Evoracle[n]))
            Pearsonr_cov[method].append(RC.Pearsonr(perf[reference][1], cov_Evoracle[n]))
            MAE_selection[method].append(RC.MAE(perf[reference][2], selection_Evoracle[n]))
            Spearmanr_selection[method].append(RC.Spearmanr(perf[reference][2], selection_Evoracle[n]))
            Pearsonr_selection[method].append(RC.Pearsonr(perf[reference][2], selection_Evoracle[n]))
            if computePerfOnGenotypeFitness:
                fitness = RC.computeFitnessOfGenotypes(genotypes, selection_Evoracle[n])
                MAE_fitness[method].append(RC.MAE(genotype_fitness, fitness))
                Spearmanr_fitness[method].append(RC.Spearmanr(genotype_fitness, fitness))
                Pearsonr_fitness[method].append(RC.Pearsonr(genotype_fitness, fitness))

        method = 'haploSep'
        if include_haploSep:
            # Skip cases where haploSep does not run successfully
            if cov_haploSep[n] is None or selection_haploSep[n] is None:
                continue
            MAE_cov[method].append(RC.MAE(perf[reference][1], cov_haploSep[n]))
            Spearmanr_cov[method].append(RC.Spearmanr(perf[reference][1], cov_haploSep[n]))
            Pearsonr_cov[method].append(RC.Pearsonr(perf[reference][1], cov_haploSep[n]))
            MAE_selection[method].append(RC.MAE(perf[reference][2], selection_haploSep[n]))
            Spearmanr_selection[method].append(RC.Spearmanr(perf[reference][2], selection_haploSep[n]))
            Pearsonr_selection[method].append(RC.Pearsonr(perf[reference][2], selection_haploSep[n]))
            if computePerfOnGenotypeFitness:
                fitness = RC.computeFitnessOfGenotypes(genotypes, selection_haploSep[n])
                MAE_fitness[method].append(RC.MAE(genotype_fitness, fitness))
                Spearmanr_fitness[method].append(RC.Spearmanr(genotype_fitness, fitness))
                Pearsonr_fitness[method].append(RC.Pearsonr(genotype_fitness, fitness))

    if print_process:
        print()
    if computePerfOnGenotypeFitness:
        return MAE_cov, Spearmanr_cov, Pearsonr_cov, MAE_selection, Spearmanr_selection, Pearsonr_selection, MAE_fitness, Spearmanr_fitness, Pearsonr_fitness
    else:
        return MAE_cov, Spearmanr_cov, Pearsonr_cov, MAE_selection, Spearmanr_selection, Pearsonr_selection


def parse_performance_for_ps_for_WF_simulation(ps, reg=1, muForInference=1e-3, methods=['true_cov', 'recovered', 'SL']):

    MAE_selection_dic = {_: [] for _ in methods}
    Spearmanr_selection_dic = {_: [] for _ in methods}
    Pearsonr_selection_dic = {_: [] for _ in methods}
    MAE_fitness_dic = {_: [] for _ in methods}
    Spearmanr_fitness_dic = {_: [] for _ in methods}
    Pearsonr_fitness_dic = {_: [] for _ in methods}

    for i, p in enumerate(ps):
        MAE_cov, Spearmanr_cov, Pearsonr_cov, MAE_selection, Spearmanr_selection, Pearsonr_selection, MAE_fitness, Spearmanr_fitness, Pearsonr_fitness = parse_performance_on_simulated_data_for_WF_simulation(p, reg=reg, methods=methods, muForInference=muForInference, reference='true', computePerfOnGenotypeFitness=True, print_process=True)
        for method in methods:
            MAE_selection_dic[method].append(np.mean(MAE_selection[method]))
            MAE_fitness_dic[method].append(np.mean(MAE_fitness[method]))
            Spearmanr_selection_dic[method].append(np.mean(Spearmanr_selection[method]))
            Spearmanr_fitness_dic[method].append(np.mean(Spearmanr_fitness[method]))
            Pearsonr_selection_dic[method].append(np.mean(Pearsonr_selection[method]))
            Pearsonr_fitness_dic[method].append(np.mean(Pearsonr_fitness[method]))

    return MAE_selection_dic, Spearmanr_selection_dic, Pearsonr_selection_dic, MAE_fitness_dic, Spearmanr_fitness_dic, Pearsonr_fitness_dic


def parse_performance_on_simulated_data_for_WF_simulation(p, reg=1, methods=METHODS, muForInference=1e-3, reference='true', computePerfOnGenotypeFitness=False, print_process=True):

    start_n, num_trials, s, i, N, L, T, mu = p.parse_all_parameters()

    MAE_cov, Spearmanr_cov, MAE_selection, Spearmanr_selection = {m: [] for m in methods}, {m: [] for m in methods}, {m: [] for m in methods}, {m: [] for m in methods}
    Pearsonr_cov, Pearsonr_selection = {m: [] for m in methods}, {m: [] for m in methods}

    MAE_fitness, Spearmanr_fitness, Pearsonr_fitness = {m: [] for m in methods}, {m: [] for m in methods}, {m: [] for m in methods}

    for n in range(num_trials):
        if print_process:
            print(f'n={n}', end='\t')
        simulation = load_WF_simulation(p, n)
        T, L = simulation['traj'].shape
        N = np.sum(simulation['nVec'][0])

        if 'recovered' in methods:
            rec, evaluation = get_reconstruction_of_simulation(simulation, mu=muForInference, useEffectiveMu=False)
            perf = rec.summary
        else:
            perf = get_inference_of_true_cov_and_SL_for_simulation(simulation, mu=muForInference, reg=reg, useEffectiveMu=False)

        if computePerfOnGenotypeFitness:
            genotypes, _, _ = AP.getDominantGenotypeTrajectories(simulation, threshold=0.05, T=T, totalPopulation=N)
            genotype_fitness = RC.computeFitnessOfGenotypes(genotypes, perf[reference][2])

        for method in methods:
            MAE_cov[method].append(RC.MAE(perf[reference][1], perf[method][1]))
            Spearmanr_cov[method].append(RC.Spearmanr(perf[reference][1], perf[method][1]))
            Pearsonr_cov[method].append(RC.Pearsonr(perf[reference][1], perf[method][1]))
            MAE_selection[method].append(RC.MAE(perf[reference][2], perf[method][2]))
            Spearmanr_selection[method].append(RC.Spearmanr(perf[reference][2], perf[method][2]))
            Pearsonr_selection[method].append(RC.Pearsonr(perf[reference][2], perf[method][2]))

            if computePerfOnGenotypeFitness:
                fitness = RC.computeFitnessOfGenotypes(genotypes, perf[method][2])
                MAE_fitness[method].append(RC.MAE(genotype_fitness, fitness))
                Spearmanr_fitness[method].append(RC.Spearmanr(genotype_fitness, fitness))
                Pearsonr_fitness[method].append(RC.Pearsonr(genotype_fitness, fitness))

    if print_process:
        print()
    if computePerfOnGenotypeFitness:
        return MAE_cov, Spearmanr_cov, Pearsonr_cov, MAE_selection, Spearmanr_selection, Pearsonr_selection, MAE_fitness, Spearmanr_fitness, Pearsonr_fitness
    else:
        return MAE_cov, Spearmanr_cov, Pearsonr_cov, MAE_selection, Spearmanr_selection, Pearsonr_selection



############################################
#
# Plotting
#
############################################


def plot_num_genotypes_vs_num_alleles_for_ps(ps, xs, xlabel='recombination_rate', figsize=(5, 8), plot_figure=True, plot_show=True, log_scale_for_x=True, alpha=0.8):

    mean_num_alleles = []
    mean_num_genotypes = []
    mean_ratios = []
    mean_num_recombinations = []
    for i, p in enumerate(ps):
        num_genotypes, num_alleles = get_num_genotypes_and_num_alleles(p)
        num_recombination = get_num_recombinations(p)
        mean_num_recombinations.append(np.mean(num_recombination))
        mean_ratios.append(np.mean(num_genotypes) / np.mean(num_alleles))
        mean_num_alleles.append(np.mean(num_alleles))
        mean_num_genotypes.append(np.mean(num_genotypes))
        if xlabel == 'recombination_rate':
            print(f'{xlabel}={xs[i]}, #genotypes={num_genotypes[:10]}...')

    if plot_figure:
        # plt.figure(figsize=figsize)
        fig, axes = plt.subplots(4, 1, figsize=figsize)

    yss = [mean_num_alleles, mean_num_genotypes, mean_ratios, mean_num_recombinations]
    ylabels = ['mean #alleles', 'mean #genotypes', 'mean #genotypes/#alleles', 'mean #recombinations']
    for i, (ys, ylabel) in enumerate(zip(yss, ylabels)):
        plt.sca(axes[i])
        plt.scatter(xs, ys, alpha=alpha)
        if log_scale_for_x:
            plt.xscale('log')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

    if plot_show:
        plt.show()


def plot_performance_for_ps(MAE_selection_dic, Spearmanr_selection_dic, MAE_fitness_dic, Spearmanr_fitness_dic, xs, xlabel='recombination_rate', methods=['true_cov', 'recovered', 'SL'], figsize=(5, 8), plot_figure=True, plot_show=True, log_scale_for_x=True, Pearsonr_for_y=False, alpha=0.8):

    metrics = [MAE_selection_dic, Spearmanr_selection_dic, MAE_fitness_dic, Spearmanr_fitness_dic]
    if Pearsonr_for_y:
        ylabels = ['MAE of\nselection', 'Pearsonr of\nselection', 'MAE of\nfitness', 'Pearsonr of\nfitness']
    else:
        ylabels = ['MAE of\nselection', 'Spearmanr of\nselection', 'MAE of\nfitness', 'Spearmanr of\nfitness']

    if plot_figure:
        fig, axes = plt.subplots(len(metrics), 1, figsize=figsize)

    for i, (metric, ylabel) in enumerate(zip(metrics, ylabels)):
        plt.sca(axes[i])
        at_bottom = i == len(metrics) - 1
        for method in methods:
            plt.scatter(xs, metric[method], alpha=alpha, label=method)
        for i, x in enumerate(xs):
            y = (metric['true_cov'][i] +  metric['SL'][i]) / 2
            text = '%.2f' % (metric['true_cov'][i] - metric['SL'][i])
            plt.text(x, y, text)
        if i == 0:
            plt.legend()
        if log_scale_for_x:
            plt.xscale('log')
        if at_bottom:
            plt.xlabel(xlabel)
            plt.xticks(ticks=xs, labels=xs, rotation=60)
        else:
            plt.xticks(ticks=[], labels=[])
        plt.xlim(xs[0] * 0.9, xs[-1] * 1.1)
        plt.ylabel(ylabel)

    if plot_show:
        plt.show()


def plot_num_genotypes_vs_num_alleles(p, alpha=0.5):

    num_genotypes, num_alleles = get_num_genotypes_and_num_alleles(p)

    compare_performance(num_alleles, num_genotypes, xlabel='num_alleles', ylabel='num_genotypes', alpha=alpha)


def compare_performance(perf1, perf2, xlabel, ylabel, figsize=(5, 4), plot_figure=True, plot_show=True, alpha=0.8):
    if plot_figure:
        plt.figure(figsize=figsize)
    plt.scatter(perf1, perf2, alpha=alpha)
    min_v, max_v = min(np.min(perf1), np.min(perf2)), max(np.max(perf1), np.max(perf2))
    plt.plot([min_v, max_v], [min_v, max_v], linestyle='dashed', color='grey', alpha=0.5)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if plot_show:
        plt.show()


def plot_selection_distribution_of_simulated_data(p):

    start_n, num_trials, N, T, mu, meanS, stdS, minS, maxS, threshold, uniform, recombination, recombination_rate, cooccurence, max_cooccuring_mutations, controlled_genotype_fitness, genotype_fitness_increase_rate, covariance, covAtEachTime, saveCompleteResults, verbose = p.parse_all_parameters()

    if uniform:
        print(f'Uniform distribution over [{minS}, {maxS}]')
    else:
        print(f'Gaussian distribution centered at meanS={meanS}, std={stdS}')
    all_selections = []
    for n in range(num_trials):
        all_selections += list(load_simulation(p, n)['selections'])
    plt.hist(all_selections)
    plt.show()


############################################
#
# Miscellaneous
#
############################################


def degenerate_a_mutation(simulation, l, alpha=0.6):

    traj, cov, sVec, nVec = simulation['traj'], simulation['cov'], simulation['sVec'], simulation['nVec']
    selections = simulation['selections']
    T, L = traj.shape
    N = np.sum(nVec[0])
    # print(traj.shape)
    traj = np.concatenate((traj[:, 0:l+1], traj[:, l:l+1], traj[:, l+1:]), axis=1)
    # print(traj.shape)
    # AP.plotTraj(traj[:, l:l+2], alpha=0.6)
    cov_new = np.zeros((L + 1, L + 1))
    for i in range(0, l + 1):
        for j in range(0, l + 1):
            cov_new[i, j] = cov[i, j]
        cov_new[i, l + 1] = cov[i, l]
        cov_new[l + 1, i] = cov[l, i]
        for j in range(l + 2, L + 1):
            cov_new[i, j] = cov[i, j - 1]

    cov_new[l + 1, l + 1] = cov[l, l]

    for i in range(l + 2, L + 1):
        for j in range(0, l + 1):
            cov_new[i, j] = cov[i - 1, j]
        cov_new[i, l + 1] = cov[i - 1, l]
        cov_new[l + 1, i] = cov[l, i - 1]
        for j in range(l + 2, L + 1):
            cov_new[i, j] = cov[i - 1, j - 1]

    sVec_new = [[] for t in range(len(sVec))]
    for t, seqs in enumerate(sVec):
        for seq in seqs:
            seq_new = np.concatenate((seq[0:l+1], seq[l:l+1], seq[l+1:]))
            sVec_new[t].append(seq_new)

    selections_new = np.concatenate((selections[0:l+1], selections[l:l+1], selections[l+1:]))
    selections_new[l] /= 2
    selections_new[l + 1] /= 2

    return dict(traj=traj, cov=cov_new, sVec=sVec_new, nVec=nVec, selections=selections_new)


def evaluate_perf_as_we_degenerate_a_simulation(simulation, num_degeneration, methods=['SL', 'true_cov']):

    traj, cov, sVec, nVec = simulation['traj'], simulation['cov'], simulation['sVec'], simulation['nVec']
    selections = simulation['selections']
    T, L = traj.shape
    N = np.sum(nVec[0])

    MAE_fitness_list = {m: [] for m in methods}
    Spearmanr_fitness_list = {m: [] for m in methods}
    Pearsonr_fitness_list = {m: [] for m in methods}

    last_ls = {l: l for l in range(L)}

    deg_sim = simulation
    num_muts = L
    for _ in range(num_degeneration):
        l = np.random.randint(L)
        l_deg = last_ls[l]
        for after_l in range(l, L):
            last_ls[after_l] = last_ls[after_l] + 1
        # print(len(deg_sim['traj'][0]), l_deg, last_ls[L - 1])
        deg_sim = degenerate_a_mutation(deg_sim, l_deg)
        MAE_cov, Spearmanr_cov, Pearsonr_cov, MAE_selection, Spearmanr_selection, Pearsonr_selection, MAE_fitness, Spearmanr_fitness, Pearsonr_fitness = parse_performance_for_a_simulation(deg_sim, reg=1, methods=methods, muForInference=0, reference='true', include_Lolipop=False, computePerfOnGenotypeFitness=True)
        for m in methods:
            MAE_fitness_list[m].append(MAE_fitness[m][0])
            Spearmanr_fitness_list[m].append(Spearmanr_fitness[m][0])
            Pearsonr_fitness_list[m].append(Pearsonr_fitness[m][0])
        num_muts += 1

    return MAE_fitness_list, Spearmanr_fitness_list, Pearsonr_fitness_list
