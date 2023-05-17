#!/usr/bin/env python
# Infer genotypes and genotype trajectories with Evoracle.

import sys
import argparse
import numpy as np                          # numerical tools
from timeit import default_timer as timer   # timer for performance
import math
import MPL                                  # MPL inference tools
import data_parser as DP
import simulation_helper as SH
sys.path.append('../../evoracle/')  # path to Evoracle
import evoracle


DATA_DIR_REL = '../data'
SRC_DIR_REL = f'{DATA_DIR_REL}/src'
SIMULATION_DIR_REL = f'{DATA_DIR_REL}/simulation'
SELECTION_DIR_REL = f'{DATA_DIR_REL}/selection'
EVORACLE_DIR_REL = f'{DATA_DIR_REL}/evoracle'
EVORACLE_DIR_SIMULATION_REL = f'{EVORACLE_DIR_REL}/simulation'
EVORACLE_DIR_SIMULATION_PARSED_OUTPUT_REL = f'{EVORACLE_DIR_REL}/simulation_parsed_output'


def main(verbose=False):
    """Infer genotypes and genotype trajectories with Evoracle.
    """

    parser = argparse.ArgumentParser(description='Infer genotypes and genotype trajectories with Evoracle.')
    parser.add_argument('-o', type=str, default=None, help="prefix of output .npz file (filename is arg_list.o + f'_truncate=*_window=*.npz')")
    parser.add_argument('-n', type=int, default=0, help='index of trial')
    parser.add_argument('--save_geno_traj', action='store_true', default=False, help='whether or not save inferred genotype trajectories.')

    arg_list = parser.parse_args(sys.argv[1:])

    n = arg_list.n

    # Load simulation
    p = SH.Params()
    sim = SH.load_simulation(p, n, directory=SIMULATION_DIR_REL)
    traj = sim['traj']

    # Save traj as input for Evoracle
    directory = f'{EVORACLE_DIR_SIMULATION_REL}/n={n}'
    obsreads_filename = f'simulation_n={n}_obsreads.csv'
    obsreads_input = f'{directory}/{obsreads_filename}'
    DP.save_traj_for_evoracle(traj, obsreads_input)

    # Run Evoracle
    proposed_genotypes_output = f'{directory}/proposed_genotypes.csv'
    obs_reads_df = DP.parse_obs_reads_df(obsreads_input)
    evoracle.propose_gts_and_infer_fitness_and_frequencies(obs_reads_df, proposed_genotypes_output, directory)
    results = DP.parse_evoracle_results(obsreads_filename, directory, save_geno_traj=arg_list.save_geno_traj)

    if arg_list.o is not None:
        output = arg_list.o
    else:
        output = f'evoracle_parsed_output_n={n}.npz'

    np.savez_compressed(f'{EVORACLE_DIR_SIMULATION_PARSED_OUTPUT_REL}/{output}', **results)


if __name__ == '__main__': main()
