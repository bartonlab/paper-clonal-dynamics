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
EVORACLE_DIR_REL = f'{DATA_DIR_REL}/evoracle'


def main(verbose=False):
    """Infer genotypes and genotype trajectories with Evoracle.
    """

    parser = argparse.ArgumentParser(description='Infer genotypes and genotype trajectories with Evoracle.')
    parser.add_argument('-o', type=str, default=None, help="prefix of output .npz file (filename is arg_list.o + f'_truncate=*_window=*.npz')")
    parser.add_argument('-d', type=str, default=None, help="directory")
    parser.add_argument('-i', type=str, default=None, help="obsreads_filename")
    parser.add_argument('--save_geno_traj', action='store_true', default=False, help='whether or not save inferred genotype trajectories.')

    arg_list = parser.parse_args(sys.argv[1:])
    obsreads_filename = arg_list.i
    directory = arg_list.d

    # Run Evoracle
    proposed_genotypes_output = f'{directory}/proposed_genotypes.csv'
    obs_reads_df = DP.parse_obs_reads_df(f'{directory}/{obsreads_filename}')
    evoracle.propose_gts_and_infer_fitness_and_frequencies(obs_reads_df, proposed_genotypes_output, directory)
    results = DP.parse_evoracle_results(obsreads_filename, directory, save_geno_traj=arg_list.save_geno_traj)

    np.savez_compressed(arg_list.o, **results)


if __name__ == '__main__': main()
