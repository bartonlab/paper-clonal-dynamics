#!/usr/bin/env python

import sys
import argparse
import math
import numpy as np                          # numerical tools
from timeit import default_timer as timer   # timer for performance
import reconstruct_clades as RC

def main(verbose=False):
    """ Initial clusterization of clade reconstruction algorithm. """

    # Read in simulation parameters from command line

    parser = argparse.ArgumentParser(description="Initial clusterization of clade reconstruction algorithm.")
    parser.add_argument('-i', type=str, default=None, help='input file containing traj')
    parser.add_argument('-o', type=str, default=None, help='output destination')
    parser.add_argument('--weightByBothVariance', action='store_true', default=False, help='weight option')
    parser.add_argument('--weightBySmallerVariance', action='store_true', default=True, help='weight option')

    arg_list = parser.parse_args(sys.argv[1:])

    out_str = arg_list.o
    traj = np.load(arg_list.i)
    weightByBothVariance = arg_list.weightByBothVariance
    weightBySmallerVariance = arg_list.weightBySmallerVariance

    reconstruction = RC.CladeReconstruction(traj)
    reconstruction.setParamsForClusterization(weightByBothVariance=weightByBothVariance, weightBySmallerVariance=weightBySmallerVariance)
    reconstruction.clusterMutations()
    groups, segmentedIntDxdx = reconstruction.groups, reconstruction.segmentedIntDxdx

    print("Saving results...")
    start = timer()
    f = open(out_str + '.npz', 'wb')
    np.savez(f, groups=groups, segmentedIntDxdx=segmentedIntDxdx)
    f.close()
    end = timer()
    print(f'Takes %lfs' % (end - start))

if __name__ == '__main__': main()
