import os
import pickle
import argparse
import time
import numpy as np

import utils_common as utils
import utils_job
from indep_bnet import IndepBnet
from cl_tree import CLTree
from cutset_net import CutsetNet


parser = argparse.ArgumentParser(description='Fit a desired model and output the test data metrics')

parser.add_argument("-p", "--dirpath", required=False,  type=str, default='../data/',
                    help="Data directory path")

parser.add_argument("-d", "--data", required=True,  type=str, 
        help="Dataset name from: nltcs, msnbc, kdd, plants, baudio, jester, bnetflix, accidents, r52, dna")

parser.add_argument("-m", "--model", required=True,  type=str, 
        help="Model type: \"indep\", \"cltree\", \"cnet\".")

parser.add_argument("-o", "--output", required=False,  type=str,  default='../results/',
        help="Results output directory")

parser.add_argument("--depth", required=False,  type=int, default=10,
        help="Depth of OR tree. Default=10")

parser.add_argument("-a", "--approx", required=False,  type=int, default=0,
        help="Approx Method for CLtree. Default=0 (No Approx)")

parser.add_argument("-f", "--frac", required=False,  type=float, default=1,
        help="Fraction for data sampling in OR tree.  Default=1 (No HYP tuning). If greater than 1 then HYP")

args = parser.parse_args()
datadir = args.dirpath
dataset = args.data
model = args.model
res_outputdir = args.output
depth = args.depth
approx_method = args.approx
fraction = args.frac

def main():
    utils_job.run_experiment(datadir, dataset, model, res_outputdir, depth, approx_method, fraction)

if __name__ == "__main__":
    main()

"""
Scripts:
python main.py -d dset -m indep
python main.py -d dset -m cltree -a 0
python main.py -d dset -m cltree -a 1
python main.py -d dset -m cltree -a 2
python main.py -d dset -m cnet -a 0 -f 1
python main.py -d dset -m cnet -a 1 -f 100
python main.py -d dset -m cnet -a 2 -f 100
"""
