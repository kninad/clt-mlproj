import os
import pickle
import argparse
import time
import numpy as np

import utils_common as utils
from utils_job import getall_kvals_clt, getall_fracs_cnet, plot_val_runt
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

print("Running:", dataset, model)
trn_file = os.path.join(datadir, dataset + '.ts.data') # 'small-10-datasets/accidents.ts.data'
val_file = os.path.join(datadir, dataset + '.valid.data') #'small-10-datasets/accidents.valid.data'
tst_file = os.path.join(datadir, dataset + '.test.data') #'small-10-datasets/accidents.test.data'
trn_data = utils.load_data(trn_file)
val_data = utils.load_data(val_file)
tst_data = utils.load_data(tst_file)
print("Datasets loaded!")

n = trn_data.shape[1]
frac_list = np.linspace(0.1, 1.0, num=9, endpoint=False)
kvals = np.linspace(2, n, num=10, endpoint=False, dtype=int)

if model == "cltree":
    val_scores, runtimes = getall_kvals_clt(trn_data, val_data, kvals, approx_method)
    title = "K vs Perf: " + model + " on " + dataset
    xlab = "K"
    img_base_name = "./Kval_" + model + '_' + dataset + 'a_' + str(approx_method)
    plot_val_runt(kvals, val_scores, title, xlab, "ValLL", img_base_name + "_ValLL.jpg")
    plot_val_runt(kvals, runtimes, title, xlab, "RunTime", img_base_name + "_RunTime.jpg")
elif model == "cnet":
    val_scores, runtimes = getall_fracs_cnet(trn_data, val_data, frac_list, approx_method)
    title = "Data Fraction vs Perf: " + model + " on " + dataset
    xlab = "Fraction f"
    img_base_name = "./frac_" + model + '_' + dataset + 'a_' + str(approx_method)
    plot_val_runt(frac_list, val_scores, title, xlab, "ValLL", img_base_name + "_ValLL.jpg")
    plot_val_runt(frac_list, runtimes, title, xlab, "RunTime", img_base_name + "_RunTime.jpg")

