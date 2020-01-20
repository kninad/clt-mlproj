import os
import pickle
from utils_job import run_experiment, getall_fracs_cnet

data_list = ['nltcs', 'msnbc', 'kdd', 'plants', 'baudio', 'jester', 'bnetflix', 'accidents', 'r52', 'dna']
datadir = "../data/"
res_outputdir = "../results/"
depth = 10

model = "cnet"
approx_method = 2
fraction = 100

for dataset in data_list:
    run_experiment(datadir, dataset, model, res_outputdir, depth, approx_method, fraction)















"""
Scripts:
python main.py -d dset -m indep  -a 0 -f 1
python main.py -d dset -m cltree -a 0 -f 1
python main.py -d dset -m cltree -a 1 -f 1

python main.py -d dset -m cltree -a 2 -f 1

python main.py -d dset -m cnet -a 0 -f 1
python main.py -d dset -m cnet -a 1 -f 100

python main.py -d dset -m cnet -a 2 -f 100

"""
