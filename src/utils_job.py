import os
import pickle
import argparse
import time
import numpy as np
from matplotlib import pyplot as plt

import utils_common as utils
from indep_bnet import IndepBnet
from cl_tree import CLTree
from cutset_net import CutsetNet


def run_experiment(datadir, dataset, model, res_outputdir, depth, approx_method, fraction):
    current_time = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())
    experiment_name = dataset  + '_' + model + '_a' + str(approx_method) + '_f' + str(fraction)
    resdict_file = os.path.join(res_outputdir, experiment_name + "_res.pkl" )
    res_txt_file = os.path.join(res_outputdir, dataset + "_text_results.txt" )
    experiment_name += '-T' + current_time   # Add current time

    # Populate the results dict
    results_dict = {}
    results_dict['exp_name'] = experiment_name
    results_dict['data'] = dataset
    results_dict['model'] = model
    results_dict['clt_approx'] = approx_method

    print("Running:", experiment_name)
    trn_file = os.path.join(datadir, dataset + '.ts.data') # 'small-10-datasets/accidents.ts.data'
    val_file = os.path.join(datadir, dataset + '.valid.data') #'small-10-datasets/accidents.valid.data'
    tst_file = os.path.join(datadir, dataset + '.test.data') #'small-10-datasets/accidents.test.data'
    trn_data = utils.load_data(trn_file)
    val_data = utils.load_data(val_file)
    tst_data = utils.load_data(tst_file)
    print("Datasets loaded!")

    hyp_flag = 1 if (model=="cnet" and approx_method!=0 and fraction>1) else 0

    if hyp_flag == 0:
        best_fraction = fraction
    else:
        frac_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        best_fraction = utils.hyptune_cnet(trn_data, val_data, depth, frac_list, 
                                            approx_method)
    results_dict['cnet_fraction'] = best_fraction 

    runtimes, trn_ll, val_ll, tst_ll = utils.average_out_results(model, trn_data, 
                                        val_data, tst_data, depth, best_fraction, 
                                        approx_method, runs=10)

    results_dict['train_time'] = runtimes
    results_dict['train_ll'] = trn_ll
    results_dict['valid_ll'] = val_ll
    results_dict['test_ll'] = tst_ll

    with open(resdict_file, 'wb') as outfile:
        pickle.dump(results_dict, outfile)

    print("Training runtime:", np.mean(runtimes))
    print("Train LL:", np.mean(trn_ll))
    print("Val LL:", np.mean(val_ll))
    print("Test LL:", np.mean(tst_ll))

    with open(res_txt_file, 'a') as ofile:
        print("Experiment:", experiment_name, file=ofile)
        print("Training runtime:", np.mean(runtimes), file=ofile)
        print("Train LL:", np.mean(trn_ll), file=ofile)
        print("Val LL:", np.mean(val_ll), file=ofile)
        print("Test LL:", np.mean(tst_ll), file=ofile)
        print("------------------------------------------------------", file=ofile)



def getall_kvals_clt(trn_data, val_data, kval_list, clt_approx, cnet_depth=10):
    total_hyparams = len(kval_list)
    val_scores = np.zeros(total_hyparams)
    run_times = np.zeros(total_hyparams)
    for i in range(total_hyparams):
        curr_K = kval_list[i]
        print("K:", curr_K)
        res_tup = utils.average_out_results("cltree", trn_data, val_data, val_data,
                            cnet_depth, 1.0, clt_approx, curr_K, runs=5, hyp=True)
        run_times[i] = np.mean(res_tup[0])
        val_scores[i] = np.mean(res_tup[2])
    return val_scores, run_times


def getall_fracs_cnet(trn_data, val_data, frac_list, clt_approx, cnet_depth=10):
    total_hyparams = len(frac_list)
    val_scores = np.zeros(total_hyparams)
    run_times = np.zeros(total_hyparams)
    for i in range(total_hyparams):
        curr_frac = frac_list[i]
        print("Frac:", curr_frac)
        res_tup = utils.average_out_results("cnet", trn_data, val_data, val_data,
                            cnet_depth, curr_frac, clt_approx, runs=5, hyp=True)
        run_times[i] = np.mean(res_tup[0])
        val_scores[i] = np.mean(res_tup[2])
    return val_scores, run_times


def plot_val_runt(values, results, my_title, xlab, ylab, img_name):
    plt.figure()
    plt.plot(values, results, 'g-')    
    plt.title(my_title)    
    plt.xlabel(xlab)
    plt.ylabel(ylab)    
    plt.savefig(img_name, dpi=300)

