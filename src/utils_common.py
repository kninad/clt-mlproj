import os
import pickle
import numpy as np
import time

from indep_bnet import IndepBnet
from cl_tree import CLTree
from cutset_net import CutsetNet
from old_cnet import OldCutsetNet

def load_data(filename):
    data = np.loadtxt(filename, dtype=int, delimiter=',')
    return data

def compute_LL(clf, dataset):
    logll = 0
    for vec in dataset:
        logll += clf.log_prob(vec)
    N = dataset.shape[0]
    avg_logll = logll / N
    # print(logll, avg_logll)
    return avg_logll

def select_nodes(nodes, num=None):
    return np.random.choice(nodes, size=num, replace=False)


def hyptune_cnet(trn_data, val_data, cnet_depth, frac_list, clt_approx):
    total_hyparams = len(frac_list)
    val_scores = np.zeros(total_hyparams)
    for i in range(total_hyparams):
        curr_frac = frac_list[i]
        print("Frac:", curr_frac)
        res_tup = average_out_results("cnet", trn_data, val_data, val_data,
                            cnet_depth, curr_frac, clt_approx, runs=5, hyp=True)
        val_scores[i] = np.mean(res_tup[2])
    mx_idx = np.argmax(val_scores)
    best_fraction = frac_list[mx_idx]
    return best_fraction

def average_out_results(model, trn_data, val_data, tst_data, cnet_depth, 
                        cnet_frac, clt_approx, samp_k=False, runs=10, hyp=False):
    runtimes = np.zeros(runs)
    trn_logl = np.zeros(runs)
    val_logl = np.zeros(runs)
    tst_logl = np.zeros(runs)        
    for i in range(runs):
        print("Run num:", i+1)
        if model == "indep":
                clf = IndepBnet()
                start_time = time.time()
                clf.train(trn_data)
                end_time = time.time()
                time_exp = end_time - start_time
        elif model == "cltree":
                clf  = CLTree()
                start_time = time.time()
                clf.train(trn_data, clt_approx, samp_k)
                end_time = time.time()
                time_exp = end_time - start_time                    
        elif model == "cnet":
                clf = CutsetNet(depth=cnet_depth)
                start_time = time.time()
                clf.train(trn_data, cnet_frac, clt_approx)
                end_time = time.time()
                time_exp = end_time - start_time        
        runtimes[i] = time_exp
        val_logl[i] = compute_LL(clf, val_data)
        if not hyp:
            trn_logl[i] = compute_LL(clf, trn_data)        
            tst_logl[i] = compute_LL(clf, tst_data)            
    # mean_time, std_time = np.mean(runtimes), np.std(runtimes)        
    # return (mean_time, std_time)
    return runtimes, trn_logl, val_logl, tst_logl


def compute_mutualinfo(dataset, x, y):
    N = dataset.shape[0] + 1
    r_x = dataset[:, x] == 1
    r_y = dataset[:, y] == 1
    r_xy = r_x * r_y
    N_x = np.sum(r_x) + 1 #r_x.shape[0]
    N_y = np.sum(r_y) + 1#r_y.shape[0]
    N_xy = np.sum(r_xy) + 1

    nums = np.array([N_y, N-N_y, N_xy, N_x-N_xy, N_y-N_xy, N-N_x-N_y+N_xy], dtype=float)
    nums[nums < 1] = 1.0
    mask = np.ones(shape=nums.shape)
    mask[:2] *= -1.0
    nums *= np.log(nums)
    return np.sum(nums * mask, axis=0)

def compute_cpt_xy(dataset, x, y):
    joint_pt_xy = np.zeros((2,2), dtype=float)
    joint_pt_yx = np.zeros((2,2), dtype=float)
    N = dataset.shape[0]
    r_x = dataset[:, x] == 1
    r_y = dataset[:, y] == 1
    r_xy = r_x * r_y
    N_x = np.sum(r_x)
    N_y = np.sum(r_y)
    N_xy = np.sum(r_xy)
    
    joint_pt_xy[1,1] = N_xy + 1
    joint_pt_xy[0,0] = N - N_x - N_y + N_xy + 1
    
    joint_pt_yx[1,1] = N_xy + 1
    joint_pt_yx[0,0] = N - N_x - N_y + N_xy + 1
    
    joint_pt_xy[1,0] = N_x - N_xy + 1
    joint_pt_xy[0,1] = N_y - N_xy + 1

    joint_pt_yx[0,1] = N_x - N_xy + 1
    joint_pt_yx[1,0] = N_y - N_xy + 1

    joint_pt_xy /= (N+1)
    joint_pt_yx /= (N+1)
    return joint_pt_xy, joint_pt_yx 


def compute_pairwise_counts(dataset):
        nvariables=dataset.shape[1]
        count_xy = np.zeros((nvariables, nvariables, 2, 2))
        count_xy[:, :, 0, 0] = np.einsum('ij,ik->jk', (dataset == 0).astype(int), (dataset == 0).astype(int))
        count_xy[:, :, 0, 1] = np.einsum('ij,ik->jk', (dataset == 0).astype(int), (dataset == 1).astype(int))
        count_xy[:, :, 1, 0] = np.einsum('ij,ik->jk', (dataset == 1).astype(int), (dataset == 0).astype(int))
        count_xy[:, :, 1, 1] = np.einsum('ij,ik->jk', (dataset == 1).astype(int), (dataset == 1).astype(int))
        return count_xy

def compute_single_counts(dataset):
        nvariables = dataset.shape[1]
        count_x = np.zeros((nvariables, 2))
        count_x[:,0]=np.einsum('ij->j',(dataset == 0).astype(int))
        count_x[:,1] = np.einsum('ij->j',(dataset == 1).astype(int))
        return count_x

def compute_pairwise_counts_weighted(dataset, weights):
        # weights assings a weight to every data point. Computes the weighted counts
        nvariables=dataset.shape[1]
        count_xy = np.zeros((nvariables, nvariables, 2, 2))
        count_xy[:, :, 0, 0] = np.einsum('ij,ik->jk', (dataset == 0).astype(int) * weights[:, np.newaxis], (dataset == 0).astype(int))
        count_xy[:, :, 0, 1] = np.einsum('ij,ik->jk', (dataset == 0).astype(int) * weights[:, np.newaxis], (dataset == 1).astype(int))
        count_xy[:, :, 1, 0] = np.einsum('ij,ik->jk', (dataset == 1).astype(int) * weights[:, np.newaxis], (dataset == 0).astype(int))
        count_xy[:, :, 1, 1] = np.einsum('ij,ik->jk', (dataset == 1).astype(int) * weights[:, np.newaxis], (dataset == 1).astype(int))
        return count_xy

def compute_single_counts_weighted(dataset, weights):
        # weights assings a weight to every data point. Computes the weighted counts
        nvariables = dataset.shape[1]
        count_x = np.zeros((nvariables, 2))
        count_x[:,0] = np.einsum('ij->j',(dataset == 0).astype(int) * weights[:, np.newaxis])
        count_x[:,1] = np.einsum('ij->j',(dataset == 1).astype(int) * weights[:, np.newaxis])
        return count_x

def normalize(weights):
        norm_const=np.sum(weights)
        return weights/norm_const

def normalize2D(counts):
        pcountsf = counts.astype(np.float64)
        norm_const = np.einsum('ijkl->ij', pcountsf)
        return pcountsf/norm_const[:,:,np.newaxis,np.newaxis]

def normalize1D(counts):
        scountsf = counts.astype(np.float64)
        norm_const = np.einsum('ij->i', scountsf)
        return scountsf/norm_const[:,np.newaxis]

def compute_adjmatrix(prob_pair, prob_sing):
    inv_ps = np.reciprocal(prob_sing)
    D = prob_sing.shape[0]
    adjmat = np.zeros((D, D))
    adjmat += prob_pair[:,:,0,0] * np.log(np.einsum('ij,i,j->ij', prob_pair[:,:,0,0], inv_ps[:,0], inv_ps[:,0]))
    adjmat += prob_pair[:,:,0,1] * np.log(np.einsum('ij,i,j->ij', prob_pair[:,:,0,1], inv_ps[:,0], inv_ps[:,1]))
    adjmat += prob_pair[:,:,1,0] * np.log(np.einsum('ij,i,j->ij', prob_pair[:,:,1,0], inv_ps[:,1], inv_ps[:,0]))
    adjmat += prob_pair[:,:,1,1] * np.log(np.einsum('ij,i,j->ij', prob_pair[:,:,1,1], inv_ps[:,1], inv_ps[:,1]))
    return adjmat

