from collections import defaultdict
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree, depth_first_order
import numpy as np
import utils_common as utils


class CLTree(object):
    def __init__(self):                
        self.prob_pair = None
        self.prob_sing = None
        self.node_order = None
        self.parent = None        

    def compute_exact_graph(self, data):
        pairwise_counts = utils.compute_pairwise_counts(data) + 1
        self.prob_pair = utils.normalize2D(pairwise_counts)        
        single_counts = utils.compute_single_counts(data) + 2
        self.prob_sing = utils.normalize1D(single_counts)
        adjmat = utils.compute_adjmatrix(self.prob_pair, self.prob_sing)
        adjmat[adjmat == 0.0] = 1e-10   # complete graph!
        return adjmat

    def compute_exact_mst(self, adjmat):
        return minimum_spanning_tree(csr_matrix(adjmat))

    def compute_approx_graph(self, data, samp_k):
        nvars = data.shape[1]
        adjmat = np.zeros((nvars, nvars))        
        single_counts = utils.compute_single_counts(data) + 2
        self.prob_sing = utils.normalize1D(single_counts)        
        pairprob_arr = np.zeros((nvars, nvars, 2, 2))

        nodes = [i for i in range(nvars)]        
        curr_node = utils.select_nodes(nodes)        
        nodes_in_tree = [curr_node]
        nodes.remove(curr_node)
        steps = 0
        flag = 0  # all nodes visited?
        while nodes or steps < nvars-1:
        # while nodes:
            if flag == 1:
                # If all nodes exhausted, just select from all nodes except the current!                
                tmp_nodes = [i for i in range(nvars) if i != curr_node]
                candidates =  utils.select_nodes(tmp_nodes, num=samp_k)            
            elif len(nodes) <= samp_k:    # Stop sampling!
                candidates = nodes  
            else:            
                candidates = utils.select_nodes(nodes, num=samp_k)

            for can in candidates:
                score = utils.compute_mutualinfo(data, curr_node, can)
                adjmat[curr_node, can] = score
                adjmat[can, curr_node] = score
                # Set the CPTs corresponding to this pair (curr_node, can)
                cpt_xy, cpt_yx = utils.compute_cpt_xy(data, curr_node, can)
                pairprob_arr[curr_node, can, :, :] = cpt_xy
                pairprob_arr[can, curr_node, :, :] = cpt_yx

            # update remaining nodes list
            if flag == 0:   # not visited all the nodes
                nodes_in_tree.extend(candidates)
                nodes = [ele for ele in nodes if ele not in candidates]
            if not nodes:
                flag = 1                
            steps += 1
            # print(curr_node, nodes, candidates, nodes_in_tree, steps)
            # Reset the curr_node
            curr_node = utils.select_nodes(nodes_in_tree)

        self.prob_pair = pairprob_arr   # For inference!!
        adjmat += np.identity(nvars)
        return adjmat

    def give_best_candidate(self, data, root, candidates):
        best = candidates[0]
        best_score = -1
        for can in candidates:
            score = utils.compute_mutualinfo(data, root, can)
            # print(root, can, score)
            if score > best_score:
                best = can
                best_score = score
        return best
        
    def compute_approx_spantree(self, data, samp_k):        
        nvars = data.shape[1]
        single_counts = utils.compute_single_counts(data) + 2
        self.prob_sing = utils.normalize1D(single_counts)
        pairprob_arr = np.zeros((nvars, nvars, 2, 2))
        parent = np.zeros(nvars, dtype=int)        
        nodes = [i for i in range(nvars)]
        edge_count = 0
        nodes_in_tree = []  # Already selected nodes for the tree

        curr_node = utils.select_nodes(nodes)
        nodes_in_tree.append(curr_node)
        nodes.remove(curr_node)
        parent[curr_node] = -9999   # Following the convention! Its a ROOT!

        while edge_count < nvars - 1:
            if len(nodes) <= samp_k:    # Stop sampling!
                candidates = nodes  
            else:            
                candidates = utils.select_nodes(nodes, num=samp_k)

            best_node = self.give_best_candidate(data, curr_node, candidates)                        
            nodes_in_tree.append(best_node)
            nodes.remove(best_node) # remove from remaining list of nodes
            parent[best_node] = curr_node
            edge_count += 1            
            # print(curr_node, best_node, candidates, nodes, nodes_in_tree, edge_count)

            # Set the CPTs corresponding to this pair (curr_node, best_node)
            cpt_xy, cpt_yx = utils.compute_cpt_xy(data, curr_node, best_node)
            pairprob_arr[curr_node, best_node, :, :] = cpt_xy
            pairprob_arr[best_node, curr_node, :, :] = cpt_yx
            
            # Reset the curr_node        
            curr_node = utils.select_nodes(nodes_in_tree)

        self.prob_pair = pairprob_arr
        return np.arange(nvars), parent


    def train(self, data, approx=0, samp_k=None):
        """
        approx (default = 0) means no approx.
        approx = 1 means use the AST method described in todo
        approx = 2 means use the AGH method described in todo
        For either of the above, use samp_k number of nodes to sample.
        samp_k should be about log N
        """
        np.random.shuffle(data)
        num_feats = data.shape[1]
        if not samp_k:
            samp_k = int(np.ceil(np.log2(num_feats)))
        # print(samp_k)
        if approx == 1: # Use the AST method (approx_spantree)
            self.node_order, self.parent = self.compute_approx_spantree(data, samp_k)
        else:
            if approx == 0:
                adjmat = self.compute_exact_graph(data)
            elif approx == 2:   # Use the AGH method (approx_graph)
                adjmat = self.compute_approx_graph(data, samp_k)
            adjmat *= -1.0 # making negative for MST calc
            mstree = self.compute_exact_mst(adjmat)
            self.node_order, self.parent = depth_first_order(mstree, i_start=0, directed=False)


    def train_weighted(self, weights, data):
        # weights is a np vector assigning weights to every data-vector in data
        N = data.shape[0]
        alpha = max(np.sum(weights), 1)
        alpha /= N        
        pairwise_counts = utils.compute_pairwise_counts_weighted(data, weights) + alpha
        self.prob_pair = utils.normalize2D(pairwise_counts)
        single_counts = utils.compute_single_counts_weighted(data, weights) + 2 * alpha
        self.prob_sing = utils.normalize1D(single_counts)

        adjmat = utils.compute_adjmatrix(self.prob_pair, self.prob_sing) 
        adjmat *= -1.0 # making negative for MST calc
        adjmat[adjmat == 0.0] = 1e-10       
        mstree = minimum_spanning_tree(csr_matrix(adjmat))
        self.node_order, self.parent = depth_first_order(mstree, 0, directed=False)
        
    def is_root(self, var):
        if self.parent[var] == -9999:
            return True
        else:
            return False

    def prob(self, datavec):
        prob = 1.0
        for var_i in self.node_order:
            val_i = datavec[var_i]
            if self.is_root(var_i):
                # ROOT!
                prob *= self.prob_sing[var_i][val_i]
            else:
                par = self.parent[var_i]
                val_par = datavec[par]
                num = self.prob_pair[var_i, par, val_i, val_par]
                denom = self.prob_sing[par, val_par]
                prob *= (num / denom)
        return prob

    def log_prob(self, datavec):
        lprob = 0
        for var_i in self.node_order:            
            val_i = datavec[var_i]
            if self.is_root(var_i): # ROOT!
                lprob += np.log(self.prob_sing[var_i][val_i])
            else:
                par = self.parent[var_i]
                # print(var_i, par)
                val_par = datavec[par]
                num = self.prob_pair[var_i, par, val_i, val_par]
                denom = self.prob_sing[par, val_par]
                lprob += np.log(num / denom)
        return lprob

if __name__ == "__main__":
    # Run tests
    pass
