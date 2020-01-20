import numpy as np
import utils_common as utils
from cl_tree import CLTree


class CnetTreeNode(object):

    def __init__(self, var):
        self.var = var    # feat-column or leaf value
        self.ids_var = None
        self.prob0 = None
        self.prob1 = None
        self.child0 = None
        self.child1 = None

    def is_leaf(self):
        if self.child0 or self.child1:
            return False
        else:
            return True

    def print_node(self):
        if not self.is_leaf():
            print("var: ", self.var)
            # print("depth: ", self.depth)        
        else:
            print("Leaf node!")


class CutsetNet(object):
    """
    Class to implement cutset networks (without pruning)
    """

    def __init__(self,depth=10, min_rec=100, min_var=10):
        self.nvariables=0
        self.depth=depth
        self.tree=[]        
        # 2 thresholds to stop going deeper
        self.min_rec = min_rec   
        self.min_var = min_var
        # for get node and edge potential
        self.internal_list = []
        self.internal_var_list = []
        self.leaf_list = []
        self.leaf_ids_list = []
    
    def compute_best_attr(self, dataset):
        pairs = utils.compute_pairwise_counts(dataset) + 1  # Laplace correction
        pairs = utils.normalize2D(pairs)
        singles = utils.compute_single_counts(dataset) + 2  # laplace correction
        singles = utils.normalize1D(singles)
        edgemat = utils.compute_adjmatrix(pairs, singles)
        np.fill_diagonal(edgemat, 0) #  #           
        scores = np.sum(edgemat, axis=0)
        variable = np.argmax(scores)    # this variable (number) will not correspond exactly to our global list of feature column numbers
        return variable

    def build_tree(self, dataset, ids, samp_f, clt_approx, clt_samp):
        curr_depth = self.nvariables - dataset.shape[1]

        # Termination condition satisfied, learn the CLTree as a leaf.
        if dataset.shape[0]<self.min_rec or dataset.shape[1]<self.min_var or curr_depth >= self.depth:
            clt = CLTree()
            clt.train(dataset, clt_approx, clt_samp)
            leaf_node = CnetTreeNode(clt)    # for leaf node, the var = clt tree object, not a feature col num.
            return leaf_node
        num_trn = dataset.shape[0]
        condition_sampf = (np.abs(1.0 - samp_f) <= 1e-2)
        if condition_sampf or num_trn<=100: # Include fractions higher than 0.99
            # Use the full dataset
            dataset_s = dataset
        else:            
            last_index = int(np.ceil(samp_f * num_trn)) - 1  # safety check for last index
            dataset_s = dataset[:last_index]

        # this variable (number) will not correspond exactly to our global list of feature column numbers
        variable = self.compute_best_attr(dataset_s)

        root_node = CnetTreeNode(variable)  # Make a new node corresponding to the data, ids.
        root_node.ids_var = ids[variable]   # what's the "actual" variable corresponding to the ids list.
        
        rows_1 = (dataset_s[:, variable] == 1)    # Boolean mask array, maybe used later
        rows_0 = (dataset_s[:, variable] == 0)                
        new_ids_list = np.delete(ids, variable, axis=0)  # don't need the variable now. 
        new_ids_mask = [True if i  != variable else False for i in range(len(ids))]
        tmp_dataset_1 = dataset_s[rows_1][:, new_ids_mask]       
        tmp_dataset_0 = dataset_s[rows_0][:, new_ids_mask]
        p1 = float(tmp_dataset_1.shape[0]) + 1.0                        
        p0 = float(tmp_dataset_0.shape[0]) + 1.0                
        p0 = p0 / (p0+p1)   # Normalize
        p1 = 1.0 - p0         
        
        root_node.prob0 = p0 # Set probs in root_node
        root_node.prob1 = p1                
        root_node.child0 = self.build_tree(tmp_dataset_0, new_ids_list, samp_f, clt_approx, clt_samp)
        root_node.child1 = self.build_tree(tmp_dataset_1, new_ids_list, samp_f, clt_approx, clt_samp)       
        return root_node

    def train(self, dataset, sampling_fraction=1.0, clt_approx=0, samp_k=None):
        """
            Class method to train the model using the recursice `build_tree` method.
        """
        np.random.shuffle(dataset)
        self.nvariables = dataset.shape[1]
        ids = np.arange(self.nvariables)        
        self.tree = self.build_tree(dataset, ids, sampling_fraction, clt_approx, samp_k)

    def log_prob(self, datavec):
        logprob = 0.0
        curr_node = self.tree
        ids = np.arange(self.nvariables)
        while (not curr_node.is_leaf()):
            feat_pos = curr_node.var
            feat_col = curr_node.ids_var    # actual index in the global all ids list            
            p0 = curr_node.prob0
            p1 = curr_node.prob1
            if datavec[feat_col] == 0: # go to subtree for 0
                logprob += np.log(p0)
                curr_node = curr_node.child0                
            elif datavec[feat_col] == 1: # go to subtree for 1
                logprob += np.log(p1)
                curr_node = curr_node.child1            
            ids = np.delete(ids, feat_pos, axis=0)  # remove the feat_col which is at feat_pos
            # if curr_node.is_leaf(): # Catch the leaf node = CLTree here!
            #     # CLTree object also has a log_prob method 
            #     # Just pass the "reduced" datavec using ids list
            #     cltree = curr_node.var
            #     logprob += cltree.log_prob(datavec[ids])
        cltree = curr_node.var
        logprob += cltree.log_prob(datavec[ids])
        return logprob


