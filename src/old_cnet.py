import sys
import copy
import numpy as np
import utils_common
from cl_tree import CLTree
# from Util import *
# import utilM


class CnetTreeNode(object):
    def __init__(self, var):
        self.var = var    # feat-column or leaf value
        self.ids_var = None
        self.prob0 = None
        self.prob1 = None
        self.child0 = None
        self.child1 = None
        self.rows1 = None
        self.rows0 = None
        self.depth = 1 # depth is 1 on init

    def is_leaf(self):
        if self.child0 or self.child1:
            return False
        else:
            return True

    def print_node(self):
        if not self.is_leaf():
            print("var: ", self.var)
            print("depth: ", self.depth)        
        else:
            print("Leaf node!")


class OldCutsetNet(object):
    def __init__(self,depth=100, min_rec=10, min_var=5):
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
    
    def build_tree(self, dataset, ids):
        curr_depth = self.nvariables - dataset.shape[1]
        #print ("curr_dept: ", curr_depth)
         
        # Termination condition satisfied, learn the CLTree as a leaf.
        if dataset.shape[0]<self.min_rec or dataset.shape[1]<self.min_var or curr_depth >= self.depth:
            clt = CLTree()
            clt.train(dataset)
            leaf_node = CnetTreeNode(clt)    # for leaf node, the var = clt tree object, not a feature col num.
            return leaf_node
        
        pairs = utils_common.compute_pairwise_counts(dataset) + 1  # Laplace correction
        pairs = utils_common.normalize2D(pairs)
        singles = utils_common.compute_single_counts(dataset) + 2  # laplace correction
        singles = utils_common.normalize1D(singles)
        edgemat = utils_common.compute_adjmatrix(pairs, singles)
        np.fill_diagonal(edgemat, 0) #  #           
        scores = np.sum(edgemat, axis=0)
        variable = np.argmax(scores)    # this variable (number) will not correspond exactly to our global list of feature column numbers

        root_node = CnetTreeNode(variable)  # Make a new node corresponding to the data, ids.
        root_node.ids_var = ids[variable]   # what's the "actual" variable corresponding to the ids list.
        root_node.depth = curr_depth

        rows_1 = (dataset[:, variable] == 1)    # Boolean mask array, maybe used later
        rows_0 = (dataset[:, variable] == 0)        
        root_node.rows1 = rows_1 # Set the rows in root_node, maybe used later. 
        root_node.rows0 = rows_0

        new_ids_list = np.delete(ids, variable, axis=0)  # don't need the variable now. 
        new_ids_mask = [True if i  != variable else False for i in range(len(ids))]
        tmp_dataset_1 = dataset[rows_1][:, new_ids_mask]       
        tmp_dataset_0 = dataset[rows_0][:, new_ids_mask]
        # new_dataset1 = np.delete(dataset[rows_1], variable, axis=1)
        # new_dataset0 = np.delete(dataset[rows_0], variable, axis=1)               
        p1 = float(tmp_dataset_1.shape[0]) + 1.0                        
        p0 = float(tmp_dataset_0.shape[0]) + 1.0                
        p0 = p0 / (p0+p1)   # Normalize
        p1 = 1.0 - p0         
        root_node.prob0 = p0 # Set probs in root_node
        root_node.prob1 = p1
                
        root_node.child0 = self.build_tree(tmp_dataset_0, new_ids_list)
        root_node.child1 = self.build_tree(tmp_dataset_1, new_ids_list)
        
        return root_node

        # return [variable,ids[variable],p0,p1,self.learnStructureHelper(new_dataset0,new_ids),
        #         self.learnStructureHelper(new_dataset1,new_ids)]    


    '''
        Recursively learn the structure and parameter
    '''    
    def learnStructureHelper(self, dataset, ids):
        curr_depth = self.nvariables - dataset.shape[1]
        #print ("curr_dept: ", curr_depth)
         
        # Termination condition satisfied, learn the CLTree as a leaf.
        if dataset.shape[0]<self.min_rec or dataset.shape[1]<self.min_var or curr_depth >= self.depth:
            clt = CLTree()
            clt.train(dataset)
            return clt
        
        pairs = utils_common.compute_pairwise_counts(dataset) + 1  # Laplace correction
        pairs = utils_common.normalize2D(pairs)
        singles = utils_common.compute_single_counts(dataset) + 2  # laplace correction
        singles = utils_common.normalize1D(singles)
        edgemat = utils_common.compute_adjmatrix(pairs, singles)
        np.fill_diagonal(edgemat, 0) #  #           
        scores = np.sum(edgemat, axis=0)
        variable = np.argmax(scores)    # this variable (number) will not correspond exactly to our global list of feature column numbers
        
        new_dataset1=np.delete(dataset[dataset[:,variable]==1],variable,1)
        p1=float(new_dataset1.shape[0])+1.0
        new_ids=np.delete(ids,variable,0)        
        new_dataset0 = np.delete(dataset[dataset[:, variable] == 0], variable, 1)
        p0 = float(new_dataset0.shape[0]) +1.0        
        # Normalize
        p0 = p0/(p0+p1)
        p1 = 1.0 - p0        

        return [variable,ids[variable],p0,p1,self.learnStructureHelper(new_dataset0,new_ids),
                self.learnStructureHelper(new_dataset1,new_ids)]
        
        
    def train(self, dataset):
        self.nvariables = dataset.shape[1]
        ids = np.arange(self.nvariables)
        self.tree = self.learnStructureHelper(dataset, ids)
        # self.tree = self.build_tree(dataset, ids)
        

    """
        Compute the log probability for a single data vector
    """
    def log_prob(self, datavec):
        logprob = 0.0
        node = self.tree
        # ids = self.nvariables
        ids = np.arange(self.nvariables)
        while isinstance(node, list):   
            id, x, p0, p1, node0, node1 = node
            assignx = datavec[x]
            ids=np.delete(ids, id, 0)
            if assignx==1:
                logprob += np.log(p1)
                node=node1
            else:
                logprob += np.log(p0)
                node = node0
        # After exiting the while loop, node is just a CLTree object (not a list, hence loop breaks)
        # the ids list is also updated to only contain ids pertaining to the leaf node (cltree)
        # it (cltree object) also has a log_prob method for a given data vector. 
        logprob += node.log_prob(datavec[ids])
        return logprob


    """
        Compute the log-likelihood score for the input dataset
    """
    def computeLL(self, dataset):
        prob = 0.0
        for i in range(dataset.shape[0]):
            vec = dataset[i]
            prob += self.log_prob(vec)
        return prob
    
    
    '''
        Update the parameters 
        1) if all datapoints has the same weight, update directly
        2) if different weights associated with each datapoint, update approximately
           using sampling
    '''
    def update(self,dataset_, weights=np.array([])):
        if weights.shape[0]==dataset_.shape[0]:
            norm_weights = utils_common.normalize(weights)
            indices = np.argwhere(np.random.multinomial(dataset_.shape[0], norm_weights)).ravel()
            dataset = dataset_[indices, :]
        else:
            dataset=dataset_
        for i in range(dataset.shape[0]):
            node=self.tree
            ids=np.arange(self.nvariables)
            while isinstance(node,list):
                id,x,p0,p1,node0,node1=node
                p0_index=2
                p1_index=3
                assignx=dataset[i,x]
                ids=np.delete(ids,id,0)
                if assignx==1:
                    node[p1_index]=p1+1.0
                    node=node1
                else:
                    node[p0_index]=p0+1.0
                    node = node0
            node.update(dataset[i:i+1,ids])


    '''
        Recursively learn the structure and parameter using weighted data
    '''
    def learn_structure_weight(self, dataset, weights, ids, smooth):
        curr_depth=self.nvariables-dataset.shape[1]
        
        
        if dataset.shape[0]<self.min_rec or dataset.shape[1]<self.min_var or curr_depth >= self.depth:
            clt = CLTree()
            clt.train(dataset) 
            clt.prob_pair = np.zeros((1, 1, 2, 2))
            clt.prob_sing = np.zeros((1, 2))             
            return clt
        
        pairs = utils_common.compute_pairwise_counts_weighted(dataset, weights) + smooth  # Laplace correction
        pairs = utils_common.normalize2D(pairs)
        singles = utils_common.compute_single_counts_weighted(dataset, weights) + 2.0 * smooth  # laplace correction
        singles = utils_common.normalize1D(singles)
        edgemat = utils_common.compute_adjmatrix(pairs, singles)
  
        np.fill_diagonal(edgemat, 0)        
        scores = np.sum(edgemat, axis=0)
        variable = np.argmax(scores)        
        
        index1 = np.where(dataset[:,variable]==1)[0]
        index0 = np.where(dataset[:,variable]==0)[0]        

        new_dataset =  np.delete(dataset, variable, axis = 1)        
        new_dataset1 = new_dataset[index1]
        new_weights1 = weights[index1]
        p1= np.sum(new_weights1)+smooth
                
        new_dataset0 = new_dataset[index0]
        new_weights0 = weights[index0]
        p0 = np.sum(new_weights0)+smooth
        
        # Normalize
        p0 = p0/(p0+p1)
        p1 = 1.0 - p0        
        
        new_ids=np.delete(ids,variable,0)
        
        return [variable,ids[variable],p0,p1,self.learn_structure_weight(new_dataset0,new_weights0,new_ids, smooth),
                self.learn_structure_weight(new_dataset1,new_weights1, new_ids, smooth)]
    
    
    '''
        Update the parameters using weighted data
    '''
    def update_parameter(self, node, dataset, weights, ids, smooth):
        
        if dataset.shape[0] == 0:
            return
        
        # internal nodes, not reach the leaf
        if isinstance(node,list):
            id,x,p0,p1,node0,node1 = node
            index1 = np.where(dataset[:,x]==1)[0]
            index0 = np.where(dataset[:,x]==0)[0]
            
            
            new_weights1 = weights[index1]
            new_weights0 = weights[index0]
            new_dataset1 = dataset[index1]
            new_dataset0 = dataset[index0]
            
            p1 = np.sum(new_weights1) + smooth
            p0 = np.sum(new_weights0) + smooth
            
            # Normalize
            p0 = p0/(p0+p1)
            p1 = 1.0 - p0
            
            
            node[2] = p0
            node[3] = p1
            
            new_ids = np.delete(ids, id)
            
            self.update_parameter(node0, new_dataset0, new_weights0, new_ids, smooth)
            self.update_parameter(node1, new_dataset1, new_weights1, new_ids, smooth)
        
        else:
            clt_dataset = dataset[:, ids]
            node.update_exact(clt_dataset, weights, structure_update_flag = False)
            return
       

    '''
        Update the CNet using weighted samples, exact update
    '''
    def update_exact(self, dataset, weights, structure_update_flag = False):
        
        if dataset.shape[0] != weights.shape[0]:
            print ('ERROR: weight size not equal to dataset size!!!')
            exit()
        # Perform based on weights
        # assume that dataset_.shape[0] equals weights.shape[0] because each example has a weight
        # try to avoid sum(weights = 0
        smooth = max(np.sum(weights), 1.0) / dataset.shape[0]
        ids = np.arange(dataset.shape[1])
        self.nvariables = dataset.shape[1]
       
        
        if structure_update_flag == True:
            # update the structure as well as parameters
            self.tree = self.learn_structure_weight(dataset, weights, ids, smooth)
        else:
            # only update parameters
            node=self.tree
            self.update_parameter(node, dataset, weights, ids,smooth)
            

    '''
        The recursivley part for function getWeights()
    '''   
    def get_prob_each(self, node, samples, row_index, ids, probs):
                
        if isinstance(node, list):
            id,x,p0,p1,node0,node1=node
            p0 = p0 / float(p0+p1)
            p1 = 1.0 - p0
            
            index1 = np.where(samples[:,id]==1)[0]
            index0 = np.where(samples[:,id]==0)[0]            
            row_index1 = row_index[index1]
            row_index0 = row_index[index0]         
            probs[row_index1] += np.log(p1)
            probs[row_index0] += np.log(p0)
            
            new_samples =  np.delete(samples, id, axis = 1)
            new_samples1 = new_samples[index1]
            new_samples0 = new_samples[index0]            
            new_ids = np.delete(ids, id)
            
            if new_samples0.shape[0] > 0:
                self.get_prob_each(node0, new_samples0, row_index0, new_ids, probs)
            if new_samples1.shape[0] > 0:
                self.get_prob_each(node1, new_samples1, row_index1, new_ids, probs)        
        # leaf node
        else:
            # clt_prob = node.getWeights (samples)
            clt_prob = utils_common.compute_LL(node, samples)             
            probs[row_index] += clt_prob
            
            
    '''
        Recursivley get the LL score for each datapoint
        Much faster than computeLL_each_datapoint
    ''' 
    def getWeights_LL(self, dataset):        
        probs = np.zeros(dataset.shape[0])
        row_index = np.arange(dataset.shape[0])
        ids = np.arange(self.nvariables)
        node = self.tree        
        self.get_prob_each(node, dataset, row_index, ids, probs)
        return probs
        

    def log_prob_new(self, datavec):
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



'''
   Main function for Learning Cutset Network from Data by given depth
'''    
def main_cutset():
    
    dataset_dir = sys.argv[2]
    data_name = sys.argv[4]
    depth = int(sys.argv[6])

    
    train_filename = dataset_dir + data_name + '.ts.data'
    test_filename = dataset_dir + data_name +'.test.data'
    valid_filename = dataset_dir + data_name + '.valid.data'
    

    train_dataset = np.loadtxt(train_filename, dtype=int, delimiter=',')
    valid_dataset = np.loadtxt(valid_filename, dtype=int, delimiter=',')
    test_dataset = np.loadtxt(test_filename, dtype=int, delimiter=',')
    #train_dataset = np.concatenate((train_dataset, valid_dataset), axis=0)
    

    print("Learning Cutset Networks.....")


    cnet = CutsetNet(depth=depth)
    cnet.train(train_dataset)
    

    train_ll =  np.sum(cnet.getWeights_LL(train_dataset)) / train_dataset.shape[0]
    valid_ll =  np.sum(cnet.getWeights_LL(valid_dataset)) / valid_dataset.shape[0]
    test_ll  =  np.sum(cnet.getWeights_LL(test_dataset))  / test_dataset.shape[0]

    print (train_ll)
    print (valid_ll)
    print (test_ll)


'''
   Main function for Learning an optimal Cutset Network from Data bounded by max depth
   Store the learned Cutset Network
'''  
def main_cutset_opt(parms_dict):
    
    print ("----------------------------------------------------")
    print ("Learning Cutset Networks on original data           ")
    print ("----------------------------------------------------")
    
    
    dataset_dir = parms_dict['dir']
    data_name = parms_dict['dn']
    max_depth = int(parms_dict['max_depth']) 
    out_dir = parms_dict['output_dir']

    

    train_filename = dataset_dir + data_name + '.ts.data'
    test_filename = dataset_dir + data_name +'.test.data'
    valid_filename = dataset_dir + data_name + '.valid.data'
    

    train_dataset = np.loadtxt(train_filename, dtype=int, delimiter=',')
    valid_dataset = np.loadtxt(valid_filename, dtype=int, delimiter=',')
    test_dataset = np.loadtxt(test_filename, dtype=int, delimiter=',')
    
    

    train_ll = np.zeros(max_depth)
    valid_ll = np.zeros(max_depth)
    test_ll = np.zeros(max_depth)
    
    best_valid = -np.inf
    best_module = None
    for i in range(1, max_depth+1):
        cnet = CutsetNet(depth=i)
        cnet.train(train_dataset)
        train_ll[i-1] = np.sum(cnet.getWeights_LL(train_dataset)) / train_dataset.shape[0]
        valid_ll[i-1] = np.sum(cnet.getWeights_LL(valid_dataset)) / valid_dataset.shape[0]
        test_ll[i-1] = np.sum(cnet.getWeights_LL(test_dataset))  / test_dataset.shape[0]
        
        if best_valid < valid_ll[i-1]:
            best_valid = valid_ll[i-1]
            best_module = copy.deepcopy(cnet)
            
    
    print('Train set cnet LL scores')
    for l in range(max_depth):
        print (train_ll[l], l+1)
    print()
    
    print('Valid set cnet LL scores')
    for l in xrange(max_depth):
        print (valid_ll[l], l+1)
    print()   
    
    print('test set cnet LL scores')
    for l in xrange(max_depth):
        print (test_ll[l], l+1)
        
    best_ind = np.argmax(valid_ll)
    
    print ()
    print ('Best Validation ll score achived in layer: ', best_ind )    
    print( 'Train set LL score: ', np.sum(best_module.getWeights_LL(train_dataset)) / train_dataset.shape[0])
    print( 'valid set LL score: ', np.sum(best_module.getWeights_LL(valid_dataset)) / valid_dataset.shape[0])
    print( 'test set LL score : ',np.sum(best_module.getWeights_LL(test_dataset)) / test_dataset.shape[0])
    
    # main_dict = {}
    # # utilM.save_cutset(main_dict, best_module.tree, np.arange(train_dataset.shape[1]), ccpt_flag = True)
    # np.savez_compressed(out_dir + data_name, module = main_dict)
    



