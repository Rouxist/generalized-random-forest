import math
import numpy as np
from numpy.random import RandomState
from scipy.optimize import fsolve
from joblib import Parallel, delayed

from ._gradient_tree import GradientTree

MAX_INT = np.iinfo(np.int32).max

class GRF:
    def __init__(self, 
                 n_estimators:int=100, 
                 min_samples_leaf:int=5,
                 max_depth:int=5, 
                 max_features:int=None, 
                 honest:bool=True, 
                 subforest_size:int=4, 
                 block_size:int=1,
                 quantile:float=0.5,
                 random_state:int=None) -> None:

        # Hyperparameters
        self.n_estimators = n_estimators                  # # of the gradient trees to be fitted
        self.min_samples_leaf = min_samples_leaf          # minimum numbers of datapoints in a leaf node
        self.max_depth = max_depth                        # max depth of branch of tree
        self.max_features = max_features                  # max featuers to be explored for splitting
        self.honest = honest                              # honesty
        self.subforest_size = subforest_size
        self.block_size = block_size                      # block size to be used as a parameter of block sampling.
        self.quantile = quantile                          # quantile for quantile regression
        self.random_state = RandomState(random_state)     # RandomState object

        # Attributes
        self.estimators_ = []                             # list of estimators (gradient trees)
        self.subsample_random_state_seed = 0

    def fit(self, X, y) -> None:
        # Subsample generation
        self.subsample_random_state_seed = self.random_state.randint(MAX_INT)
        subsample_random_state = np.random.RandomState(self.subsample_random_state_seed)

        n_samples = X.shape[0]
        n_samples_subsample = int(np.floor(n_samples * 0.45))
        n_blocks = int(n_samples_subsample // self.block_size) + 1

        n_groups = self.n_estimators // self.subforest_size
        estimator_idx_groups = np.array_split(np.arange(0, self.n_estimators), n_groups)

        slice_indices = []

        if self.block_size == 1:
            for estimator_indices in estimator_idx_groups:
                half_sample_inds = subsample_random_state.choice(n_samples, n_samples // 2, replace=False)
                slice_indices.extend([half_sample_inds[subsample_random_state.choice(n_samples // 2,
                                                                                    n_samples_subsample,
                                                                                    replace=False)]
                                      for _ in range(len(estimator_indices))])
        else:
            for estimator_indices in estimator_idx_groups:
                for _ in range(len(estimator_indices)):
                    block_start_indices = subsample_random_state.choice(n_samples - self.block_size + 1, n_blocks, replace=False)
                    block_sampled_data = []
                    for start_idx in block_start_indices:
                        block_sampled_data.extend([i for i in range(start_idx, start_idx + self.block_size)])
                    block_sampled_data = np.array(block_sampled_data)
                    block_sampled_data = block_sampled_data[:n_samples_subsample]
                    slice_indices.append(block_sampled_data)

        # Fit gradient trees
        trees = []

        for _ in range(self.n_estimators):
            seed = self.random_state.randint(MAX_INT)
            tree = GradientTree(max_features=self.max_features, 
                                min_samples_leaf=self.min_samples_leaf,
                                max_depth=self.max_depth,
                                honest=True,
                                random_state=seed)
            trees.append(tree)

        trees_fitted = Parallel(n_jobs=4, backend="threading")(
            delayed(tree.fit)(X[slice], y[slice])
            for slice, tree in zip(slice_indices, trees))

        self.estimators_.extend(trees_fitted)

    def predict(self, X)->np.ndarray:
        val_X_list = [self.estimators_[i].X_parent[self.estimators_[i].indices_val,:]
                            for i in range(self.n_estimators)]
        val_y_list = [self.estimators_[i].y_parent[self.estimators_[i].indices_val]
                            for i in range(self.n_estimators)]
        
        # Pool data from all trees
        pooled_val_X = np.concatenate(val_X_list)
        pooled_val_y = np.concatenate(val_y_list)
        if pooled_val_y.ndim == 1:
            pooled_val_y = np.expand_dims(pooled_val_y, (-1))
        pooled_data = np.concatenate([pooled_val_y, pooled_val_X], axis=1)

        # Drop duplicates
        aggr_val_data = np.unique(pooled_data, axis=0)
        aggr_val_X = aggr_val_data[:,1:].copy()
        aggr_val_y = aggr_val_data[:,0].copy()
        aggr_val_y = aggr_val_y.tolist()
        n_samples_val = aggr_val_X.shape[0]

        # Pre-calculate indices of {val datapoints in each tree} in the aggregated dataset
        indices_from_whole = [[np.where((aggr_val_X == dp).all(axis=1))[0].squeeze() for dp in val_X_list[tree_idx]] 
                              for tree_idx in range(self.n_estimators)]

        # Output initialization
        n_given_datapoints = X.shape[0]
        predictions = np.zeros(n_given_datapoints)

        leaf_matrices = [self.estimators_[tree_idx].get_weight(val_X_list[tree_idx]) 
                          for tree_idx in range(self.n_estimators)]

        
        # Main prediction procedure
        for dp_idx in range(n_given_datapoints):
            # index of leaf node from each tree, given a single datapoint
            leaf_node_idx_list = [self.estimators_[tree_idx].apply(np.expand_dims(X[dp_idx], axis=0))
                            for tree_idx in range(self.n_estimators)]

            # element of this list is a list of index of y values of datapoints which are in leaf node of each tree
            leaf_y_idx_list = [[x.item() for x, y in zip(indices_from_whole[tree_idx], leaf_matrices[tree_idx][leaf_node_idx_list[tree_idx].item()]) if y == 1] 
                               for tree_idx in range(self.n_estimators)]
            
            # element of this list is a list of y values of datapoints which are in leaf node of each tree
            leaf_y_list = [[aggr_val_y[idx] for idx in leaf_y_idx_list[tree_idx]] for tree_idx in range(self.n_estimators)]

            # element of this list is y values in leaf nodes, including all duplicates
            pooled_leaf_y = sorted([x for sublist in leaf_y_list for x in sublist]) # flattening

            # Quantile calculate
            n = len(pooled_leaf_y)
            q_idx = self.quantile * (n - 1)
            low = math.floor(q_idx)
            high = low + 1 if low + 1 < n else low
            weight = q_idx - low
            res = pooled_leaf_y[low] * (1 - weight) + pooled_leaf_y[high] * weight

            predictions[dp_idx] = res

        return predictions
