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
                 model_spec:str="y=b+u",
                 random_state:int=None) -> None:

        # Hyperparameters
        self.n_estimators = n_estimators                  # # of the gradient trees to be fitted
        self.min_samples_leaf = min_samples_leaf          # minimum numbers of datapoints in a leaf node
        self.max_depth = max_depth                        # max depth of branch of tree
        self.max_features = max_features                  # max featuers to be explored for splitting
        self.honest = honest                              # honesty
        self.subforest_size = subforest_size
        self.block_size = block_size                      # block size to be used as a parameter of block sampling.
        self.model_spec = model_spec                      # model specification
        self.random_state = RandomState(random_state)     # RandomState object

        # Attributes
        self.estimators_ = []                             # list of estimators (gradient trees)
        self.subsample_random_state_seed = 0

    def fit(self, X, y, T=None) -> None:
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
                                model_spec=self.model_spec,
                                random_state=seed)
            trees.append(tree)

        if self.model_spec=="y=b+u":
            trees_fitted = Parallel(n_jobs=4, backend="threading")(
                delayed(tree.fit)(X[slice], y[slice])
                for slice, tree in zip(slice_indices, trees))
        else:
            trees_fitted = Parallel(n_jobs=4, backend="threading")(
                delayed(tree.fit)(X[slice], y[slice], T[slice])
                for slice, tree in zip(slice_indices, trees))

        self.estimators_.extend(trees_fitted)

    def predict(self, X)->np.ndarray:
        val_X_list = [self.estimators_[i].X_parent[self.estimators_[i].indices_val,:]
                            for i in range(self.n_estimators)]
        val_y_list = [self.estimators_[i].y_parent[self.estimators_[i].indices_val]
                            for i in range(self.n_estimators)]
        val_T_list = [self.estimators_[i].T_parent[self.estimators_[i].indices_val]
                            for i in range(self.n_estimators)]
        
        # Pool data from all trees
        pooled_val_X = np.concatenate(val_X_list)
        pooled_val_y = np.concatenate(val_y_list)
        pooled_val_T = np.concatenate(val_T_list)
        if pooled_val_y.ndim == 1:
            pooled_val_y = np.expand_dims(pooled_val_y, (-1))
        if pooled_val_T.ndim == 1:
            pooled_val_T = np.expand_dims(pooled_val_T, (-1))
        pooled_data = np.concatenate([pooled_val_T, pooled_val_y, pooled_val_X], axis=1)

        # Drop duplicates
        aggr_val_data = np.unique(pooled_data, axis=0)
        aggr_val_X = aggr_val_data[:,2:].copy()
        aggr_val_y = aggr_val_data[:,1].copy()
        aggr_val_T = aggr_val_data[:,0].copy()
        n_samples_val = aggr_val_X.shape[0]

        # Pre-calculate indices of {val datapoints in each tree} in the aggregated dataset
        indices_from_whole = [[np.where((aggr_val_X == dp).all(axis=1))[0].squeeze() for dp in val_X_list[tree_idx]] for tree_idx in range(self.n_estimators)]

        # Moment condition setup        
        def sum_moment_condition(theta, alpha) -> float: # Eq (2) of Athey, S., Tibshirani, J., & Wager, S. (2019). Generalized random forests.
            return np.sum(alpha.dot(aggr_val_y-theta))
        
        def sum_moment_conditions(params, y, T, alpha) -> float:
            # Moment condition for regression equation y_i = \nu + \theta T_i + u_i
            theta = params[0]
            nu = params[1]

            moment1 = np.sum(alpha.dot(T * (y - nu - theta * T)))
            moment2 = np.sum(alpha.dot(y - nu - theta * T))

            return np.array([moment1, moment2])
        
        # Output initialization
        n_given_datapoints = X.shape[0]
        predictions = np.zeros(n_given_datapoints)

        """
        Comment: leaf_matrix
        
        One gradient tree has one leaf_matrix with size of (node_count, n_samples_val).
        It represents which datapoints in validation set fall to certain leaf node.
        It helps to find neighbor datapoints easily.
        
        Rows of non-leaf node is full of zero.

        Example: leaf_matrix of a gradient tree with 3 nodes, given validation set with 23 datapoints:
        [[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]      << root node
        [0 1 0 1 1 0 1 0 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1]       << leaf node
        [1 0 1 0 0 1 0 1 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0]]      << leaf node
        """
        leaf_matrices = [self.estimators_[tree_idx].get_weight(val_X_list[tree_idx]) 
                          for tree_idx in range(self.n_estimators)]
        
        # Main prediction procedure
        for dp_idx in range(n_given_datapoints):
            alpha = np.zeros((n_samples_val))
            leaf_indices = np.concatenate([tree.apply(np.expand_dims(X[dp_idx], axis=0)) for tree in self.estimators_])
            # leaf_indices = np.concatenate(Parallel(n_jobs=4, backend="threading")(delayed(tree.apply)(np.expand_dims(X[dp_idx], axis=0)) for tree in self.estimators_))

            for tree_idx in range(self.n_estimators):
                leaf_idx = leaf_indices[tree_idx]
                neighbors = (leaf_matrices[tree_idx][leaf_idx] > 0).squeeze()

                for i, neighbor in enumerate(neighbors):
                    if neighbor:
                        alpha[indices_from_whole[tree_idx][i]] += 1/sum(neighbors)/self.n_estimators
            
            # Initial guess
            theta_0 = np.array([0.0, np.mean(aggr_val_y)])
            
            # Solve the equation using fsolve
            result = fsolve(sum_moment_conditions, theta_0, args=(aggr_val_y, aggr_val_T, alpha))
            predictions[dp_idx] = result[0]

        return predictions
