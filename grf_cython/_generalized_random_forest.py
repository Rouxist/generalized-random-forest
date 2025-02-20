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
                                quantile=self.quantile,
                                honest=True,
                                random_state=seed)
            trees.append(tree)

        trees_fitted = Parallel(n_jobs=4, backend="threading")(
            delayed(tree.fit)(X[slice], y[slice])
            for slice, tree in zip(slice_indices, trees))

        self.estimators_.extend(trees_fitted)

    def predict(self, X)->np.ndarray:
        # Output initialization
        n_given_datapoints = X.shape[0]
        predictions = np.zeros(n_given_datapoints)

        # Main prediction procedure
        for dp_idx in range(n_given_datapoints):
            pred_list = [self.estimators_[tree_idx].predict(np.expand_dims(X[dp_idx], axis=0)) 
                         for tree_idx in range(self.n_estimators)]

            predictions[dp_idx] = sum(pred_list) / len(pred_list)

        return predictions
