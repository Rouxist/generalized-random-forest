import pandas as pd
import numpy as np
from numpy.random import RandomState
from scipy.optimize import fsolve
from joblib import Parallel, delayed

from _gradient_tree import GradientTree

MAX_INT = np.iinfo(np.int32).max

class GRF:
    """
    ### Methods

    self.fit()
        It divides given dataset into two subsets to satisfy hoensty condition and fits multiple gradient trees on bootstrapped subsets.
    
    self.predict()
        Estimates $\theta$ of given data point after calculating weights.
    
    self.visualize()
        Creates .txt file containing visualization of fitted gradient trees.

    """
    
    def __init__(self, target:str, n_estimators:int=5, min_samples_leaf:int=3, max_depth:int=5, honest:bool=True, data_weight_ratio:float = 0.5, max_samples:float=.45, random_state:int=None) -> None:
        self.target = target                          # name of the target variable column
        
        # Hyperparameters
        self.n_estimators = n_estimators                  # # of the gradient trees to be fitted
        self.honest = honest                              # honesty
        self.data_weight_ratio = data_weight_ratio        # ratio of given dataset to be used when calculating weights for local estimation
        self.min_samples_leaf = min_samples_leaf          # minimum numbers of datapoints required for a new leaf when splitting
        self.max_depth = max_depth                        # max depth of branch of tree
        self.seed = random_state                          # seed
        self.random_state = RandomState(random_state)     # RandomState object
        self.max_samples = max_samples                    # each tree is trained on this portion of whole train data.
        
        # Attributes
        # self.data_split = None                            # dataset to be used when applying splitting rules for each gradient tree
        # self.data_weight = None                           # dataset to be used when calculating weights; self.data_split and self.data_weight are disjoint 
        self.tree_list = []                               # list of base estimators(gradient trees)
        self.alpha = None                                 # list of weights of data points in self.data_weight
        
    def fit(self, data:pd.DataFrame) -> None:
        self.data_columns = data.columns
        self.subsample_random_state_seed = self.random_state.randint(MAX_INT) # `self.subsample_random_seed_` of econml _base_grf.py
        print("random state for sampling subsample (from scratch):",self.subsample_random_state_seed)
        subsample_random_state = np.random.RandomState(self.subsample_random_state_seed)

        # Subsample generation

        n_samples = len(data)
        n_samples_subsample = int(np.floor(n_samples * self.max_samples))
        print("n_samples_subsample:",n_samples_subsample)

        slice_indices = []

        half_sample_inds = subsample_random_state.choice(n_samples, n_samples // 2, replace=False)
        slice_indices.extend([half_sample_inds[subsample_random_state.choice(n_samples // 2,
                                                                        n_samples_subsample,
                                                                        replace=False)]
                            for _ in range(self.n_estimators)])
        
        # Fit gradient trees
        trees = []

        for idx in range(self.n_estimators):
            seed = self.random_state.randint(MAX_INT)
            tree = GradientTree(idx=idx+1, min_samples_leaf=self.min_samples_leaf, max_depth=self.max_depth, honest=True, random_state=seed)
            trees.append(tree)

        trees_fitted = Parallel(n_jobs=-1, backend="loky")(
            delayed(tree.fit)(bootstrapped_data=data.iloc[slice], target=self.target)
            for slice, tree in zip(slice_indices, trees))

        self.tree_list.extend(trees_fitted)
    
    def predict(self, x:pd.DataFrame|pd.Series) -> float:
        data_weight_list = [self.tree_list[i].data.iloc[self.tree_list[i].indices_weight] for i in range(self.n_estimators)]
    
        for split_data in data_weight_list:
            split_data.columns = self.data_columns # why do columns' names get shuffled

        whole_data_weight = pd.concat(data_weight_list)
        whole_data_weight.drop_duplicates(inplace=True)
        n_samples_weight = len(whole_data_weight)
        whole_data_weight_indices = whole_data_weight.index.to_list()

        ### new predict func
        n_given_datapoints = 1 if x.ndim == 1 else x.shape[0]
        self.alpha = [np.zeros(n_samples_weight) for _ in range(n_given_datapoints)]
        predictions = []
        
        def sum_moment_condition(theta, dp_idx) -> float: # Eq (2)
            y = whole_data_weight[self.target]

            # Depends on Regression equation
            return np.sum(self.alpha[dp_idx].dot(y-theta))

        if n_given_datapoints == 1:
            # for tree in self.tree_list:
            #     estimate_given_dp = tree.predict(x)
            #     multi = tree.predict(data_weight)
            #     neighbors = [idx for (idx, weight_data_estim) in enumerate(multi) if estimate_given_dp==weight_data_estim]

            # # Update weights
            # for neighbor in neighbors:
            #     self.alpha[0][neighbor] += 1 / len(neighbors) / self.n_estimators

            # theta_0 = np.mean(self.data_weight[self.target])
            # res = fsolve(sum_moment_condition, theta_0, 0)
            # return res[0]
            pass # To-Do
        else:
            # simply iterates datapoint with time complexity O(n)
            estimate_given_dps = [tree.predict(is_own_weight_set=False, X=x) for tree in self.tree_list] # length of n_estimator and each element has length of given datapoints

            for dp_idx in range(n_given_datapoints):

                for tree_idx, tree in enumerate(self.tree_list):
                    # Find neighbors
                    indices, preds = tree.predict(is_own_weight_set=True, col_names=x.columns)
                    prediction_pairs = zip(indices, preds)
                    neighbors = [idx for (idx, weight_data_estim) in prediction_pairs if estimate_given_dps[tree_idx][dp_idx]==weight_data_estim]

                    # Update weights
                    if len(neighbors) > 0:
                        for neighbor_idx in neighbors:
                            self.alpha[dp_idx][whole_data_weight_indices.index(neighbor_idx)] += 1 / len(neighbors) / self.n_estimators

                theta_0 = np.mean(whole_data_weight[self.target])
                res = fsolve(sum_moment_condition, theta_0, dp_idx)
                predictions.append(res[0])
                
            return predictions

    def visualize(self, file_name:str) -> None:
        if len(self.tree_list) != 0:
            for tree in self.tree_list:
                tree.visualize(file_name)
