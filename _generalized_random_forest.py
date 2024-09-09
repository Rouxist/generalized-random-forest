import pandas as pd
import numpy as np
from numpy.random import RandomState
from scipy.optimize import fsolve
from joblib import Parallel, delayed

from _gradient_tree import GradientTree

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
    
    def __init__(self, target:str, n_estimators:int=5, min_samples_leaf:int=3, max_depth:int=5, honest:bool=True, data_weight_ratio:float = 0.5, min_balancedness_tol:float=0.45, random_state:int=None) -> None:
        self.target = target                          # name of the target variable column
        
        # Hyperparameters
        self.n_estimators = n_estimators                  # # of the gradient trees to be fitted
        self.honest = honest                              # honesty
        self.data_weight_ratio = data_weight_ratio        # ratio of given dataset to be used when calculating weights for local estimation
        self.min_samples_leaf = min_samples_leaf          # minimum numbers of datapoints required for a new leaf when splitting
        self.max_depth = max_depth                        # max depth of branch of tree
        self.random_state = RandomState(random_state)     # RandomState object
        self.min_balancedness_tol = min_balancedness_tol  # 1. each child node must contain at least this portion of data points. 2. each tree is trained on this portion of whole train data.
        
        # Attributes
        self.data_split = None                            # dataset to be used when applying splitting rules for each gradient tree
        self.data_weight = None                           # dataset to be used when calculating weights; self.data_split and self.data_weight are disjoint 
        self.tree_list = []                               # list of base estimators(gradient trees)
        self.alpha = None                                 # list of weights of data points in self.data_weight
    
    def bootstrap_data(self, data:pd.DataFrame, random_state:int) -> tuple:
        rand_state = np.random.RandomState(random_state)
        bootstrap_indices = list(rand_state.choice(range(len(data)), len(data), replace = True))
        oob_indices = [i for i in range(len(data)) if i not in bootstrap_indices]
        data_bootstrap = data.iloc[bootstrap_indices] # to turn into ndarray, data.iloc[bootstrap_indices].values
        n_bootstrapped_data = round(len(data_bootstrap) * self.min_balancedness_tol)
        data_bootstrap = data_bootstrap.iloc[:n_bootstrapped_data]
        data_oob = data.iloc[oob_indices].values

        return data_bootstrap, data_oob
        
    def fit(self, data:pd.DataFrame) -> None:
        indices = np.arange(len(data))
        if self.honest:
            # Divide dataset into `split set` and `weight set` to satisfy honesty condition
            self.random_state.shuffle(indices)
            indices_train, indices_val = indices[:len(data) // 2], indices[len(data) // 2:]
        else:
            indices_train, indices_val = indices, indices
        self.data_weight = data.iloc[indices_train]
        self.data_split = data.iloc[indices_val]
        
        # Fit gradient trees
        trees = []
        datas = []
        for i in range(self.n_estimators):
            seed = self.random_state.randint(0,10000)
            tree = GradientTree(idx=i+1, min_samples_leaf=self.min_samples_leaf, max_depth=self.max_depth, random_state=seed)
            trees.append(tree)
            data, _ = self.bootstrap_data(data=self.data_split, random_state=seed)
            datas.append(data)
        
        trees_fitted = Parallel(n_jobs=-1, backend="loky")(
            delayed(tree.fit)(bootstrapped_data=data, target=self.target) 
            for data, tree in zip(datas, trees))

        self.tree_list.extend(trees_fitted)
    
    def predict(self, x:pd.DataFrame|pd.Series) -> float:
        ### new predict func
        n_samples = 1 if x.ndim == 1 else x.shape[0]
        self.alpha = [np.zeros(len(self.data_weight)) for _ in range(n_samples)]
        predictions = []
        
        def sum_moment_condition(theta, dp_idx) -> float: # Eq (2)
            y = self.data_weight[self.target]

            # Depends on Regression equation
            return np.sum(self.alpha[dp_idx].dot(y-theta))

        if n_samples == 1:
            for tree in self.tree_list:
                estimate_given_dp = tree.predict(x)
                multi = tree.predict(self.data_weight)
                neighbors = [idx for (idx, weight_data_estim) in enumerate(multi) if estimate_given_dp==weight_data_estim]

            # Update weights
            for neighbor in neighbors:
                self.alpha[0][neighbor] += 1 / len(neighbors) / self.n_estimators

            theta_0 = np.mean(self.data_weight[self.target])
            res = fsolve(sum_moment_condition, theta_0, 0)
            return res[0]
        else:
            # simply iterates datapoint with time complexity O(n)
            estimate_given_dps = [tree.predict(x) for tree in self.tree_list] # length of n_estimator and each element has length of given datapoints
            for dp_idx in range(len(x)):
                for tree_idx in range(self.n_estimators):
                    # Find neighbors
                    multi = self.tree_list[tree_idx].predict(self.data_weight)
                    neighbors = [idx for (idx, weight_data_estim) in enumerate(multi) if estimate_given_dps[tree_idx][dp_idx]==weight_data_estim]
                    
                    # Update weights
                    if len(neighbors) > 0:
                        for neighbor in neighbors:
                            self.alpha[dp_idx][neighbor] += 1 / len(neighbors) / self.n_estimators

                theta_0 = np.mean(self.data_weight[self.target])
                res = fsolve(sum_moment_condition, theta_0, dp_idx)
                predictions.append(res[0])
            return predictions

    def visualize(self, file_name:str) -> None:
        if len(self.tree_list) != 0:
            for tree in self.tree_list:
                tree.visualize(file_name)
