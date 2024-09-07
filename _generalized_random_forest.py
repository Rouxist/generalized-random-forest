import pandas as pd
import numpy as np
from numpy.random import RandomState
from scipy.optimize import fsolve

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
    
    def __init__(self, target:str, n_estimators:int=5, min_samples_leaf:int=3, max_depth:int=5, honest:bool=True, data_weight_ratio:float = 0.5, random_state:int=None) -> None:
        self.target = target                          # name of the target variable column
        
        # Hyperparameters
        self.n_estimators = n_estimators              # # of the gradient trees to be fitted
        self.honest = honest                          # honesty
        self.data_weight_ratio = data_weight_ratio    # ratio of given dataset to be used when calculating weights for local estimation
        self.min_samples_leaf = min_samples_leaf      # minimum numbers of datapoints required for a new leaf when splitting
        self.max_depth = max_depth                    # max depth of branch of tree
        self.random_state = RandomState(random_state) # RandomState object
        
        # Attributes
        self.data_split = None                        # dataset to be used when applying splitting rules for each gradient tree
        self.data_weight = None                       # dataset to be used when calculating weights; self.data_split and self.data_weight are disjoint 
        self.tree_list = []                           # list of base estimators(gradient trees)
        self.alpha = None                             # list of weights of data points in self.data_weight
    
    def bootstrap_data(self, data:pd.DataFrame) -> tuple:
        bootstrap_indices = list(self.random_state.choice(range(len(data)), len(data), replace = True))
        oob_indices = [i for i in range(len(data)) if i not in bootstrap_indices]
        data_bootstrap = data.iloc[bootstrap_indices] # to turn into ndarray, data.iloc[bootstrap_indices].values
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
        for i in range(self.n_estimators):
            data_split_bootstrapped, _ = self.bootstrap_data(self.data_split)

            tree = GradientTree(idx=i+1, min_samples_leaf=self.min_samples_leaf, max_depth=self.max_depth)
            tree.fit(data_split_bootstrapped, self.target)
            self.tree_list.append(tree)
    
    def predict(self, x:pd.Series) -> float:
        # Initialize weights of each data point in data_weight
        self.alpha = np.zeros(len(self.data_weight))

        # Calculates weights of each data point in data_weight
        for tree in self.tree_list:
            neighbors = []
            estimate_given_datapoint = tree.predict(x)
            for index, (idx, data_point) in enumerate(self.data_weight.iterrows()):
                estimate_weight_data = tree.predict(data_point)
                if estimate_weight_data == estimate_given_datapoint: # if the datapoint is in same leaf with the given datapoint
                    neighbors.append(index) # alt: neighbors.append(data_point.name)

            # Update weights
            for neighbor in neighbors:
                self.alpha[neighbor] += 1 / len(neighbors) / self.n_estimators

        def sum_moment_condition(theta) -> float: # Eq (2)
            y = self.data_weight[self.target]

            # Depends on Regression equation
            return np.sum(self.alpha.dot(y-theta))

        theta_0 = np.mean(self.data_weight[self.target])
        res = fsolve(sum_moment_condition, theta_0)
        return round(res[0], 2)

    def visualize(self, file_name:str) -> None:
        if len(self.tree_list) != 0:
            for tree in self.tree_list:
                tree.visualize(file_name)
