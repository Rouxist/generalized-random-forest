import numpy as np
import pandas as pd
from _criterion import GRFCriterion

class BestSplitter:
    def __init__(self, data_train:pd.DataFrame, data_val:pd.DataFrame, target:str, min_samples_leaf:int=5, max_depth:int=5, min_balancedness_tol:float=0.45, honest:bool=True) -> None:
        self.data_train = data_train
        self.data_val = data_val
        self.target = target
        feature_list = self.data_train.columns.to_list()
        feature_list.remove("price")
        self.features = feature_list
        self.indices = self.data_train.index
        self.indices_val = self.data_val.index

        self.n_data = len(self.data_train)
        self.n_data_val = len(self.data_val)
        self.n_features = len(self.features)

        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.min_balancedness_tol = min_balancedness_tol
        self.honest = honest

        self.criterion = GRFCriterion()

    def get_best_node_split(self) -> tuple:
        # actual_idx = {i:idx for i, idx in enumerate(self.indices)}
        # actual_idx_val = {i:idx for i, idx in enumerate(self.indices_val)}

        best_feature = self.features[0]
        best_split_point = 0
        best_proxy_delta_tilde = 0 # always positive
        best_p, best_p_val = 0,0

        # explore all possible features
        for feature in self.features:
            if feature == self.target:
                continue
            Xf = self.data_train[feature].copy().sort_values()
            Xf_val = self.data_val[feature].copy().sort_values()

            # explore all possible splits of the feature
            p, p_val = 0, 0

            while p < self.n_data - 1 and p_val < self.n_data_val - 1:
                p += 1 # if indices of left-child-data is 0 1 2 and that of right-child-data is 3 4 5, than p=3

                if Xf.iloc[p-1] == Xf.iloc[p]:
                    continue
                
                split_point = Xf.iloc[p-1] / 2.0 + Xf.iloc[p] / 2.0 # split point candidate

                if self.honest:
                    while p_val < len(Xf_val) and Xf_val.iloc[p_val] < split_point:
                        p_val += 1
                else:
                    p_val = p

                # check validity conditions
                if (self.n_data - p) < (.5 - self.min_balancedness_tol) * (self.n_data - 0):
                    # print("broke by tolerance constraint")
                    break
                if (p_val - 0) < (.5 - self.min_balancedness_tol) * (self.n_data_val - 0):
                    # print("continued by tolerance(val) constraint")
                    continue
                if (self.n_data_val - p_val) < (.5 - self.min_balancedness_tol) * (self.n_data_val - 0):
                    # print("broke by tolerance(val) constraint")
                    break


                if (p - 0) < self.min_samples_leaf:
                    # print("continued by min_samples_split constraint")
                    continue
                if (self.n_data - p) < self.min_samples_leaf:
                    break
                # Reject if min_samples_leaf is not guaranteed on val
                if (p_val - 0) < self.min_samples_leaf:
                    # print("continued by min_samples_split(val) constraint")
                    continue
                if (self.n_data_val - p_val) < self.min_samples_leaf:
                    # print("broke by min_samples_split(val) constraint")
                    break

                # skipped min_weight_leaf, min_eig_leaf conditions

                y_data = self.data_train.sort_values(feature)[self.target].to_numpy()
                proxy_delta_tilde = self.criterion.get_proxy_delta_tilde(y_left=y_data[:p], y_right=y_data[p:])

                if proxy_delta_tilde > best_proxy_delta_tilde:
                    best_p = p
                    best_p_val = p_val
                    best_proxy_delta_tilde = proxy_delta_tilde
                    best_feature = feature
                    best_split_point = split_point

        return best_feature, best_split_point, best_p, best_p_val
    