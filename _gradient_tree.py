import pandas as pd
import numpy as np
from numpy.linalg import inv
from scipy.optimize import fsolve

class GradientNode:
    """
    ### Note

    Depending on regression equation, following functions must be modified: 

    - self.get_theta_p_hat()
    - self.get_psi()
    - self.get_xi()
    - self.get_a_p()
    - self.get_inv_a_p()

    """

    def __init__(self, data_bootstrapped:pd.DataFrame, target:str, min_samples_leaf:int=5, depth:int=1, max_depth:int=5) -> None:
        self.data = data_bootstrapped
        self.target = target
        self.target_median = np.median(data_bootstrapped[target])

        # Hyperparameters
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth

        # Attributes
        self.left = None
        self.right = None
        self.depth = depth
        self.split_feature = ''
        self.split_point = 0
        self.label = ''
        self.estimate = self.get_theta_p_hat()

    def get_theta_p_hat(self) -> float:
        
        def sum_moment_condition(theta) -> float:  # Eq (4)
            y = self.data[self.target]

            # Depends on Regression equation
            return np.mean((y-theta))

        theta_0 = self.target_median # initial value
        res = fsolve(sum_moment_condition, theta_0)
        return res[0]
    
    def get_psi(self, y_c: list, theta_hat_p: float) -> np.ndarray:
        # Depends on Regression equation
        return np.array([np.mean([y_c - theta_hat_p])])
    
    def get_xi(self) -> np.ndarray:
        # Depends on Regression equation
        return np.array([1])
    
    def get_a_p(self, y_c: list, theta_hat_p: float) -> np.ndarray:
        # Depends on Regression equation
        return np.array([[-1]])
    
    def get_inv_a_p(self, a_p: np.ndarray) -> np.ndarray:
        if len(a_p) == 1:
            return 1/a_p
        else:
            return inv(a_p)

    def get_delta_tilde(self, left_subset:pd.DataFrame) -> float:
        X_left_subset = left_subset.drop(self.target, axis=1)
        y_left_subset = left_subset[self.target]

        X_right_subset = self.data.drop(X_left_subset.index, axis=0)
        X_right_subset = X_right_subset.drop(self.target, axis=1)
        y_right_subset = self.data.drop(y_left_subset.index, axis=0)
        y_right_subset = y_right_subset[self.target]

        theta_p_hat = self.get_theta_p_hat() # for y_i = b + u_i, it equals to mean of y_P

        xi = self.get_xi()
        a_p = self.get_a_p(self.data[self.target], theta_p_hat)
        psi_left = self.get_psi(y_left_subset, theta_p_hat)
        psi_right = self.get_psi(y_right_subset, theta_p_hat)
        
        inv_a_p = self.get_inv_a_p(a_p)
        
        theta_tilde_left = theta_p_hat - np.mean(xi.T @ inv_a_p @ psi_left)
        theta_tilde_right = theta_p_hat - np.mean(xi.T @ inv_a_p @ psi_right)

        # print("\nX:\n", self.data)
        # print("\ny_left:\n", y_left_subset)
        # print("\n\ny_right:\n", y_right_subset)
        # print(f"\n xi: {xi}, inv_a_p: {inv_a_p}, psi_left: {psi_left}, psi_right: {psi_right}")
        # print(f"len(X_left_subset): {len(X_left_subset)},  len(X_right_subset): {len(X_right_subset)}, len(self.data): {len(self.data)}, theta_tilde_left: {theta_tilde_left}, theta_tilde_right: {theta_tilde_right}")

        return len(X_left_subset) * len(X_right_subset) / (len(self.data)**2) * ((theta_tilde_left - theta_tilde_right) ** 2)
    
    def get_all_splits(self) -> dict: 
        features = self.data.columns
        splits = {}

        for feature in features:
            if feature == self.target:
                continue
            for unique in self.data[feature].unique():
                length_subset = len(self.data[self.data[feature] <= unique])
                if length_subset != 0 and length_subset != len(self.data):
                    splits[(feature, unique, 'continuous')] = self.data[self.data[feature] <= unique]
        return splits

    def split(self) -> None:
        # Terminal Condition
        if (self.depth >= self.max_depth or 
            len(self.data) <= self.min_samples_leaf or
            len(self.data.drop_duplicates()) <= 1
            ):

            self.label = '({}), n_leaf: {}'.format(
                self.estimate, len(self.data))
            return

        # Find split point
        splits = self.get_all_splits()

        delta_list = {}

        for key, split in splits.items():
            delta_list[key] = self.get_delta_tilde(split)
        
        selected_feature, split_point, _ = max(delta_list, key=delta_list.get)
        
        is_possible_split = False

        while not is_possible_split:
            left_node_data = self.data[
                self.data[selected_feature] <= split_point
            ]
            right_node_data = self.data[
                self.data[selected_feature] > split_point
            ]

            if left_node_data.shape[0] >= self.min_samples_leaf and right_node_data.shape[0] >= self.min_samples_leaf:
                is_possible_split = True
            else:
                del delta_list[(selected_feature, split_point, 'continuous')]
                if not delta_list:
                    self.label = '({}), n_leaf: {}'.format(
                    self.estimate, len(self.data))
                    return
                else:
                    selected_feature, split_point, _ = max(delta_list, key=delta_list.get)

        self.split_feature = selected_feature
        self.split_point = split_point


        self.label = "{} <= {}".format(
            self.split_feature, self.split_point
        )

        child_params = {
            'target': self.target,
            'min_samples_leaf': self.min_samples_leaf,
            'depth': self.depth + 1,
            'max_depth': self.max_depth
        }

        self.left = GradientNode(left_node_data, **child_params)
        self.right = GradientNode(right_node_data, **child_params)

        self.left.split()
        self.right.split()

        return

    def visualize(self, file_name:str, tree_idx:int) -> None:
        lines, _, _, _ = self._visualize_aux()

        with open("result/" + file_name, 'a') as file:
            file.writelines(f"Tree {tree_idx}\n")
            file.writelines(line + '\n' for line in lines)
            file.writelines("\n\n")

    def _visualize_aux(self) -> tuple: # code from https://stackoverflow.com/a/54074933/8650928
        # No child.
        if self.right is None and self.left is None:
            line = '%s' % self.label
            width = len(line)
            height = 1
            middle = width // 2
            return [line], width, height, middle

        # Only left child.
        if self.right is None:
            lines, n, p, x = self.left._visualize_aux()
            s = '%s' % self.label
            u = len(s)
            first_line = (x + 1) * ' ' + (n - x - 1) * '_' + s
            second_line = x * ' ' + '/' + (n - x - 1 + u) * ' '
            shifted_lines = [line + u * ' ' for line in lines]
            return [first_line, second_line] + shifted_lines, n + u, p + 2, n + u // 2

        # Only right child.
        if self.left is None:
            lines, n, p, x = self.right._visualize_aux()
            s = '%s' % self.label
            u = len(s)
            first_line = s + x * '_' + (n - x) * ' '
            second_line = (u + x) * ' ' + '\\' + (n - x - 1) * ' '
            shifted_lines = [u * ' ' + line for line in lines]
            return [first_line, second_line] + shifted_lines, n + u, p + 2, u // 2

        # Two children.
        left, n, p, x = self.left._visualize_aux()
        right, m, q, y = self.right._visualize_aux()
        s = '%s' % self.label
        u = len(s)
        first_line = (x + 1) * ' ' + (n - x - 1) * '_' + s + y * '_' + (m - y) * ' '
        second_line = x * ' ' + '/' + (n - x - 1 + u + y) * ' ' + '\\' + (m - y - 1) * ' '
        if p < q:
            left += [n * ' '] * (q - p)
        elif q < p:
            right += [m * ' '] * (p - q)
        zipped_lines = zip(left, right)
        lines = [first_line, second_line] + [a + u * ' ' + b for a, b in zipped_lines]
        return lines, n + m + u, max(p, q) + 2, n + u // 2

class GradientTree:
    def __init__(self, idx:int=0, min_samples_leaf:int=3, max_depth:int=5) -> None:
        self.root = None            # root node of the tree
        self.searching_node = None  # node being searched in prediction step
        self.idx = idx              # ID of the tree in the GRF
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
    
    def fit(self, bootstrapped_data:pd.DataFrame, target:str) -> None:
        self.root = GradientNode(data_bootstrapped=bootstrapped_data, target=target, min_samples_leaf=self.min_samples_leaf, depth=1, max_depth=self.max_depth)
        self.root.split()

        return self

    def predict(self, X: pd.DataFrame|pd.Series) -> list[float | bool]:
        if X.ndim == 1:
            self.searching_node = self.root
        
            while self.searching_node:
                if not self.searching_node.left and not self.searching_node.right:
                    return self.searching_node.estimate
                
                elif X[self.searching_node.split_feature] <= self.searching_node.split_point:
                    if self.searching_node.left is not None:
                        self.searching_node = self.searching_node.left
                    else:
                        return False
                
                elif X[self.searching_node.split_feature] > self.searching_node.split_point:
                    if self.searching_node.right is not None:
                        self.searching_node = self.searching_node.right
                    else:
                        return False

        else:
            feature_idx = {col:idx for idx, col in enumerate(X.columns)}
            X = X.to_numpy()  # Convert DataFrame to numpy array
            num_samples = X.shape[0]
            predictions = np.empty(num_samples, dtype=object)

            for i in range(num_samples):
                x = X[i]
                self.searching_node = self.root
                
                while self.searching_node:
                    if not self.searching_node.left and not self.searching_node.right:
                        predictions[i] = self.searching_node.estimate
                        break
                   
                    if x[feature_idx[self.searching_node.split_feature]] <= self.searching_node.split_point:
                        if self.searching_node.left is not None:
                            self.searching_node = self.searching_node.left
                        else:
                            predictions[i] = False
                            break
                    
                    else:
                        if self.searching_node.right is not None:
                            self.searching_node = self.searching_node.right
                        else:
                            predictions[i] = False
                            break
            
            return predictions

    def visualize(self, file_name:str) -> None:
        self.root.visualize(file_name, self.idx)
