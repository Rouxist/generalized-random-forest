import pandas as pd
import numpy as np
from numpy.random import RandomState
from collections import deque
from numpy.linalg import inv
from scipy.optimize import fsolve
from _splitter import BestSplitter

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

    def __init__(self, data_train:pd.DataFrame, data_val:pd.DataFrame, target:str, min_samples_leaf:int=5, depth:int=1, max_depth:int=5, min_balancedness_tol:float=0.45, random_state:int=None) -> None:
        self.data_train = data_train
        self.data_val = data_val
        self.target = target
        self.random_state = random_state

        # Hyperparameters
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.min_balancedness_tol = min_balancedness_tol

        # Attributes
        self.left = None
        self.right = None
        self.depth = depth
        self.split_feature = ''
        self.split_point = 0
        self.label = ''
        self.splitter = BestSplitter(data_train=self.data_train, 
                                     data_val=self.data_val, 
                                     target=self.target, 
                                     min_samples_leaf=self.min_samples_leaf,
                                     max_depth=self.max_depth,
                                     min_balancedness_tol=0.45)
        self.estimate = self.splitter.criterion.get_theta_p_hat(y_parent=self.data_train[self.target].to_numpy())

    def split(self) -> None:
        # Terminal Condition
        if (self.depth >= self.max_depth or 
            len(self.data_train) < 2 * self.min_samples_leaf or
            len(self.data_train.drop_duplicates()) <= 1
            ):

            self.label = '({}), n_leaf: {}'.format(
                self.estimate, len(self.data_train))
            return

        # Find split point
        selected_feature, split_point, p, p_val = self.splitter.get_best_node_split()

        self.split_feature = selected_feature
        self.split_point = split_point

        # Identify this node as leaf if none of the split point candidates was valid
        if (p==0 or p_val==0 or p==len(self.data_train)-1 or p_val==len(self.data_val)-1):

            self.label = '({}), n_leaf: {}'.format(
                self.estimate, len(self.data_train))
            return


        self.label = "{} <= {}".format(
            self.split_feature, self.split_point
        )

        rand_state = np.random.RandomState(self.random_state)

        sorted_data = self.data_train.copy()
        sorted_data = sorted_data.sort_values(selected_feature)
        
        sorted_data_val = self.data_val.copy()
        sorted_data_val = sorted_data_val.sort_values(selected_feature)

        left_child_params = {
            'data_train' : sorted_data[:p],
            'data_val' : sorted_data_val[:p_val],
            'target': self.target,
            'min_samples_leaf': self.min_samples_leaf,
            'depth': self.depth + 1,
            'max_depth': self.max_depth,
            'min_balancedness_tol': 0.45, 
            'random_state': rand_state.randint(0,10000) 
        }
        
        right_child_params = {
            'data_train' : sorted_data[p:],
            'data_val' : sorted_data_val[p_val:],
            'target': self.target,
            'min_samples_leaf': self.min_samples_leaf,
            'depth': self.depth + 1,
            'max_depth': self.max_depth,
            'min_balancedness_tol': 0.45, 
            'random_state': rand_state.randint(0,10000) 
        }

        self.left = GradientNode(**left_child_params)
        self.right = GradientNode(**right_child_params)

        return

    def visualize(self, file_name:str, tree_idx:int=0) -> None:
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
    def __init__(self, idx:int=0, min_samples_leaf:int=3, max_depth:int=5, min_balancedness_tol:float=0.45, honest:bool=True, random_state:int=None) -> None:
        self.root = None                                    # root node of the tree
        self.searching_node = None                          # node being searched in prediction step
        self.idx = idx                                      # ID of the tree in the GRF
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.min_balancedness_tol = min_balancedness_tol
        self.honest = honest
        self.seed = random_state
        self.random_state = RandomState(random_state)
    
    def fit(self, bootstrapped_data:pd.DataFrame, target:str) -> None:

        self.data_parent = bootstrapped_data
        self.data_indices = bootstrapped_data.index
        n_samples = len(bootstrapped_data)
        auxil_indices = np.arange(n_samples, dtype=np.intp)

        if self.honest:
            self.random_state.shuffle(auxil_indices)
            
            self.indices_train, self.indices_val = auxil_indices[:n_samples // 2], auxil_indices[n_samples // 2:]
        else:
            self.indices_train, self.indices_val = auxil_indices, auxil_indices

        root_node = GradientNode(data_train=self.data_parent.iloc[self.indices_train], 
                                 data_val=self.data_parent.iloc[self.indices_val], 
                                 target=target, 
                                 min_samples_leaf=self.min_samples_leaf, 
                                 depth=1, 
                                 max_depth=self.max_depth, 
                                 random_state=1 # To-Do
                                 )
        
        queue = deque([root_node])
        
        while queue:
            current_node = queue.popleft()
            if current_node is not None:
                current_node.split()
                queue.append(current_node.left)
                queue.append(current_node.right)

        self.root = root_node

        return self

    def predict(self, is_own_weight_set:bool, X: pd.DataFrame|pd.Series=None, col_names:list=None) -> list[float | bool]:
        if not is_own_weight_set:
            data = X
        else: 
            data = self.data_parent.iloc[self.indices_val]   # classifying this tree's own weight data
            data = data.iloc[:,1:]                       # remove target column
            data.columns = pd.Index(col_names)           # why do columns' names get shuffled

        if data.ndim == 1: 
            self.searching_node = self.root
        
            while self.searching_node:
                if not self.searching_node.left and not self.searching_node.right:
                    return self.searching_node.estimate
                
                elif data[self.searching_node.split_feature] <= self.searching_node.split_point:
                    if self.searching_node.left is not None:
                        self.searching_node = self.searching_node.left
                    else:
                        return False
                
                elif data[self.searching_node.split_feature] > self.searching_node.split_point:
                    if self.searching_node.right is not None:
                        self.searching_node = self.searching_node.right
                    else:
                        return False

        else:
            feature_idx = {col:idx for idx, col in enumerate(data.columns)}
            data = data.to_numpy()  # Convert DataFrame to numpy array
            num_samples = data.shape[0]
            predictions = np.empty(num_samples, dtype=object)

            for i in range(num_samples):
                x = data[i]
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
            
            if not is_own_weight_set:
                return predictions
            else:
                # print("now returning zip:\n",self.data_indices[self.indices_val], predictions)
                return self.data_indices[self.indices_val].to_list(), predictions

    def visualize(self, file_name:str) -> None:
        self.root.visualize(file_name, self.idx)
