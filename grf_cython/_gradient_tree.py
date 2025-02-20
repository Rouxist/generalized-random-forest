import numpy as np
from numpy.random import RandomState

from ._tree import Tree, DepthFirstTreeBuilder
from ._splitter import BestSplitter
from ._criterion_qf import GRFCriterionQF

class GradientTree():
    def __init__(self,
                 max_depth=5,
                 min_samples_split=10,
                 min_samples_leaf=5,
                 max_features=None,
                 random_state=None,
                 min_impurity_decrease=0.,
                 min_balancedness_tol=0.45,
                 quantile=0.5,
                 honest=True):
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.min_impurity_decrease = min_impurity_decrease
        self.min_balancedness_tol = min_balancedness_tol
        self.quantile = quantile
        self.honest = honest

        if not max_features:
            raise ValueError("Valid value for max_features required.")
        else:
            self.max_features = max_features
        
        self.random_seed_ = random_state
        self.random_state = RandomState(self.random_seed_)
    
    def fit(self, X, y) -> None:
        self.X_parent = X
        self.y_parent = y
        self.n_samples, self.n_features = X.shape

        auxil_indices = np.arange(self.n_samples, dtype=np.intp)

        if self.honest:
            self.random_state.shuffle(auxil_indices)

            self.indices_train, self.indices_val = auxil_indices[:self.n_samples // 2], auxil_indices[self.n_samples // 2:]
        else:
            self.indices_train, self.indices_val = auxil_indices, auxil_indices

        self.tree_ = Tree(n_features=self.n_features)


        criterion = GRFCriterionQF(n_samples=self.n_samples, 
                                   quantile=self.quantile,
                                   random_state=self.random_state.randint(np.iinfo(np.int32).max))
        
        criterion_val = GRFCriterionQF(n_samples=self.n_samples, 
                                       quantile=self.quantile,
                                       random_state=self.random_state.randint(np.iinfo(np.int32).max))

        splitter = BestSplitter(criterion=criterion,
                                criterion_val=criterion_val,
                                max_features=self.max_features, 
                                min_samples_leaf=self.min_samples_leaf, 
                                min_balancedness_tol=self.min_balancedness_tol, 
                                honest=self.honest, 
                                random_state = self.random_state.randint(np.iinfo(np.int32).max))
        
        builder = DepthFirstTreeBuilder(splitter, self.min_samples_split,
                                        self.min_samples_leaf,
                                        0,
                                        self.max_depth,
                                        self.min_impurity_decrease,)

        builder.build(self.tree_, X, y.reshape(-1, 1), self.indices_train, self.indices_val)

        return self

    def predict(self, X) -> np.ndarray:
        out = self.tree_.predict(X)
        return out
    
    def apply(self, X) -> np.ndarray:
        out = self.tree_.apply(X)
        return out
    
    def get_weight(self, X) -> np.ndarray:
        out = self.tree_.get_weight(X)
        return out

        