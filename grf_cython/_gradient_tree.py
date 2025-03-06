import numpy as np
from numpy.random import RandomState

from ._tree import Tree, DepthFirstTreeBuilder
from ._splitter import BestSplitter
from ._criterion import GRFCriterion
from ._criterion_cf import GRFCriterionCF

class GradientTree():
    def __init__(self,
                 max_depth=5,
                 min_samples_split=10,
                 min_samples_leaf=5,
                 max_features=None,
                 random_state=None,
                 min_impurity_decrease=0.,
                 min_balancedness_tol=0.45,
                 model_spec:str="y=b+u",
                 honest=True):
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.min_impurity_decrease = min_impurity_decrease
        self.min_balancedness_tol = min_balancedness_tol
        self.model_spec = model_spec
        self.honest = honest

        if not max_features:
            raise ValueError("Valid value for max_features required.")
        else:
            self.max_features = max_features
        
        self.random_seed_ = random_state
        self.random_state = RandomState(self.random_seed_)
    
    def fit(self, X, y, T=None) -> None:
        self.X_parent = X
        self.y_parent = y
        self.T_parent = T
        self.n_samples, self.n_features = X.shape

        auxil_indices = np.arange(self.n_samples, dtype=np.intp)

        if self.honest:
            self.random_state.shuffle(auxil_indices)

            self.indices_train, self.indices_val = auxil_indices[:self.n_samples // 2], auxil_indices[self.n_samples // 2:]
        else:
            self.indices_train, self.indices_val = auxil_indices, auxil_indices

        self.tree_ = Tree(n_features=self.n_features)

        if self.model_spec=="y=b+u":
            criterion = GRFCriterion(n_samples=self.n_samples, 
                                    random_state=self.random_state.randint(np.iinfo(np.int32).max))
            
            criterion_val = GRFCriterion(n_samples=self.n_samples, 
                                    random_state=self.random_state.randint(np.iinfo(np.int32).max))
            
        elif self.model_spec=="y=a+bx+u":
            # No type check for T yet
            criterion = GRFCriterionCF(n_samples=self.n_samples, 
                                       random_state=self.random_state.randint(np.iinfo(np.int32).max))
            
            criterion_val = GRFCriterionCF(n_samples=self.n_samples, 
                                           random_state=self.random_state.randint(np.iinfo(np.int32).max))
        
        else:
            raise ValueError("GRF for other model specification is not implemented yet")


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

        builder.build(self.tree_, X, y.reshape(-1, 1), T.reshape(-1, 1), self.indices_train, self.indices_val)

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

        