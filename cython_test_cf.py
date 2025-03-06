# Example is from: https://github.com/py-why/EconML/blob/main/notebooks/Generalized%20Random%20Forests.ipynb

from grf_cython import GRF
import numpy as np
import pandas as pd
import scipy.special
import matplotlib.pyplot as plt
from econmltest6.EconML.econml.grf import CausalForest

np.random.seed(123)
n_samples = 1000

n_features = 10
n_treatments = 1

# true_te = lambda X: np.hstack([X[:, [0]]**2 + 1, np.ones((X.shape[0], n_treatments - 1))])
# true_te = lambda X: np.hstack([X[:, [0]]>0, np.ones((X.shape[0], n_treatments - 1))])
def true_te(X):
    return np.hstack([(X[:, [0]] > 0) * X[:, [0]],
                       np.ones((X.shape[0], n_treatments - 1)) * np.arange(1, n_treatments).reshape(1, -1)])
X = np.random.normal(0, 1, size=(n_samples, n_features))
T = np.random.normal(0, 1, size=(n_samples, n_treatments))
for t in range(n_treatments):
    T[:, t] = np.random.binomial(1, scipy.special.expit(X[:, 0]))
u = np.random.normal(0, .5, size=(n_samples, 1))
y = np.sum(true_te(X) * T, axis=1, keepdims=True) + u
X_test = X[:min(10, n_samples)].copy()
X_test[:, 0] = np.linspace(np.percentile(X[:, 0], 1), np.percentile(X[:, 0], 99), min(10, n_samples))


"""
### DGP example
 
         X1         (true_te  *   T)  +         u  =         y
7  0.176570         0.176570  *  0.0  +  0.551228  =  0.551228  . . . case 1) X_1 > 0, T=0
1 -0.883899        -0.000000  *  0.0  + -0.554930  = -0.554930  . . . case 2) X_1 < 0, T=0
9  0.722347         0.722347  *  1.0  +  0.500935  =  1.223282  . . . case 3) X_1 > 0, T=1
2 -2.368534        -0.000000  *  1.0  +  0.118847  =  0.118847  . . . case 4) X_1 < 0, T=1
"""
df = pd.DataFrame({'X1': np.squeeze(X[:10, 0]),
                   ' ': "     ",
                   '(true_te': np.squeeze(true_te(X)[:10, 0]), 
                   '*': "*",
                   'T)': np.squeeze(T[:10, 0]), 
                   '+': "+",
                   'u': np.squeeze(u[:10]),
                   '=': "=",
                   'y': np.squeeze(y[:10]),
                #    'pred': np.squeeze(point)
                   })
print("Data:")
print(df.iloc[:15,:])


grf = GRF(n_estimators=4,
          min_samples_leaf=5,
          max_depth=5,
          max_features=3,
          model_spec="y=a+bx+u",
          random_state=1235)

grf.fit(X=X, y=y, T=T)
point1 = grf.predict(X_test)


# EconML
cf = CausalForest(criterion='het', 
                  n_estimators=4, 
                  min_samples_leaf=5, 
                  max_depth=5,
                  max_features=3, 
                  honest=True,
                  random_state=1235)

cf.fit(X, T, y)
point2 = cf.predict(X_test)
point2 = np.squeeze(point2)

compare_df = pd.DataFrame({'scratch': point1, 
                           'econml': point2,
                           'abs. diff.': abs(point1-point2)})

print(compare_df)