# Example is from: https://github.com/py-why/EconML/blob/main/notebooks/Generalized%20Random%20Forests.ipynb

from grf_cython import GRF
import numpy as np
import pandas as pd
import scipy.special
import matplotlib.pyplot as plt

np.random.seed(123)
n_samples = 50

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
# print("X:\n", X[:6,:3])
# print("True TE:\n", (X[:6,[0]]>0) * X[:6,[0]])
# print("True TE:\n", true_te(X)[:6,:])
# print("u:\n", u[:6])
# print("y:\n", y[:6])
# print("T:\n", T[:6,:])


# TEST_TYPE = "y=b+u"
TEST_TYPE = "y=a+bx+u"

if TEST_TYPE == "y=b+u":
    grf = GRF(n_estimators=4,
            min_samples_leaf=5,
            max_depth=5,
            max_features=3,
            model_spec="y=b+u",
            random_state=1235)

    grf.fit(X=X, y=y)

elif TEST_TYPE == "y=a+bx+u":
    grf = GRF(n_estimators=4,
            min_samples_leaf=5,
            max_depth=5,
            max_features=3,
            model_spec="y=a+bx+u",
            random_state=1235)

    grf.fit(X=X, y=y, T=T)


point = grf.predict(X_test)

df = pd.DataFrame({'X1': np.squeeze(X[:min(10, n_samples),0]),
                   'X2': np.squeeze(X[:min(10, n_samples),1]),
                   'X3': np.squeeze(X[:min(10, n_samples),2]),
                   'X4': np.squeeze(X[:min(10, n_samples),3]),
                   'T': np.squeeze(T[:min(10, n_samples),0]), 
                   'y': np.squeeze(y[:min(10, n_samples)]), 
                   'pred': np.squeeze(point)})
print(df.iloc[:5,:])
