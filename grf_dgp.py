import time
from grf_cython import GRF
from econml_original.EconML.econml.grf import RegressionForest
import numpy as np

## Model setup
T=480
N_ESTIMATORS=400
MAX_DEPTH = 5
HONEST = True
MAIN_SEED = 42

## DGP setup
N_FEATURES = 10
N_EXTRA_ERROR_TERM = 50
MU = 0
BETA_1 = 1
BETA_2 = 2
RHO = 0.5

def get_covariance_matrix(n: int, rho: float) -> np.ndarray:
    cov = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            cov[i, j] = rho ** abs(i - j)
    return cov

def normal_pdf(x, mu, sigma):
    return (1 / np.sqrt(2 * np.pi)) * np.exp((-(x-mu)**2)/(2*sigma**2))

## DGP
np.random.seed(MAIN_SEED)

### Generating X_t
cov_matrix = get_covariance_matrix(n=N_FEATURES, rho=0.7)
X = np.random.multivariate_normal(mean=np.zeros(N_FEATURES), 
                                  cov=cov_matrix, 
                                  size=T)

### Generating u_t
extended_T = T + N_EXTRA_ERROR_TERM           # Total number of T required including
                                              # initial extra timesteps that will later be truncated
u = np.zeros(extended_T)                      # Empty u_t array including u_0 = 0
epsilon = np.random.normal(0, 1, extended_T)  # \epsilon_t of each timestep

for t in range(1, extended_T):
    u[t] = RHO * u[t-1] + epsilon[t]

u = u[N_EXTRA_ERROR_TERM:]                    # Truncate initial N_EXTRA_ERROR_TERM error terms
                                              # to get rid of the effect of the u_0

### Treatment Variable
T1 = (X[:,0] < 0).astype(int)
T2 = (X[:,1] < 0).astype(int)

### Generating y_t
y = MU + BETA_1 * T1 + BETA_2 * T2 + u

tester_econml = RegressionForest(n_estimators=N_ESTIMATORS,
                     max_depth=MAX_DEPTH,
                     max_features=3,
                     min_samples_leaf=5,
                     honest=HONEST,
                     random_state=42)

tester_scratch = GRF(n_estimators=N_ESTIMATORS,
                     max_depth=MAX_DEPTH,
                     max_features=3,
                     min_samples_leaf=5,
                     honest=HONEST,
                     random_state=42)

print("\n\n================ EconML model =================")
start_time = time.time()
tester_econml.fit(X, y)
print(f"Model from econml(fit): {time.time() - start_time:.5f} sec")

start_time = time.time()
pred1 = tester_econml.predict(X[:T,:]).squeeze()
# print("predictions:\n", pred1[:10])
print(f"Model from econml(predict): {time.time() - start_time:.5f} sec")

print("\n\n================ Scratch model ================")
start_time = time.time()
tester_scratch.fit(X, y)
print(f"Model from scratch(fit): {time.time() - start_time:.5f} sec")

start_time = time.time()
pred2 = tester_scratch.predict(X[:T,:])
# print("predictions:\n", pred1[:10])
print(f"Model from scratch(predict): {time.time() - start_time:.5f} sec")

print("mae =", sum(abs(pred1-pred2))/len(X))


import pandas as pd
df_result = pd.DataFrame({'Predicted(EconML)': pred1, 'Predicted(From Scratch)': pred2})
df = pd.concat([pd.Series(y), df_result], axis=1)
df['Difference'] = abs(df['Predicted(EconML)'] - df['Predicted(From Scratch)'])

print(df)
# print("EconML:", df['Predicted(EconML)'].nunique())
# print("From Scratch:", df['Predicted(From Scratch)'].nunique())

df.to_csv("./test/cython_grf_test.csv")
