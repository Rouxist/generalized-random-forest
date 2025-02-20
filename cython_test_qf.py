import time
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from grf_cython import GRF

## Model setup
N_ESTIMATORS=100
MAX_DEPTH = 5
HONEST = True
MAIN_SEED = 43

## DGP setup
N=2000
N_FEATURES = 20
MU = 0
BETA_1 = 0.8

## DGP
np.random.seed(MAIN_SEED)

### Generating X_{ij}
X = np.random.uniform(low=-1, high=1, size=(N, N_FEATURES))

### Generating u_i
u = np.random.normal(0, 1, N)

### Treatment Variable
T1 = (X[:,0] > 0).astype(int)

### Generating y_i
y = MU + BETA_1 * T1 + u


df = pd.DataFrame({'y': y, 'X_0': X[:,0], 'X_1': X[:,1], 'X_2': X[:,2], 'X_3': X[:,3], 'u': u})
print("mean of y|X_{i0}>0 :", df[df['X_0']>0]['y'].mean())
print("mean of y|X_{i0}<=0:", df[df['X_0']<=0]['y'].mean())


## Model 1 ############################################################################################################

model = GRF(n_estimators=N_ESTIMATORS,
            max_depth=MAX_DEPTH,
            max_features=math.floor(N_FEATURES * 0.3),
            min_samples_leaf=5,
            honest=HONEST,
            quantile=0.5,
            random_state=42)


print("\n\n================ Scratch model ================")
start_time = time.time()
model.fit(X, y)
print(f"Model from scratch(fit): {time.time() - start_time:.5f} sec")

start_time = time.time()
pred_1 = model.predict(X)
print(f"Model from scratch(predict): {time.time() - start_time:.5f} sec")

plt.step([-1, 0, 1], [0, 0.8, 0.8], where='post', label='truth') # Create step plot
plt.scatter(X[:,0], pred_1, c='black', s=10)

#####################################################################################################################

## Model 2 ############################################################################################################

model = GRF(n_estimators=N_ESTIMATORS,
            max_depth=MAX_DEPTH,
            max_features=math.floor(N_FEATURES * 0.3),
            min_samples_leaf=5,
            honest=HONEST,
            quantile=0.1,
            random_state=42)


print("\n\n================ Scratch model ================")
start_time = time.time()
model.fit(X, y)
print(f"Model from scratch(fit): {time.time() - start_time:.5f} sec")

start_time = time.time()
pred_5 = model.predict(X)
print(f"Model from scratch(predict): {time.time() - start_time:.5f} sec")

plt.step([-1, 0, 1], [-1.28, -0.48, -0.48], where='post', label='truth') # Create step plot
plt.scatter(X[:,0], pred_5, c='black', s=10)

#####################################################################################################################

## Model 3 ############################################################################################################

model = GRF(n_estimators=N_ESTIMATORS,
            max_depth=MAX_DEPTH,
            max_features=math.floor(N_FEATURES * 0.3),
            min_samples_leaf=5,
            honest=HONEST,
            quantile=0.9,
            random_state=42)


print("\n\n================ Scratch model ================")
start_time = time.time()
model.fit(X, y)
print(f"Model from scratch(fit): {time.time() - start_time:.5f} sec")

start_time = time.time()
pred_9 = model.predict(X)
print(f"Model from scratch(predict): {time.time() - start_time:.5f} sec")

plt.step([-1, 0, 1], [1.28, 2.08, 2.08], where='post', label='truth') # Create step plot
plt.scatter(X[:,0], pred_9, c='black', s=10)

#####################################################################################################################

df = pd.DataFrame({'pred_0.1':pred_1, 'pred_0.5':pred_5, 'pred_0.9':pred_9, 'y': y, 'X_0': X[:,0], 'u': u})
print("Data:\n",df.loc[:20])

plt.tight_layout()
plt.savefig(f"../results_qf/N_{N}__trees_{N_ESTIMATORS}__features_{N_FEATURES}__seed_{MAIN_SEED}__full.png")
