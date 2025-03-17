import time
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from grf_cython import GRF

## Model setup
N_ESTIMATORS=400
MAX_DEPTH = 5
HONEST = False
MAIN_SEED = 443

## DGP setup
N=2000
N_FEATURES = 40
MU = 0
BETA_1 = 0.8

## DGP
np.random.seed(MAIN_SEED)

# x1 = np.linspace(-1, 1, 200)
# x1 = np.expand_dims(x1, axis=1)
# x_rem = np.zeros((200,39))
# X_test = np.hstack((x1, x_rem))
# print(X_test[:2,:])

### Generating X_{ij}
X = np.random.uniform(low=-1, high=1, size=(N, N_FEATURES))

"""
## Mean shift
### Generating u_i
u = np.random.normal(0, 1, N)

### Treatment Variable
T1 = (X[:,0] > 0).astype(int)

### Generating y_i
y = MU + BETA_1 * T1 + u

# quantiles = np.quantile(y, [0.1, 0.5, 0.9])

# Print results
# print(f"10% quantile: {quantiles[0]}")
# print(f"50% quantile (median): {quantiles[1]}")
# print(f"90% quantile: {quantiles[2]}")

# plt.step([-1, 0, 1], [quantiles[0], quantiles[0], quantiles[0]], where='post', label='truth')
# plt.step([-1, 0, 1], [quantiles[1], quantiles[1], quantiles[1]], where='post', label='truth')
# plt.step([-1, 0, 1], [quantiles[2], quantiles[2], quantiles[2]], where='post', label='truth')
# plt.show()
"""


## Scale shift
### Define standad deviation
sigma = (1 + (X[:,0] > 0))

## Generate Y from the normal distribution with scale shift
y = np.random.normal(0, sigma, N)

df = pd.DataFrame({'y': y, 'X_0': X[:,0], 'X_1': X[:,1], 'X_2': X[:,2], 'X_3': X[:,3]})
print("mean of y|X_{i0}>0 :", df[df['X_0']>0]['y'].mean())
print("std of y|X_{i0}>0 :", df[df['X_0']>0]['y'].std())
print("mean of y|X_{i0}<=0:", df[df['X_0']<=0]['y'].mean())
print("std of y|X_{i0}<=0:", df[df['X_0']<=0]['y'].std())

print("\n=========Experiment Setup=========")
print(f"N_ESTIMATORS={N_ESTIMATORS}")
print(f"N={N}")
print(f"HONEST={HONEST}")
print(f"MAIN_SEED={MAIN_SEED}")
print("Goal: to test seed=443 at N=2000, trees=200")


## Model 1 ############################################################################################################

model = GRF(n_estimators=N_ESTIMATORS,
            max_depth=MAX_DEPTH,
            max_features=math.floor(N_FEATURES * 0.5),
            min_samples_leaf=5,
            honest=HONEST,
            quantile=0.5,
            random_state=42)


print("\n================ Scratch model ================")
start_time = time.time()
model.fit(X, y)
print(f"Model from scratch(fit): {time.time() - start_time:.5f} sec")

start_time = time.time()
pred_1 = model.predict(X)
print(f"Model from scratch(predict): {time.time() - start_time:.5f} sec")


# sorted_list1, sorted_list2 = zip(*sorted(zip(X[:,0], pred)))

# sorted_list1 = list(sorted_list1)
# sorted_list2 = list(sorted_list2)

# plt.scatter(sorted_list1, sorted_list2)

plt.scatter(X[:,0], pred_1, c='black', s=10, alpha=0.5)

#####################################################################################################################

## Model 2 ############################################################################################################

model = GRF(n_estimators=N_ESTIMATORS,
            max_depth=MAX_DEPTH,
            max_features=math.floor(N_FEATURES * 0.5),
            min_samples_leaf=5,
            honest=HONEST,
            quantile=0.1,
            random_state=42)


print("\n================ Scratch model ================")
start_time = time.time()
model.fit(X, y)
print(f"Model from scratch(fit): {time.time() - start_time:.5f} sec")

start_time = time.time()
pred_5 = model.predict(X)
print(f"Model from scratch(predict): {time.time() - start_time:.5f} sec")

plt.scatter(X[:,0], pred_5, c='black', s=10, alpha=0.5)

#####################################################################################################################

## Model 3 ############################################################################################################

model = GRF(n_estimators=N_ESTIMATORS,
            max_depth=MAX_DEPTH,
            max_features=math.floor(N_FEATURES * 0.5),
            min_samples_leaf=5,
            honest=HONEST,
            quantile=0.9,
            random_state=42)


print("\n================ Scratch model ================")
start_time = time.time()
model.fit(X, y)
print(f"Model from scratch(fit): {time.time() - start_time:.5f} sec")

start_time = time.time()
pred_9 = model.predict(X)
print(f"Model from scratch(predict): {time.time() - start_time:.5f} sec")

plt.scatter(X[:,0], pred_9, c='black', s=10, alpha=0.5)

#####################################################################################################################
# df = pd.DataFrame({'pred_0.1':pred_1, 'pred_0.5':pred_5, 'pred_0.9':pred_9, 'y': y, 'X_0': X[:,0], 'u': u})
# print("Data:\n",df.loc[:20])

### Mean shift
# plt.step([-1, 0, 1], [0, 0.8, 0.8], where='post', label='truth') # Create step plot
# plt.step([-1, 0, 1], [-1.28, -0.48, -0.48], where='post', label='truth') # Create step plot
# plt.step([-1, 0, 1], [1.28, 2.08, 2.08], where='post', label='truth') # Create step plot

### Scale shift 
#### std max2
plt.step([-1, 0, 1], [0, 0, 0], where='post', label='truth') # Create step plot
plt.step([-1, 0, 1], [-1.28, -2.5631, -2.5631], where='post', label='truth') # Create step plot
plt.step([-1, 0, 1], [1.28, 2.5631, 2.5631], where='post', label='truth') # Create step plot

#### std max3
# plt.step([-1, 0, 1], [0, 0, 0], where='post', label='truth') # Create step plot
# plt.step([-1, 0, 1], [-1.28, -3.84, -3.84], where='post', label='truth') # Create step plot
# plt.step([-1, 0, 1], [1.28, 3.84, 3.84], where='post', label='truth') # Create step plot

plt.tight_layout()
plt.savefig(f"../results_qrf/N_{N}__trees_{N_ESTIMATORS}__features_{N_FEATURES}__seed_{MAIN_SEED}__std_max2__md5__nf50__scale_shift.png")
