import numpy as np
import pandas as pd
from grf import GRF
from econmltest.EconML.econml.grf import RegressionForest
import matplotlib.pyplot as plt
from collections import Counter
import time
from tqdm import tqdm

import os

## Setup
### functional setup
FROM_SCRATCH_MODEL = True
FROM_SCRATCH_MODEL_P = False
ECONML_MODEL = False
SHOW_DATA = False
SHOW_T_VAR = False
SHOW_PLOT = True

### Experiment setup
T = 60
N_FEATURES = 10
N_ESTIMATORS = 4
N_ITER = 1
TARGET_NAME = 'y'

BLOCK_SIZE = 1 # 1 if when not using block sampling

### Parameters of time-series DGP
MU = 0
BETA_1 = 1
BETA_2 = 2
RHO = 0

N_EXTRA_ERROR_TERM = 50

### Seeds
MAIN_SEED = 142
BETA_1_TESTING_POINT_SEED,BETA_2_TESTING_POINT_SEED = 2558, 2114

### ETC
MAX_INT = np.iinfo(np.int32).max


## utils
def get_covariance_matrix(n, rho):
    cov_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            cov_matrix[i, j] = rho ** abs(i - j)
    
    return cov_matrix


## DGP
np.random.seed(MAIN_SEED)

### Generating X_t
cov_matrix = get_covariance_matrix(n=N_FEATURES, rho=0.7)
X = np.random.multivariate_normal(mean=np.zeros(N_FEATURES), cov=cov_matrix, size=T)

### Generating u_t
extended_T = T + N_EXTRA_ERROR_TERM           # Total number of T required including initial extra timesteps that will later be truncated
u = np.zeros(extended_T)                      # Empty u_t array including u_0 = 0
epsilon = np.random.normal(0, 1, extended_T)  # \epsilon_t of each timestep

for t in range(1, extended_T):
    u[t] = RHO * u[t-1] + epsilon[t]    

u = u[N_EXTRA_ERROR_TERM:]                    # Truncate initial N_EXTRA_ERROR_TERM error terms to get rid of the effect of the u_0

### Treatment Variable
T1 = (X[:,0] < 0).astype(int)
T2 = (X[:,1] < 0).astype(int)
df_T = pd.DataFrame({'T1': T1, 'T2': T2})

### Generating y_t
y = MU + BETA_1 * T1 + BETA_2 * T2 + u

### Aggregate all in DataFrame
y_vec = np.expand_dims(y, axis=1)
data = np.concatenate((y_vec, X), axis=1)
df = pd.DataFrame(data, columns=[TARGET_NAME]+['X' + str(i+1) for i in range(N_FEATURES)])

### Testing point
beta_1_test_x_random_state = np.random.RandomState(BETA_1_TESTING_POINT_SEED)
beta_1_test_x = beta_1_test_x_random_state.multivariate_normal(mean=np.zeros(N_FEATURES), cov=cov_matrix, size=1)
beta_1_test_x = pd.DataFrame(beta_1_test_x, columns=['X' + str(i+1) for i in range(N_FEATURES)])

beta_2_test_x_random_state = np.random.RandomState(BETA_2_TESTING_POINT_SEED)
beta_2_test_x = beta_2_test_x_random_state.multivariate_normal(mean=np.zeros(N_FEATURES), cov=cov_matrix, size=1)
beta_2_test_x = pd.DataFrame(beta_2_test_x, columns=['X' + str(i+1) for i in range(N_FEATURES)])


## Main simulation loop
if __name__ == "__main__":
    os.makedirs("./result/", exist_ok=True)
    time_now = time.localtime()
    time_now_date = time.strftime('%b', time_now) + " "  + time.strftime('%d', time_now) + " " + time.strftime('%Y', time_now)
    time_now_time = time.strftime('%H', time_now) + ":"  + time.strftime('%M', time_now) + ":"  + time.strftime('%S', time_now)
    time_now_zone = time.strftime('%Z', time_now)

    print()
    print("                                    Experiment Setup                                  ")
    print("======================================================================================")
    print("[General hyperparameters]")
    print("{:<30}: {:>10}".format("Main Seed", MAIN_SEED))
    print("{:<30}: {:>10}  {:<30}: {:>10}".format("Seed for beta1", BETA_1_TESTING_POINT_SEED, "Seed for beta2", BETA_2_TESTING_POINT_SEED))
    print("{:<30}: {:>10}".format("No. iteration", N_ITER))
    print()
    print("[DGP setup]")
    print("{:<30}: {:>10}  {:<30}: {:>10}".format("T", T, "rho", RHO))
    print("{:<30}: {:>10}  {:<30}: {:>10}".format("beta1", BETA_1, "beta2", BETA_2))
    print("{:<30}: {:>10}  {:<30}: {:>10}".format("No. initial u_t truncated", N_EXTRA_ERROR_TERM, "No. independent variable", N_FEATURES))
    print()
    print("[Random Forest setup]")
    print("{:<30}: {:>10}  {:<30}: {:>10}".format("No. iteration", N_ITER, "No. estiamtor", N_ESTIMATORS))
    print("{:<30}: {:>10}".format("Block size", BLOCK_SIZE))
    print("======================================================================================")
    print("\n")

    if SHOW_DATA:
        print("                                     Generated Data                                   ")
        print("======================================================================================")
        print("y_t = \\beta_1 T_1 + \\beta_2 T_2 + u_t")
        print("Whole dataset:\n", pd.DataFrame(df))
        print("u_t:\n", pd.Series(u, name="u_t").head())
        print("mean of y:", np.mean(y))
        print("\nbeta_1_test_x:\n", beta_1_test_x.iloc[:,:5])
        print("\nbeta_2_test_x:\n", beta_2_test_x.iloc[:,:5])
        print()
        print("======================================================================================")
        print("\n")
    if SHOW_T_VAR:
        print("                                   Treatment Variable                                 ")
        print("======================================================================================")
        print("Count of T1 (1 if < X_{1t})", Counter(T1))
        print("Count of T2 (1 if < X_{2t})", Counter(T2))
        print("First 5:\n", df_T.head())
        print()
        print("======================================================================================")
        print("\n")

    arr_seed = np.random.randint(low=0, high=MAX_INT, size=N_ITER)

    ### cals
    # mask_beta_1 = (T1 == 1) & (T2 == 0)
    # mask_beta_2 = (T1 == 0) & (T2 == 1)

    if ECONML_MODEL:
        arr_estimated_beta_1 = []
        arr_estimated_beta_2 = []

        start_time = time.time()
        
        for idx in tqdm(range(N_ITER)):
            # Model from EconML
            grf_econml = RegressionForest(n_estimators=N_ESTIMATORS, 
                                         honest=True, 
                                         min_samples_leaf=5, 
                                         max_depth=5, 
                                         max_features="auto", 
                                         random_state=arr_seed[idx])
            grf_econml.fit(X=X, y=y)

            beta_1_hat = grf_econml.predict(beta_1_test_x)
            beta_2_hat = grf_econml.predict(beta_2_test_x)
            
            arr_estimated_beta_1.append(beta_1_hat)
            arr_estimated_beta_2.append(beta_2_hat)

        time_taken = time.time() - start_time

        arr_estimated_beta_1 = [i.tolist()[0][0] for i in arr_estimated_beta_1]
        arr_estimated_beta_2 = [i.tolist()[0][0] for i in arr_estimated_beta_2]

        print("                                   Experiment Result                                  ")
        print("======================================================================================")
        print("{:<25}: {:>15}  {:<25}: {:>15}".format("Date", time_now_date,"Time", time_now_zone + " " + time_now_time))
        print("{:<25}: {:>15}  {:<25}: {:>11.2f} sec".format("Model", "Scratch", "Time taken", time_taken))
        print()
        print("{:<25}: {:>15.5}  {:<25}: {:>15.5}".format("Mean of beta1 hat", np.mean(arr_estimated_beta_1),"Std. dev. of beta1 hat", np.std(arr_estimated_beta_1)))
        print("{:<25}: {:>15.5}  {:<25}: {:>15.5}".format("Mean of beta2 hat", np.mean(arr_estimated_beta_2),"Std. dev. of beta2 hat", np.std(arr_estimated_beta_2)))
        print("======================================================================================")
        print("\n")
        
        if SHOW_PLOT:
            ### Histogram Plot
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            axes[0].hist(arr_estimated_beta_1, bins=50)
            axes[0].set_title('Histogram of List 1')
            axes[1].hist(arr_estimated_beta_2, bins=50)
            axes[1].set_title('Histogram of List 2')

            plt.tight_layout()
            plt.savefig(f"./result/block_{BLOCK_SIZE}__rho_{RHO}.png")


    if FROM_SCRATCH_MODEL:
        arr_estimated_beta_1 = []
        arr_estimated_beta_2 = []

        start_time = time.time()

        for idx in tqdm(range(N_ITER)):
            # Model from EconML
            grf_scratch = GRF(target=TARGET_NAME, 
                              n_estimators=N_ESTIMATORS, 
                              min_samples_leaf=5, 
                              max_depth=5, 
                              max_samples=0.45, 
                              honest=True, 
                              data_weight_ratio=0.5, 
                              block_size=BLOCK_SIZE,
                              random_state=arr_seed[idx])
            grf_scratch.fit(df)
            # grf.visualize(file_name='scratch_trees_visualized.txt')
            
            beta_1_hat = grf_scratch.predict(beta_1_test_x)
            beta_2_hat = grf_scratch.predict(beta_2_test_x)
            
            arr_estimated_beta_1.append(beta_1_hat)
            arr_estimated_beta_2.append(beta_2_hat)
        
        time_taken = time.time() - start_time
        
        arr_estimated_beta_1 = [i.tolist() for i in arr_estimated_beta_1]
        arr_estimated_beta_2 = [i.tolist() for i in arr_estimated_beta_2]

        print("                                   Experiment Result                                  ")
        print("======================================================================================")
        print("{:<25}: {:>15}  {:<25}: {:>15}".format("Date", time_now_date,"Time", time_now_zone + " " + time_now_time))
        print("{:<25}: {:>15}  {:<25}: {:>11.2f} sec".format("Model", "Scratch", "Time taken", time_taken))
        print()
        print("{:<25}: {:>15.5}  {:<25}: {:>15.5}".format("Mean of beta1 hat", np.mean(arr_estimated_beta_1),"Std. dev. of beta1 hat", np.std(arr_estimated_beta_1)))
        print("{:<25}: {:>15.5}  {:<25}: {:>15.5f}".format("Mean of beta2 hat", np.mean(arr_estimated_beta_2),"Std. dev. of beta2 hat", np.std(arr_estimated_beta_2)))
        print("======================================================================================")
        print("\n")
        
        if SHOW_PLOT:
            ### Histogram Plot
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            axes[0].hist(arr_estimated_beta_1, bins=50)
            axes[0].set_title('Histogram of List 1')
            axes[1].hist(arr_estimated_beta_2, bins=50)
            axes[1].set_title('Histogram of List 2')

            plt.tight_layout()
            plt.savefig(f"./result/block_{BLOCK_SIZE}__rho_{RHO}.png")
