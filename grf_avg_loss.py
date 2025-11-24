import math
import time
import numpy as np
import pandas as pd
import requests
from io import StringIO
from grf_cython import GRF as GRF # can be imported as GRF, RF, SEEGRF, ...

MAX_INT = np.iinfo(np.int32).max

## Data setup
data_url = 'https://lib.stat.cmu.edu/datasets/boston'
response = requests.get(data_url)
data = StringIO(response.text)

### Skip header and parse the data
raw_df = pd.read_csv(data, skiprows=22, header=None, sep='\s+')
X = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
y = raw_df.values[1::2, 2]
N_FEATURES = X.shape[1]

seed = 42


## Model setup
N_ESTIMATORS = 400
MAX_DEPTH = 5
MAIN_SEED = 43
TAU = 0.5


## CV setup
CV_FOLDS = 5
n_samples = len(X)

### Compute fold sizes
fold_sizes = np.full(CV_FOLDS, n_samples // CV_FOLDS, dtype=int)
fold_sizes[:n_samples % CV_FOLDS] += 1  # Distribute the remainder


## Simulation setup
N_SIMULATION = 3
np.random.seed(seed)  # For reproducibility

SEED_LIST = np.random.randint(0, MAX_INT, size=N_SIMULATION)


## Main Loop ##########################################################################################################

avg_loss_list = []

for n_sim in range(N_SIMULATION):
    print(f"Simulation # {n_sim+1} starts")

    ## Bootstraping with replacement
    rng = np.random.default_rng(SEED_LIST[n_sim])

    ### Sample indices with replacement
    bootstrap_indices = rng.choice(n_samples, size=n_samples, replace=True)

    ### Sample X and y using the same indices
    X_bootstrap = X[bootstrap_indices]
    y_bootstrap = y[bootstrap_indices]

    ### Shuffle the indices
    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    estimate_list = []

    grf_model = GRF(n_estimators=N_ESTIMATORS,
                max_depth=MAX_DEPTH,
                max_features=math.floor(N_FEATURES * 0.3),
                min_samples_leaf=5,
                honest=False,
                quantile=TAU,
                random_state=42)

    ### Cross Validation loop
    current = 0
    for fold in range(CV_FOLDS):
        start_time = time.time()

        start, stop = current, current + fold_sizes[fold]
        test_idx = indices[start:stop]
        train_idx = np.concatenate([indices[:start], indices[stop:]])

        X_train, y_train = X_bootstrap[train_idx], y_bootstrap[train_idx]
        X_test, y_test = X_bootstrap[test_idx], y_bootstrap[test_idx]

        grf_model.fit(X_train, y_train)
        pred_grf = grf_model.predict(X_test)
        estimate_list.extend(pred_grf)

        current = stop

        print(f"Time taken for fold {fold+1}: {time.time() - start_time:.5f} sec")

    ### Example DataFrame
    df_result = pd.DataFrame({
        'y': y_bootstrap,
        'theta_hat': estimate_list
    })

    ### Compute the 'loss' column
    df_result['loss'] = np.where(df_result['y'] >= df_result['theta_hat'],
                        (df_result['y'] - df_result['theta_hat']) * TAU,
                        (df_result['y'] - df_result['theta_hat']) * (TAU-1))

    df_result.to_csv(f"./result_simulated/result_{n_sim+1}.csv")

    ### Store average of loss

    avg_loss_list.append(df_result['loss'].mean())

    print("\n\n")

#####################################################################################################################

df_avg_loss_result = pd.DataFrame(avg_loss_list, columns=['average loss'])
df_avg_loss_result.to_csv('./result_simulated/grf_avg_loss_result.csv', index=False)

print("\n\nSimulation is done")
