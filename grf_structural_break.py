import argparse
import time
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from grf_cython import GRF

def main():
    # Parse arguments
    parser = argparse.ArgumentParser()

    ## Simulation setup
    parser.add_argument('--seed', type=int, default=43)
    parser.add_argument('--is_simple_test', dest='is_simple_test', action='store_true')

    ## Data generation setup
    parser.add_argument('--shift_type', default='mean', help='mean/scale')
    parser.add_argument('--n_obs', type=int, default=1000, help='# of observations')
    parser.add_argument('--n_feats', type=int, default=40, help='# of features')
    parser.add_argument('--mu', type=float, default=0., help='True mu')
    parser.add_argument('--beta_1', type=float, default=0.8, help='True beta_1')

    ## Model setup
    parser.add_argument('--n_estimators', type=int, default=400)
    parser.add_argument('--max_depth', type=int, default=5)
    parser.add_argument('--portion_features', type=float, default=0.5, help='Proportion of features used when splitting')
    parser.add_argument('--honest', dest='honest', action='store_true')
    
    # Save arguments to variables
    args = parser.parse_args()
    shift_type = args.shift_type
    is_simple_test = args.is_simple_test

    MAIN_SEED = args.seed
    N = args.n_obs
    N_FEATURES = args.n_feats
    MU = args.mu
    BETA_1 = args.beta_1
    N_ESTIMATORS = args.n_estimators
    MAX_DEPTH = args.max_depth
    PORTION_FEATURES = args.portion_features
    HONEST = args.honest

    ## Additional hyperparameters
    H=0.1

    # Generate data
    np.random.seed(MAIN_SEED)

    X = np.random.uniform(low=-1, high=1, size=(N, N_FEATURES))
    u = np.random.normal(0, 1, N)
    T = (X[:,0] > 0).astype(int)

    if shift_type=="mean":
        ### Mean shift
        y = MU + BETA_1 * T + u
    elif shift_type=="scale":
        sigma = (1 + (X[:,0] > 0))
        y = np.random.normal(0, sigma, N)

    ## Simplified test data
    x1 = np.linspace(-1, 1, 200)
    x1 = np.expand_dims(x1, axis=1)
    x_rem = np.zeros((200,N_FEATURES-1))
    X_test = np.hstack((x1, x_rem))

    # Main simulation part
    print("\n=========================== Experiment Setup ===========================")
    print(f"N_ESTIMATORS={N_ESTIMATORS}  |  N={N}  |  HONEST={HONEST}  |  MAIN_SEED={MAIN_SEED}")


    pred_list = []
    for quantile in [0.5, 0.1, 0.9]:
        model = GRF(n_estimators=N_ESTIMATORS,
                max_depth=MAX_DEPTH,
                max_features=math.floor(N_FEATURES * PORTION_FEATURES),
                min_samples_leaf=5,
                honest=HONEST,
                quantile=quantile,
                h=H,
                random_state=42)


        print("\n================ Scratch model ================")
        start_time = time.time()
        model.fit(X, y)
        print(f"Model from scratch(fit): {time.time() - start_time:.5f} sec")

        start_time = time.time()
        
        if is_simple_test:
            pred = model.predict(X_test)
        else:
            pred = model.predict(X)

        pred_list.append(pred)
        print(f"Model from scratch(predict): {time.time() - start_time:.5f} sec")

        if is_simple_test:
            plt.scatter(X_test[:,0], pred, c='black', s=10, alpha=0.5)
        else:
            plt.scatter(X[:,0], pred, c='black', s=10, alpha=0.5)


    # Draw true step plot
    if shift_type=="mean":
        ### Mean shift
        plt.step([-1, 0, 1], [0, 0.8, 0.8], where='post', label='truth')
        plt.step([-1, 0, 1], [-1.28, -0.48, -0.48], where='post', label='truth')
        plt.step([-1, 0, 1], [1.28, 2.08, 2.08], where='post', label='truth')
    elif shift_type=="scale":
        ### Scale shift 
        #### std max2
        plt.step([-1, 0, 1], [0, 0, 0], where='post', label='truth')
        plt.step([-1, 0, 1], [-1.28, -2.5631, -2.5631], where='post', label='truth')
        plt.step([-1, 0, 1], [1.28, 2.5631, 2.5631], where='post', label='truth')

    df_result = pd.DataFrame({'Predicted(0.1)': pred_list[1], 'Predicted(0.5)': pred_list[0], 'Predicted(0.9)': pred_list[2]})
    print()
    print(df_result)

    plt.tight_layout()
    plt.savefig(f"../results_qseegrf/N_{N}__trees_{N_ESTIMATORS}__maxfeats_{int(PORTION_FEATURES * 100):03d}__seed_{MAIN_SEED}__shifttype_{shift_type}__h_{int(H * 100):03d}.png")


if __name__ == '__main__':
    main()
