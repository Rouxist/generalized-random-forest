import argparse
import time
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from grf_cython import GRF

MAX_INT = np.iinfo(np.int32).max

def simulation(is_print, seed, n_obs, n_estimators, h, shift_type, portion_features, max_depth=5, honest=True):

    shift_type = shift_type

    MAIN_SEED = seed
    N = n_obs
    N_FEATURES = 40
    MU = 0
    BETA_1 = 0.8
    N_ESTIMATORS = n_estimators
    MAX_DEPTH = max_depth
    PORTION_FEATURES = portion_features
    HONEST = honest
    H=h

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
    x1 = np.linspace(-1, 1, 2)
    x1 = np.expand_dims(x1, axis=1)
    x_rem = np.zeros((2,39))
    X_test = np.hstack((x1, x_rem))

    # Main simulation part

    pred_list = []
    for quantile in [0.1, 0.5, 0.9]:
        model = GRF(n_estimators=N_ESTIMATORS,
                max_depth=MAX_DEPTH,
                max_features=math.floor(N_FEATURES * PORTION_FEATURES),
                min_samples_leaf=5,
                honest=HONEST,
                quantile=quantile,
                h=H,
                random_state=42)


        print(f"\n================ Quantile {quantile} ================")
        start_time = time.time()
        model.fit(X, y)
        if is_print:
            print(f"Model from scratch(fit): {time.time() - start_time:.5f} sec")

        start_time = time.time()
        pred = model.predict(X_test)
        if is_print:
            print(f"Model from scratch(predict): {time.time() - start_time:.5f} sec")

        pred_list.append(pred)
    
    return pred_list[0][0], pred_list[0][1], pred_list[1][0], pred_list[1][1], pred_list[2][0], pred_list[2][1]


def main():
    # Parse arguments
    parser = argparse.ArgumentParser()

    ## Simulation setup
    parser.add_argument('--n_simulations', type=int)

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
    parser.add_argument('--h', type=float, help='bandwidth parameter')
    
    # Save arguments to variables
    args = parser.parse_args()

    N_SIMULATIONS = args.n_simulations

    np.random.seed(0)
    SEED_LIST = np.random.randint(0, MAX_INT, size=N_SIMULATIONS)


    # Main simulation loop
    df_rows = []
    
    print("=========================== General Info ===========================")
    print(f"N={args.n_obs}  |  N_ESTIMATORS={args.n_estimators}  |  max_features={args.portion_features}   |   h={args.h}")
    print()

    for idx_sim in range(N_SIMULATIONS):
        print(f"\n\n=========================== Simulation # {idx_sim+1} ===========================")
        print(f"MAIN_SEED={SEED_LIST[idx_sim]}   |   shift type={args.shift_type}")
        res = simulation(idx_sim==N_SIMULATIONS-1, SEED_LIST[idx_sim], args.n_obs, args.n_estimators, args.h, args.shift_type, args.portion_features)
        df_rows.append({"10th_left": res[0], "10th_right": res[1], "50th_left": res[2], "50th_right": res[3], "90th_left": res[4], "90th_right": res[5]})
    
    df_result = pd.DataFrame(df_rows)

    print("Result:")
    print(df_result)

    
    # Draw true step plot
    if args.shift_type=="mean":
        ### Mean shift
        plt.step([-1, 0, 1], [-1.28, -0.48, -0.48], where='post', label='truth', color = "red")
        plt.step([-1, 0, 1], [0, 0.8, 0.8], where='post', label='truth', color = "green")
        plt.step([-1, 0, 1], [1.28, 2.08, 2.08], where='post', label='truth', color = "blue")
    elif args.shift_type=="scale":
        ### Scale shift 
        #### std max2
        plt.step([-1, 0, 1], [-1.28, -2.5631, -2.5631], where='post', label='truth', color = "red")
        plt.step([-1, 0, 1], [0, 0, 0], where='post', label='truth', color = "green")
        plt.step([-1, 0, 1], [1.28, 2.5631, 2.5631], where='post', label='truth', color = "blue")
    

    # Draw predicted dots
    cols = df_result.columns

    x_values = []
    y_values = []
    colors = []

    for i, col in enumerate(cols):
        x = -1 if i % 2 == 0 else 1   # 0,2,4 → -1 ; 1,3,5 → +1
        y = df_result[col].values

        # assign group color by column pair
        if i < 2:              # col 0,1 → group 1
            c = "red"
        elif i < 4:            # col 2,3 → group 2
            c = "green"
        else:                  # col 4,5 → group 3
            c = "blue"
        
        x_values.extend([x] * len(y))
        y_values.extend(y)
        colors.extend([c] * len(y))  # same color for all points in this column

    plt.scatter(x_values, y_values, c=colors, alpha=0.25)
    plt.tight_layout()
    plt.savefig(f"../results_qseegrf/N_SIM_{args.n_simulations}__N_{args.n_obs}__trees_{args.n_estimators}__maxfeats_{int(args.portion_features * 100):03d}__shifttype_{args.shift_type}__h_{int(args.h * 100):03d}.png")
    df_result.to_csv(f"../results_qseegrf/N_SIM_{args.n_simulations}__N_{args.n_obs}__trees_{args.n_estimators}__maxfeats_{int(args.portion_features * 100):03d}__shifttype_{args.shift_type}__h_{int(args.h * 100):03d}.csv")
    print(f"\nName of exported plot and csv : N_SIM_{args.n_simulations}__N_{args.n_obs}__trees_{args.n_estimators}__maxfeats_{int(args.portion_features * 100):03d}__shifttype_{args.shift_type}__h_{int(args.h * 100):03d}")

if __name__ == '__main__':
    main()
