import pandas as pd

from _gradient_tree import GradientTree
from _generalized_random_forest import GRF

MODEL = "GRF"

if __name__ == "__main__":
    df = pd.read_csv("./data/kc_house_data.csv") # Dataset; target variable is 'price'
    df = df[['price', 'bedrooms', 'sqft_lot15']]
    df_train = df.iloc[:20,:]                    # Dataset that includes both `split set` and `weight set`
    df_test = df.iloc[101:120,:]                 # Dataset from which one datapoint is sampled
    datapoint = df_test.iloc[6]                  # One data point to make estimation

    print(f"[Train dataset]\n\n{df_train}\n\n")
    print(f"[Datapoint to estimate]\n\n{datapoint}\n\n")

    if MODEL == "Tree": # Gradient Tree    
        tree = GradientTree()
        tree.fit(df_train, "price")
        tree.visualize(file_name='tree_test.txt')
        estimate = tree.predict(datapoint)
        
        print(f"Estimate = {estimate}")

    elif MODEL == "GRF": # Generalized Random Forest
        grf = GRF(target="price", n_estimators=1, data_weight_ratio=0.7)
        grf.fit(df_train)
        # grf.visualize(file_name='trees_visualized.txt')
        estimate = grf.predict(datapoint)
        
        print(f"[Subset for calculating weight]\n\n{grf.data_weight}\n\n")
        print(f"[α(x)]\n\n{grf.alpha}\n\n")
        print(f"[Result]\n\ntheta_hat(x) = {estimate}")
    