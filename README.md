# Generalized Quantile Random Forest with Smoothed Estimating Equations

Code implementation of [Generalized Random Forest (Athey et al., 2019.)](https://arxiv.org/abs/1610.01271) with the application of smoothed estimating equations.

<br/>

## Setup

### Requirements

```bash
pip install numpy pandas matplotlib scipy joblib Cython
```

### Cloning a specific branch

```bash
# Cloning the GRF with a smoothed estimating equation
git clone --branch feature/quantile-see-forest --single-branch https://github.com/Rouxist/generalized-random-forest.git

# Cloning the GRF (fit with mean, predict quantiles)
git clone --branch feature/quantreg-forest --single-branch https://github.com/Rouxist/generalized-random-forest.git

# Cloning the GRF (fit and predict exact quantiles without approximation)
git clone --branch feature/quantile-forest --single-branch https://github.com/Rouxist/generalized-random-forest.git
```

### Cython code Compilation

```bash
python3 setup.py build_ext --inplace
```

### Running the model

```bash
# GRF with a smoothed estimating equation
python3 cython_test_qseegrf.py
```

<br/>

## Reference

1. [Athey, S., Tibshirani, J., & Wager, S. (2019). Generalized random forests.](https://arxiv.org/abs/1610.01271)
2. [https://github.com/py-why/EconML/](https://github.com/py-why/EconML/)
