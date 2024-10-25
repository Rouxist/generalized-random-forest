import numpy as np
import pandas as pd
from numpy.linalg import inv
from scipy.optimize import fsolve

class GRFCriterion:
    def __init__(self, honest:bool=True) -> None:
        pass

    def get_theta_p_hat(self, y_parent) -> float:
        
        def sum_moment_condition(theta) -> float:  # Eq (4)
            # Depends on Regression equation
            return np.mean((y_parent-theta))

        theta_0 = np.median(y_parent) # initial value
        res = fsolve(sum_moment_condition, theta_0)
        return res[0]
    
    def get_psi(self, y_c: list, theta_hat_p: float) -> np.ndarray:
        # Depends on Regression equation
        return np.array([np.mean([y_c - theta_hat_p])])
    
    def get_psi_vec(self, y_c: list, theta_hat_p: float) -> np.ndarray:
        # Depends on Regression equation
        return np.array(y_c) - theta_hat_p
    
    def get_xi(self) -> np.ndarray:
        # Depends on Regression equation
        return np.array([1])
    
    def get_a_p(self, y_c: list, theta_hat_p: float) -> np.ndarray:
        # Depends on Regression equation
        return np.array([[-1]])
    
    def get_inv_a_p(self, a_p: np.ndarray) -> np.ndarray:
        if len(a_p) == 1:
            return 1/a_p
        else:
            return inv(a_p)

    def get_proxy_delta_tilde(self, y_left:np.ndarray, y_right:np.ndarray) -> float:
        y_parent=np.append(y_left, y_right)
        # print("wow, given y_parent =",y_parent)
        # print("wow, given y_left =",y_left)
        # print("wow, given y_right =",y_right)
        theta_p_hat = self.get_theta_p_hat(y_parent=y_parent) # for y_i = b + u_i, it equals to mean of y_parent

        xi = self.get_xi()
        a_p = self.get_a_p(y_parent, theta_p_hat)
        
        inv_a_p = self.get_inv_a_p(a_p)

        # Partitioning scheme in algorithmical perspective
        psi_left_vec = self.get_psi_vec(y_left, theta_p_hat)
        psi_right_vec = self.get_psi_vec(y_right, theta_p_hat)

        rho_left_vec = -1 * (xi.T @ inv_a_p @ psi_left_vec.reshape(-1, 1, 1))
        rho_right_vec = -1 * (xi.T @ inv_a_p @ psi_right_vec.reshape(-1, 1, 1))

        rho_squared_sum_left = ((np.sum(rho_left_vec)) ** 2 ) / len(y_left)
        rho_squared_sum_right = ((np.sum(rho_right_vec)) ** 2) / len(y_right)

        return rho_squared_sum_left + rho_squared_sum_right