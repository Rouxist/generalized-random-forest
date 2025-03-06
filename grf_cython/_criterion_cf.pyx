from libc.stdlib cimport calloc
from libc.stdlib cimport free
from libc.string cimport memcpy
from libc.string cimport memset

import numpy as np
cimport numpy as np
np.import_array()

cdef double INFINITY = np.inf

from numpy.linalg import inv
from functools import partial
from scipy.optimize import fsolve

from ._utils cimport safe_realloc

def mean_moment_conditions(params, y, T):
    # Moment condition for regression equation y_i = \nu + \theta T_i + u_i
    cdef double theta = params[0]
    cdef double nu = params[1]

    cdef double moment1 = np.mean(T * (y - nu - theta * T))
    cdef double moment2 = np.mean(y - nu - theta * T)

    return np.array([moment1, moment2])

cdef class GRFCriterionCF:
    def __cinit__(self, SIZE_t n_samples, UINT32_t random_state):
        self.n_samples = n_samples
        self.random_state = random_state

        self.nu_p_hat = NULL
        self.theta_p_hat = NULL
        self.sum_left = NULL   # Does not have to be array when n_outputs=1
        self.sum_right = NULL  # Does not have to be array when n_outputs=1
        self.y_parent = NULL
        
        self.nu_p_hat = <double*> calloc(1, sizeof(double))
        self.theta_p_hat = <double*> calloc(1, sizeof(double))
        self.sum_left = <double*> calloc(1, sizeof(double))
        self.sum_right = <double*> calloc(1, sizeof(double))

    cdef int init(self, const DTYPE_t[:, :] X, const DOUBLE_t[:, ::1] y, 
                  const DOUBLE_t[:, ::1] T, 
                  SIZE_t* samples) nogil except -1:
        """
        Stores data to be used.
        """
        self.X = X
        self.y = y
        self.T = T
        self.samples = samples

        return 0
    
    cdef int node_reset(self, SIZE_t start, SIZE_t end):
        """
        Calculates \hat{\theta_P} of node. Will later be used when calculating \tilde{\Delta}.
        GIL makes this function slow.
        """
        
        cdef SIZE_t* samples = self.samples
        cdef double* nu_p_hat = self.nu_p_hat
        cdef double* theta_p_hat = self.theta_p_hat
        self.start = start
        self.end = end
        self.n_node_samples = end - start

        cdef np.ndarray[np.float64_t, ndim=1] y_parent_new_ndarray = np.array([])
        cdef np.ndarray[np.float64_t, ndim=1] T_parent_new_ndarray = np.array([])

        for p in range(start, end):
            i = samples[p]
            y_parent_new_ndarray = np.append(y_parent_new_ndarray, self.y[i,0])
            T_parent_new_ndarray = np.append(T_parent_new_ndarray, self.T[i,0])

        # Calculate mean of T for calculation later
        self.T_bar = np.mean(T_parent_new_ndarray)

        # Initial guess
        cdef np.ndarray[np.float64_t, ndim=1] theta_0 = np.array([0.0, np.mean(y_parent_new_ndarray)])
        
        # Solve the equation using fsolve
        result = fsolve(mean_moment_conditions, theta_0, args=(y_parent_new_ndarray, T_parent_new_ndarray))
        theta_p_hat[0] = result[0]
        nu_p_hat[0] = result[1]

        # Reset to pos=start
        self.reset()
        return 0

    cdef int reset(self) nogil except -1:
        cdef SIZE_t n_bytes = sizeof(double)
        memset(self.sum_left, 0, n_bytes)
        memset(self.sum_right, 0, n_bytes)

        self.n_left = 0.0
        self.n_right = self.n_node_samples
        self.pos = self.start
        return 0

    cdef int update(self, SIZE_t new_pos):
        """
        Calculates sum_{C} rho_i.

        Comment:

        The original code from scikit-learn 
        
        1. Iterates only one child node with less # of datapoints.
        2. When explore different split point in same feature, it does not fully iterate the node again.
        
        Rather, this function iterates both nodes every time split point changes.
        """

        cdef SIZE_t start = self.start
        cdef SIZE_t pos = self.pos
        cdef SIZE_t end = self.end
        cdef double* sum_left = self.sum_left
        cdef double* sum_right = self.sum_right
        cdef double* nu_p_hat = self.nu_p_hat
        cdef double* theta_p_hat = self.theta_p_hat
        
        cdef SIZE_t* samples = self.samples

        cdef double sum_rho_left = 0
        cdef double sum_rho_right = 0

        self.n_left = 0.0
        self.n_right = 0.0

        # Elements of moment condition
        cdef np.ndarray[np.float64_t, ndim=1] xi = np.array([1.0, 0.0])
        cdef np.ndarray[np.float64_t, ndim=1] psi
        cdef np.ndarray[np.float64_t, ndim=2] A_p = np.array([[-1*self.T_bar, -1*self.T_bar],
                                                              [-1*self.T_bar, -1]])
        cdef np.ndarray[np.float64_t, ndim=2] inv_A_p = inv(A_p)

        # \sum{\rho} of left child node
        for p in range(start, new_pos):
            i = samples[p]
            psi = np.array([self.T[i,0] * (self.y[i,0]-nu_p_hat[0]-(theta_p_hat[0]*self.T[i,0])),
                            self.y[i,0]-nu_p_hat[0]-(theta_p_hat[0]*self.T[i,0])])
            sum_rho_left += -1 * (xi.T @ inv_A_p @ psi)
            self.n_left += 1.0
        self.sum_left[0] = sum_rho_left

        # \sum{\rho} of right child node
        for p in range(new_pos, end):
            i = samples[p]
            psi = np.array([self.T[i,0] * (self.y[i,0]-nu_p_hat[0]-(theta_p_hat[0]*self.T[i,0])),
                            self.y[i,0]-nu_p_hat[0]-(theta_p_hat[0]*self.T[i,0])])
            sum_rho_right += -1 * (xi.T @ inv_A_p @ psi)
            self.n_right += 1.0
        self.sum_right[0] = sum_rho_right

        self.pos = new_pos

        return 0
    
    cdef double get_proxy_delta_tilde(self) nogil:
        return ((self.sum_left[0] ** 2) / self.n_left +
                (self.sum_right[0] ** 2) / self.n_right)

    cdef void node_value(self, double* dest) nogil:
        memcpy(dest, self.theta_p_hat, sizeof(double))
