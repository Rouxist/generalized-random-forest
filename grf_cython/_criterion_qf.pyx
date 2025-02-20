from libc.stdlib cimport calloc
from libc.stdlib cimport free
from libc.string cimport memcpy
from libc.string cimport memset

from libc.math cimport floor

import numpy as np
cimport numpy as np
np.import_array()

cdef double INFINITY = np.inf

from numpy.linalg import inv
from functools import partial
from scipy.optimize import fsolve

from ._utils cimport safe_realloc

cdef inline float _get_quantile(np.ndarray[np.float64_t, ndim=1] y, double n, double quantile):
    cdef double q_idx = quantile * (n - 1)
    cdef SIZE_t low = <int>floor(q_idx)
    cdef SIZE_t high = low + 1 if low + 1 < n else low
    cdef DOUBLE_t weight = q_idx - low
    return y[low] * (1 - weight) + y[high] * weight

cdef class GRFCriterionQF:
    def __cinit__(self, SIZE_t n_samples, double quantile, UINT32_t random_state):
        self.n_samples = n_samples
        self.quantile = quantile
        self.random_state = random_state

        self.theta_p_hat = NULL
        self.delta = NULL   # Does not have to be array when n_outputs=1
        self.y_parent = NULL
        
        self.theta_p_hat = <double*> calloc(1, sizeof(double))
        self.delta = <double*> calloc(1, sizeof(double))

    cdef int init(self, const DTYPE_t[:, :] X, const DOUBLE_t[:, ::1] y, SIZE_t* samples) nogil except -1:
        """
        Stores data to be used.
        """
        self.X = X
        self.y = y
        self.samples = samples

        return 0
    
    cdef int node_reset(self, SIZE_t start, SIZE_t end):
        """
        Calculates \hat{\theta_P} of node. Will later be used when calculating \tilde{\Delta}.
        GIL makes this function slow.
        """
        
        cdef SIZE_t* samples = self.samples
        cdef double* theta_p_hat = self.theta_p_hat
        cdef double quantile = self.quantile
        self.start = start
        self.end = end
        self.n_node_samples = end - start

        cdef np.ndarray[np.float64_t, ndim=1] y_parent_new_ndarray = np.array([])

        for p in range(start, end):
            i = samples[p]
            y_parent_new_ndarray = np.append(y_parent_new_ndarray, self.y[i,0])
        
        # Calculate quantile
        y_parent_new_ndarray.sort()
        theta_p_hat[0] = _get_quantile(y_parent_new_ndarray, self.n_node_samples, quantile)

        # if self.random_state==0:
        #     printf("%ld: y_parent=", self.random_state)
        #     print(y_parent_new_ndarray)
        #     printf("%ld: thata_p_hat=%f, self.n_node_samples=%f\n\n", self.random_state, theta_p_hat[0], self.n_node_samples)

        # Reset to pos=start
        self.reset()
        return 0

    cdef int reset(self) nogil except -1:
        cdef SIZE_t n_bytes = sizeof(double)
        memset(self.delta, 0, n_bytes)

        self.n_left = 0.0
        self.n_right = self.n_node_samples
        self.pos = self.start
        return 0

    cdef int update(self, SIZE_t new_pos):
        """
        Calculates (\hat{\theta}_{C_1} - \hat{\theta}_{C_2})^2.
        """

        cdef SIZE_t start = self.start
        cdef SIZE_t pos = self.pos
        cdef SIZE_t end = self.end
        cdef double* delta = self.delta
        
        cdef SIZE_t* samples = self.samples

        cdef double psi = 1.0
        cdef double inv_a_p = -1.0
        cdef double sum_rho_left = 0
        cdef double sum_rho_right = 0

        cdef double quantile = self.quantile
        cdef double theta_hat_left, theta_hat_right

        self.n_left = 0.0
        self.n_right = 0.0

        cdef np.ndarray[np.float64_t, ndim=1] y_left_new_ndarray = np.array([])

        for p in range(new_pos, end):
            i = samples[p]
            y_left_new_ndarray = np.append(y_left_new_ndarray, self.y[i,0])
            self.n_left += 1

        theta_hat_left = _get_quantile(y_left_new_ndarray, self.n_left, quantile)


        cdef np.ndarray[np.float64_t, ndim=1] y_right_new_ndarray = np.array([])

        for p in range(new_pos, end):
            i = samples[p]
            y_right_new_ndarray = np.append(y_right_new_ndarray, self.y[i,0])
            self.n_right += 1

        theta_hat_right = _get_quantile(y_right_new_ndarray, self.n_right, quantile)


        self.delta[0] = (theta_hat_left - theta_hat_right) ** 2

        self.pos = new_pos

        return 0
    
    cdef double get_proxy_delta(self) nogil:
        """
        Calculates exact Delta(C_1, C_2).
        """
        return self.delta[0] * self.n_left * self.n_right / ((self.n_left + self.n_right)**2) 

    cdef void node_value(self, double* dest) nogil:
        memcpy(dest, self.theta_p_hat, sizeof(double))
