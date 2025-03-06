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

from ._utils cimport log
from ._utils cimport safe_realloc

cdef inline float _get_quantile(DOUBLE_t* y, SIZE_t n, SIZE_t start, DOUBLE_t quantile) nogil:
    cdef DOUBLE_t q_idx = quantile * (n - 1)
    cdef SIZE_t low = <int>floor(q_idx)
    cdef SIZE_t high = low + 1 if low + 1 < n else low
    cdef DOUBLE_t weight = q_idx - low
    return y[start+low] * (1 - weight) + y[start+high] * weight

cdef class GRFCriterionQF:
    def __cinit__(self, SIZE_t n_samples, double quantile, UINT32_t random_state):
        self.n_samples = n_samples
        self.quantile = quantile
        self.random_state = random_state

        self.theta_p_hat = NULL
        self.delta = NULL   # Does not have to be array when n_outputs=1
        self.y_parent = NULL
        self.y_left = NULL
        self.y_right = NULL
        
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
        cdef DOUBLE_t quantile = self.quantile
        self.start = start
        self.end = end

        safe_realloc(&self.y_parent, self.n_samples)
        safe_realloc(&self.y_left, self.n_samples)
        safe_realloc(&self.y_right, self.n_samples)
    
        cdef DOUBLE_t* yf_parent = self.y_parent
        for i in range(start, end):
            yf_parent[i] = self.y[samples[i], 0]
            
        sort(yf_parent + start, end - start)

        # Calculate quantile
        theta_p_hat[0] = _get_quantile(yf_parent, end - start, start, quantile)

        # Reset to pos=start
        self.reset()
        return 0

    cdef int reset(self) nogil except -1:
        cdef SIZE_t n_bytes = sizeof(double)
        memset(self.delta, 0, n_bytes)

        self.n_left = 0.0
        self.n_right = 0.0
        self.pos = self.start
        return 0

    cdef int update(self, SIZE_t new_pos) nogil except -1:
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

        cdef DOUBLE_t* yf_left = self.y_left
        cdef DOUBLE_t* yf_right = self.y_right

        self.n_left = 0.0
        self.n_right = 0.0


        for i in range(start, new_pos):
            yf_left[i] = self.y[samples[i], 0]
            self.n_left += 1
            
        sort(yf_left + start, new_pos - start)

        theta_hat_left = _get_quantile(yf_left, new_pos - start, start, quantile)

        for i in range(new_pos, end):
            yf_right[i] = self.y[samples[i], 0]
            self.n_right += 1
            
        sort(yf_right + new_pos, end - new_pos)

        theta_hat_right = _get_quantile(yf_right, end - new_pos, new_pos, quantile)

        """
        # print out calculated theta_hat of child nodes
        if self.random_state==1254920690:
            printf("%ld: yf_left=[", self.random_state)
            for i in range(start, new_pos):
                printf("%f, ", yf_left[i])
            printf("]\n")

            printf("%ld: y_left_new_ndarray=", self.random_state)
            print(y_left_new_ndarray)
            printf("%ld: theta_left(cy)=%f\n\n", self.random_state, theta_hat_left)
        
            printf("%ld: yf_right=[", self.random_state)
            for i in range(new_pos, end):
                printf("%f, ", yf_right[i])
            printf("]\n")

            printf("%ld: y_right_new_ndarray=", self.random_state)
            print(y_right_new_ndarray)
            printf("%ld: theta_right(cy)=%f\n\n", self.random_state, theta_hat_right)
        """

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

cdef inline void sort(DTYPE_t* Xf, SIZE_t n) nogil:
    if n == 0:
      return
    cdef int maxd = 2 * <int>log(n)
    introsort(Xf, n, maxd)


cdef inline void swap(DTYPE_t* Xf, SIZE_t i, SIZE_t j) nogil:
    # Helper for sort
    Xf[i], Xf[j] = Xf[j], Xf[i]


cdef inline DTYPE_t median3(DTYPE_t* Xf, SIZE_t n) nogil:
    # Median of three pivot selection, after Bentley and McIlroy (1993).
    # Engineering a sort function. SP&E. Requires 8/3 comparisons on average.
    cdef DTYPE_t a = Xf[0], b = Xf[n // 2], c = Xf[n - 1]
    if a < b:
        if b < c:
            return b
        elif a < c:
            return c
        else:
            return a
    elif b < c:
        if a < c:
            return a
        else:
            return c
    else:
        return b


# Introsort with median of 3 pivot selection and 3-way partition function
# (robust to repeated elements, e.g. lots of zero features).
cdef void introsort(DTYPE_t* Xf, SIZE_t n, int maxd) nogil:
    cdef DTYPE_t pivot
    cdef SIZE_t i, l, r

    while n > 1:
        if maxd <= 0:   # max depth limit exceeded ("gone quadratic")
            heapsort(Xf, n)
            return
        maxd -= 1

        pivot = median3(Xf, n)

        # Three-way partition.
        i = l = 0
        r = n
        while i < r:
            if Xf[i] < pivot:
                swap(Xf, i, l)
                i += 1
                l += 1
            elif Xf[i] > pivot:
                r -= 1
                swap(Xf, i, r)
            else:
                i += 1

        introsort(Xf, l, maxd)
        Xf += r
        n -= r


cdef inline void sift_down(DTYPE_t* Xf, SIZE_t start, SIZE_t end) nogil:
    # Restore heap order in Xf[start:end] by moving the max element to start.
    cdef SIZE_t child, maxind, root

    root = start
    while True:
        child = root * 2 + 1

        # find max of root, left child, right child
        maxind = root
        if child < end and Xf[maxind] < Xf[child]:
            maxind = child
        if child + 1 < end and Xf[maxind] < Xf[child + 1]:
            maxind = child + 1

        if maxind == root:
            break
        else:
            swap(Xf, root, maxind)
            root = maxind


cdef void heapsort(DTYPE_t* Xf, SIZE_t n) nogil:
    cdef SIZE_t start, end

    # heapify
    start = (n - 2) // 2
    end = n
    while True:
        sift_down(Xf, start, end)
        if start == 0:
            break
        start -= 1

    # sort by shrinking the heap, putting the max element immediately after it
    end = n - 1
    while end > 0:
        swap(Xf, 0, end)
        sift_down(Xf, 0, end)
        end = end - 1
