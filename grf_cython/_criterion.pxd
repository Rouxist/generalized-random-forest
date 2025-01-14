import numpy as np
cimport numpy as np

from ._tree cimport DTYPE_t          # Type of X
from ._tree cimport DOUBLE_t         # Type of y, sample_weight
from ._tree cimport SIZE_t           # Type for indices and counters
from ._tree cimport UINT32_t         # Unsigned 32 bit integer

cdef class GRFCriterion:
    cdef const DTYPE_t[:, :] X
    cdef const DOUBLE_t[:, ::1] y
    cdef double* y_parent

    cdef SIZE_t n_samples
    cdef double n_node_samples
    cdef double n_left
    cdef double n_right
    cdef UINT32_t random_state # not used

    cdef SIZE_t* samples
    cdef SIZE_t start
    cdef SIZE_t pos
    cdef SIZE_t end
    cdef double*theta_p_hat

    cdef double* sum_left
    cdef double* sum_right

    cdef int init(self, const DTYPE_t[:, :] X, const DOUBLE_t[:, ::1] y, 
                  SIZE_t* samples) nogil except -1
    cdef int node_reset(self, SIZE_t start, SIZE_t end)
    cdef int reset(self) nogil except -1
    cdef int update(self, SIZE_t new_pos) nogil except -1
    cdef double get_proxy_delta_tilde(self) nogil
    cdef void node_value(self, double* dest) nogil
