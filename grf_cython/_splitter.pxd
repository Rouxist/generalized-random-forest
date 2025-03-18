import numpy as np
cimport numpy as np

from ._criterion_see_qf cimport GRFCriterionSEEQF

from ._tree cimport DTYPE_t          # Type of X
from ._tree cimport DOUBLE_t         # Type of y, sample_weight
from ._tree cimport SIZE_t           # Type for indices and counters
from ._tree cimport INT32_t          # Signed 32 bit integer
from ._tree cimport UINT32_t         # Unsigned 32 bit integer

cdef struct SplitRecord:
    # Data to track sample split
    SIZE_t feature         # Which feature to split on.
    SIZE_t pos             # Split samples array at the given position,
                           # i.e. count of samples below threshold for feature.
                           # pos is >= end if the node is a leaf.
    SIZE_t pos_val         # Split samples_val array at the given position,
                           # i.e. count of samples below threshold for feature.
                           # pos_val is >= end_val if the node is a leaf.
    double threshold       # Threshold to split at.
    double improvement     # Impurity improvement given parent node.
    double impurity_left   # Impurity of the left split on train set.
    double impurity_right  # Impurity of the right split on train set.
    double impurity_left_val   # Impurity of the left split on validation set.
    double impurity_right_val  # Impurity of the right split on validation set.

cdef class BestSplitter:
    cdef public GRFCriterionSEEQF criterion
    cdef public GRFCriterionSEEQF criterion_val
    cdef public SIZE_t max_features
    cdef public SIZE_t min_samples_leaf
    cdef public double min_balancedness_tol
    cdef public bint honest

    cdef UINT32_t rand_r_state
    
    cdef SIZE_t* samples
    cdef SIZE_t n_samples
    cdef SIZE_t* samples_val
    cdef SIZE_t n_samples_val
    cdef SIZE_t* features
    cdef SIZE_t n_features
    cdef DTYPE_t* feature_values
    cdef DTYPE_t* feature_values_val

    cdef SIZE_t start                    
    cdef SIZE_t end                      
    cdef SIZE_t start_val                
    cdef SIZE_t end_val

    cdef const DTYPE_t[:, :] X
    cdef const DOUBLE_t[:, ::1] y
    cdef DOUBLE_t* sample_weight

    cdef int init_sample_inds(self, SIZE_t* samples,
                              const SIZE_t[::1] np_samples,
                              DOUBLE_t* sample_weight,
                              SIZE_t* n_samples, 
                              ) nogil except -1

    cdef int init(self, const DTYPE_t[:, :] X, const DOUBLE_t[:, ::1] y,
                  DOUBLE_t* sample_weight,
                  const SIZE_t[::1] np_samples_train,
                  const SIZE_t[::1] np_samples_val) nogil except -1

    cdef int node_reset(self, SIZE_t start, SIZE_t end,
                     SIZE_t start_val, SIZE_t end_val) nogil except -1
    
    cdef int node_split(self, SplitRecord* split) nogil except -1
    cdef void node_value_val(self, double* dest) nogil
    