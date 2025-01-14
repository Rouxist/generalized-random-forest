import numpy as np
cimport numpy as np

ctypedef np.npy_float64 DTYPE_t          # Type of X
ctypedef np.npy_float64 DOUBLE_t         # Type of y, sample_weight
ctypedef np.npy_intp SIZE_t              # Type for indices and counters
ctypedef np.npy_int32 INT32_t            # Signed 32 bit integer
ctypedef np.npy_uint32 UINT32_t          # Unsigned 32 bit integer

from ._splitter cimport BestSplitter
from ._splitter cimport SplitRecord

cdef struct Node:
    SIZE_t left_child                    # idx of the left child
    SIZE_t right_child                   # idx of the right child
    SIZE_t depth                         # the depth level of the node
    SIZE_t feature                       # Selected feature to split the node
    DOUBLE_t threshold                   # Threshold value of the node
    SIZE_t n_node_samples                # Number of samples at the node on the val set
    SIZE_t n_node_samples_train          # Number of samples at the node on the train set

cdef class Tree:
    cdef public SIZE_t n_features        # Number of features in X

    # Inner structures: values are stored separately from node structure,
    # since size is determined at runtime.
    cdef public SIZE_t max_depth         # Max depth of the tree
    cdef public SIZE_t node_count        # Counter for node IDs
    cdef public SIZE_t capacity          # Capacity of tree, in terms of nodes
    cdef Node* nodes                     # Array of nodes
    cdef double* value                   # (capacity, n_outputs, max_n_classes) array of values
    cdef SIZE_t value_stride             # = n_outputs * max_n_classes

    # Methods
    cdef SIZE_t _add_node(self, SIZE_t parent, bint is_left, bint is_leaf,
                          SIZE_t feature, double threshold, 
                          SIZE_t n_node_samples_train,
                          SIZE_t n_node_samples_val
                          ) nogil except -1
    cdef int _resize(self, SIZE_t capacity) nogil except -1
    cdef int _resize_c(self, SIZE_t capacity=*) nogil except -1

    cpdef np.ndarray get_weight(self, object X_val)
    cpdef np.ndarray predict(self, object X)

    cpdef np.ndarray apply(self, object X)
    cdef np.ndarray _apply(self, object X)

cdef class DepthFirstTreeBuilder:
    cdef BestSplitter splitter              # Splitting algorithm

    cdef SIZE_t min_samples_split       # Minimum number of samples in an internal node
    cdef SIZE_t min_samples_leaf        # Minimum number of samples in a leaf
    cdef double min_weight_leaf         # Minimum weight in a leaf
    cdef SIZE_t max_depth               # Maximal tree depth
    cdef double min_impurity_decrease   # Impurity threshold for early stopping

    # cpdef build(self, Tree tree, object X, np.ndarray y,
    cpdef build(self, Tree tree, np.ndarray[np.float64_t, ndim=2] X, np.ndarray y,
                np.ndarray samples_train,
                np.ndarray samples_val,
                np.ndarray sample_weight=*)
