from cpython cimport Py_INCREF, PyObject, PyTypeObject

from libc.stdlib cimport free
from libc.string cimport memcpy
from libc.string cimport memset
from libc.stdint cimport SIZE_MAX
from libc.math cimport pow

import numpy as np
cimport numpy as np
np.import_array()

from scipy.sparse import csr_matrix

from ._utils cimport Stack
from ._utils cimport StackRecord
from ._utils cimport safe_realloc
from ._utils cimport sizet_ptr_to_ndarray

# =============================================================================
# Types and constants
# =============================================================================

from numpy import float64 as DTYPE
from numpy import float64 as DOUBLE

cdef double INFINITY = np.inf
cdef double EPSILON = np.finfo('double').eps

TREE_LEAF = -1
TREE_UNDEFINED = -2
cdef SIZE_t _TREE_LEAF = TREE_LEAF
cdef SIZE_t _TREE_UNDEFINED = TREE_UNDEFINED
cdef SIZE_t INITIAL_STACK_SIZE = 10

NODE_DTYPE = np.dtype({
    'names': ['left_child', 'right_child', 'depth', 'feature', 
              'threshold', 'n_node_samples', 'n_node_samples_train'],
    'formats': [np.intp, np.intp, np.intp, np.intp,
                np.float64, np.intp, np.intp],
    'offsets': [
        <Py_ssize_t> &(<Node*> NULL).left_child,
        <Py_ssize_t> &(<Node*> NULL).right_child,
        <Py_ssize_t> &(<Node*> NULL).depth,
        <Py_ssize_t> &(<Node*> NULL).feature,
        <Py_ssize_t> &(<Node*> NULL).threshold,
        <Py_ssize_t> &(<Node*> NULL).n_node_samples,
        <Py_ssize_t> &(<Node*> NULL).n_node_samples_train,
    ]
})

cdef class DepthFirstTreeBuilder():
    def __cinit__(self, BestSplitter splitter, SIZE_t min_samples_split,
                  SIZE_t min_samples_leaf, double min_weight_leaf,
                  SIZE_t max_depth, double min_impurity_decrease):
        self.splitter = splitter
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_leaf = min_weight_leaf
        self.max_depth = max_depth
        self.min_impurity_decrease = min_impurity_decrease

    cpdef build(self, Tree tree, np.ndarray[np.float64_t, ndim=2] X, np.ndarray y,
                np.ndarray T,
                np.ndarray samples_train,
                np.ndarray samples_val,
                np.ndarray sample_weight=None):

        cdef DOUBLE_t* sample_weight_ptr = NULL # sample weight is not used

        cdef int init_capacity

        if tree.max_depth <= 10:
            init_capacity = <int>((2 ** (tree.max_depth + 1)) - 1)
        else:
            init_capacity = 2047

        tree._resize(init_capacity)

        cdef BestSplitter splitter = self.splitter
        cdef SIZE_t max_depth = self.max_depth
        cdef SIZE_t min_samples_leaf = self.min_samples_leaf
        cdef SIZE_t min_samples_split = self.min_samples_split

        splitter.init(X, y, T, sample_weight_ptr, samples_train, samples_val)

        cdef SIZE_t start
        cdef SIZE_t end
        cdef SIZE_t start_val
        cdef SIZE_t end_val
        cdef SIZE_t depth
        cdef SIZE_t parent
        cdef bint is_left
        cdef SIZE_t n_node_samples = splitter.n_samples
        cdef SIZE_t n_node_samples_val = splitter.n_samples_val
        cdef SplitRecord split
        cdef SIZE_t node_id 

        cdef double impurity = INFINITY
        cdef double proxy_impurity = INFINITY
        cdef bint is_leaf
        cdef bint first = 1
        cdef SIZE_t max_depth_seen = -1
        cdef int rc = 0

        cdef Stack stack = Stack(INITIAL_STACK_SIZE)
        cdef StackRecord stack_record

        with nogil:
            # root node
            rc = stack.push(0, n_node_samples, 0, n_node_samples_val,
                            0, _TREE_UNDEFINED, 0, INFINITY, INFINITY, 0)
            if rc == -1:
                with gil:
                    raise MemoryError()

            while not stack.is_empty():
                stack.pop(&stack_record)
                
                start = stack_record.start
                end = stack_record.end
                start_val = stack_record.start_val
                end_val = stack_record.end_val
                depth = stack_record.depth
                parent = stack_record.parent
                is_left = stack_record.is_left

                n_node_samples = end - start
                n_node_samples_val = end_val - start_val

                splitter.node_reset(start, end, start_val, end_val)

                is_leaf = (depth >= max_depth or
                           n_node_samples < min_samples_split or
                           n_node_samples < 2 * min_samples_leaf or
                           n_node_samples_val < min_samples_split or
                           n_node_samples_val < 2 * min_samples_leaf)
                
                if not is_leaf:
                    splitter.node_split(&split)
                    is_leaf = (is_leaf or
                                split.pos >= end or
                                split.pos_val >= end_val 
                                )
                node_id = tree._add_node(parent, is_left, is_leaf, 
                                        split.feature, split.threshold,
                                        n_node_samples, n_node_samples_val)
                
                # Memory error
                if node_id == SIZE_MAX:
                    rc = -1
                    break
                
                splitter.node_value_val(tree.value + node_id) # \hat{\theta} is saved at this point
                
                if not is_leaf:
                    # Push right child on stack
                    rc = stack.push(split.pos, end, split.pos_val, end_val, depth + 1, node_id, 0,
                                    0, 0, 0)
                    if rc == -1:
                        break

                    # Push left child on stack
                    rc = stack.push(start, split.pos, start_val, split.pos_val, depth + 1, node_id, 1,
                                    0, 0, 0)
                    if rc == -1:
                        break
                
                if depth > max_depth_seen:
                    max_depth_seen = depth
                    
            if rc >= 0:
                rc = tree._resize_c(tree.node_count)

            if rc >= 0:
                tree.max_depth = max_depth_seen
        if rc == -1:
            raise MemoryError()

cdef class Tree:
    property children_left:
        def __get__(self):
            cdef np.ndarray[np.intp_t] arr
            arr = np.array([self.nodes[i].left_child for i in range(self.node_count)])
            return arr

    property children_right:
        def __get__(self):
            cdef np.ndarray[np.intp_t] arr
            arr = np.array([self.nodes[i].right_child for i in range(self.node_count)])
            return arr

    property depth:
        def __get__(self):
            cdef np.ndarray[np.intp_t] arr
            arr = np.array([self.nodes[i].depth for i in range(self.node_count)])
            return arr

    property n_leaves:
        def __get__(self):
            return np.sum(np.logical_and(
                self.children_left == -1,
                self.children_right == -1))

    property feature:
        def __get__(self):
            cdef np.ndarray[np.intp_t] arr
            arr = np.array([self.nodes[i].feature for i in range(self.node_count)])
            return arr

    property threshold:
        def __get__(self):
            cdef np.ndarray[np.float64_t] arr
            arr = np.array([self.nodes[i].threshold for i in range(self.node_count)])
            return arr

    property n_node_samples:
        def __get__(self):
            cdef np.ndarray[np.intp_t] arr
            arr = np.array([self.nodes[i].n_node_samples for i in range(self.node_count)])
            return arr

    property n_node_samples_train:
        def __get__(self):
            cdef np.ndarray[np.intp_t] arr
            arr = np.array([self.nodes[i].n_node_samples_train for i in range(self.node_count)])
            return arr

    property value: # estimate \hat{\theta} of each node
        def __get__(self):
            cdef np.ndarray[np.float64_t] arr
            arr = np.array([self.value[i] for i in range(self.node_count)])
            return arr

    def __cinit__(self, int n_features):
        self.n_features = n_features
        self.value_stride = 1 # n_outputs

        # Inner structures
        self.max_depth = 0
        self.node_count = 0
        self.capacity = 0
        self.value = NULL
        self.nodes = NULL

    def __dealloc__(self):
        """Destructor."""
        # Free all inner structures
        free(self.value)
        free(self.nodes)


    cdef int _resize(self, SIZE_t capacity) nogil except -1:
        if self._resize_c(capacity) != 0:
            # Acquire gil only if we need to raise
            with gil:
                raise MemoryError()

    cdef int _resize_c(self, SIZE_t capacity=SIZE_MAX) nogil except -1:
        if capacity == self.capacity and self.nodes != NULL:
            return 0

        if capacity == SIZE_MAX:
            if self.capacity == 0:
                capacity = 3  # default initial value
            else:
                capacity = 2 * self.capacity

        safe_realloc(&self.nodes, capacity)
        safe_realloc(&self.value, capacity * self.value_stride)

        # value memory is initialised to 0 to enable classifier argmax
        if capacity > self.capacity:
            memset(<void*>(self.value + self.capacity * self.value_stride), 0,
                   (capacity - self.capacity) * self.value_stride *
                   sizeof(double))
        

        # if capacity smaller than node_count, adjust the counter
        if capacity < self.node_count:
            self.node_count = capacity

        self.capacity = capacity
        return 0

    cdef SIZE_t _add_node(self, SIZE_t parent, bint is_left, bint is_leaf,
                          SIZE_t feature, double threshold,
                          SIZE_t n_node_samples, SIZE_t n_node_samples_val) nogil except -1:
        
        cdef SIZE_t node_id = self.node_count
        if node_id >= self.capacity:
            if self._resize_c() != 0:
                return SIZE_MAX

        cdef Node* node = &self.nodes[node_id]
        node.n_node_samples = n_node_samples_val
        node.n_node_samples_train = n_node_samples
        
        if parent != _TREE_UNDEFINED:
            if is_left:
                self.nodes[parent].left_child = node_id
            else:
                self.nodes[parent].right_child = node_id
            node.depth = self.nodes[parent].depth + 1
        else:
            node.depth = 0

        if is_leaf:
            node.left_child = _TREE_LEAF
            node.right_child = _TREE_LEAF
            node.feature = _TREE_UNDEFINED
            node.threshold = _TREE_UNDEFINED
        else:
            node.feature = feature
            node.threshold = threshold
        
        self.node_count += 1

        return node_id

    """
    # Previous version to calculate weight
    cpdef np.ndarray get_weight(self, object X_val):
        cdef np.ndarray[np.float64_t, ndim=2] out = np.zeros((self.node_count,X_val.shape[0]), dtype=np.float64) # out=alpha
        cdef SIZE_t n_samples = X_val.shape[0]
        
        cdef np.ndarray[SIZE_t, ndim=1] leaf_idx = self.apply(X_val)

        for i in range(self.node_count):
            if self.nodes[i].left_child == -1 and self.nodes[i].right_child == -1: # if node==leaf:
                datapoint_in_same_leaf = (i == leaf_idx)
                n_nodes_in_same_leaf = sum(datapoint_in_same_leaf)
                for j in range(n_samples):
                    if i==leaf_idx[j]:
                        out[i,j] = 1/n_nodes_in_same_leaf
        return out
    """
 
    cpdef np.ndarray get_weight(self, object X_val):
        cdef np.ndarray[np.int32_t, ndim=2] out = np.zeros((self.node_count, X_val.shape[0]), dtype=np.int32)
        cdef SIZE_t n_samples = X_val.shape[0]
        
        cdef np.ndarray[SIZE_t, ndim=1] leaf_idx = self.apply(X_val) # Finds leaf node taht each datapoint in X_val falls.

        for i in range(self.node_count):
            if self.nodes[i].left_child == -1 and self.nodes[i].right_child == -1: # if node==leaf:
                for j in range(n_samples):
                    if i==leaf_idx[j]:
                        out[i,j] = 1
        return out

    cpdef np.ndarray predict(self, object X):
        cdef np.ndarray[np.float64_t] value_arr
        value_arr = np.array([self.value[i] for i in range(self.node_count)])
        out = value_arr.take(self.apply(X), axis=0, mode='clip')
        return out

    cpdef np.ndarray apply(self, object X):
        return self._apply(X)

    cdef inline np.ndarray _apply(self, object X):
        # Check input
        if not isinstance(X, np.ndarray):
            raise ValueError("X should be in np.ndarray format, got %s"
                             % type(X))

        if X.dtype != DTYPE:
            raise ValueError("X.dtype should be np.float64, got %s" % X.dtype)

        # Extract input
        cdef const DTYPE_t[:, :] X_ndarray = X
        cdef SIZE_t n_samples = X.shape[0]

        # Initialize output
        cdef np.ndarray[SIZE_t] out = np.zeros((n_samples,), dtype=np.intp)
        cdef SIZE_t* out_ptr = <SIZE_t*> out.data

        # Initialize auxiliary data-structure
        cdef Node* node = NULL
        cdef SIZE_t i = 0

        with nogil:
            for i in range(n_samples):
                node = self.nodes
                # While node not a leaf
                while node.left_child != _TREE_LEAF:
                    # ... and node.right_child != _TREE_LEAF:
                    if X_ndarray[i, node.feature] <= node.threshold:
                        node = &self.nodes[node.left_child]
                    else:
                        node = &self.nodes[node.right_child]

                out_ptr[i] = <SIZE_t>(node - self.nodes)  # node offset

        return out
    