from ._criterion cimport GRFCriterion
from ._criterion_cf cimport GRFCriterionCF

from libc.stdlib cimport free
from libc.string cimport memcpy
from libc.math cimport floor

import copy
import numpy as np
cimport numpy as np
np.import_array()

from ._utils cimport log
from ._utils cimport rand_int
from ._utils cimport rand_uniform
from ._utils cimport RAND_R_MAX
from ._utils cimport safe_realloc

cdef double INFINITY = np.inf
cdef DTYPE_t FEATURE_THRESHOLD = 1e-7

cdef inline void _init_split(SplitRecord* self, SIZE_t start_pos, SIZE_t start_pos_val) nogil:
    self.impurity_left = INFINITY
    self.impurity_right = INFINITY
    self.impurity_left_val = INFINITY
    self.impurity_right_val = INFINITY
    self.pos = start_pos
    self.pos_val = start_pos_val
    self.feature = 0
    self.threshold = 0.
    self.improvement = -INFINITY

cdef class BestSplitter():
    def __cinit__(self, GRFCriterionCF criterion, GRFCriterionCF criterion_val,
                  SIZE_t max_features, SIZE_t min_samples_leaf,
                  DTYPE_t min_balancedness_tol, bint honest, UINT32_t random_state):
        
        self.criterion = criterion
        if honest:
            self.criterion_val = criterion_val
        else:
            self.criterion_val = criterion
        
        self.features = NULL

        self.samples = NULL # Indices of datapoints being used in node. Indices are from the whole dataset X.
                            # 다만 weight가 0인 sample은 포함하지 않음.
                            # 루트 노드는 _gradient_tree.py의 builder.build에서 넘겨받은 self.indices_train 가 self.samples와 같음.
                            # 자식 노드는 그 노드에 해당하는 것만 포함되게 array가 만들어지는 것으로 추측됨... 나중에 확인해보자
        self.n_samples = 0
        self.samples_val = NULL
        self.n_samples_val = 0
        self.feature_values = NULL
        self.feature_values_val = NULL

        self.max_features = max_features
        self.min_samples_leaf = min_samples_leaf
        self.min_balancedness_tol = min_balancedness_tol
        self.honest = honest
        self.rand_r_state = random_state

    cdef int init_sample_inds(self, SIZE_t* samples,
                               const SIZE_t[::1] np_samples,
                               DOUBLE_t* sample_weight,
                               SIZE_t* n_samples
                               ) nogil except -1:
        cdef SIZE_t i, j, ind
        j = 0

        for i in range(np_samples.shape[0]):
            ind = np_samples[i]
            if sample_weight == NULL or sample_weight[ind] > 0.0:
                samples[j] = ind
                j += 1
        n_samples[0] = j

    cdef int init(self, const DTYPE_t[:, :] X, const DOUBLE_t[:, ::1] y,
                  const DOUBLE_t[:, ::1] T,
                  DOUBLE_t* sample_weight,
                  const SIZE_t[::1] np_samples_train,
                  const SIZE_t[::1] np_samples_val) nogil except -1:

        cdef SIZE_t n_features = X.shape[1]
        cdef SIZE_t n_samples = np_samples_train.shape[0]

        # Create a new array which will be used to store nonzero weighted sample indices of the training set.
        cdef SIZE_t* samples = safe_realloc(&self.samples, n_samples)
        
        # Initialize this array based on the numpy array np_samples_train
        self.init_sample_inds(self.samples, np_samples_train, sample_weight,
                              &self.n_samples)

        cdef SIZE_t* features = safe_realloc(&self.features, n_features)
        for i in range(n_features):
            features[i] = i

        self.n_features = n_features

        safe_realloc(&self.feature_values, self.n_samples)

        self.X = X
        self.y = y
        self.T = T
        self.sample_weight = sample_weight

        # Initialize criterion
        self.criterion.init(self.X, self.y, 
                            self.T,
                            self.samples)
        
        # If `honest=True` do initialize analogous validation set objects
        cdef SIZE_t n_samples_val
        cdef SIZE_t* samples_val
        if self.honest:
            n_samples_val = np_samples_val.shape[0]
            samples_val = safe_realloc(&self.samples_val, n_samples_val)
            self.init_sample_inds(self.samples_val, np_samples_val, sample_weight,
                            &self.n_samples_val, 
                            )
            safe_realloc(&self.feature_values_val, self.n_samples_val)
            self.criterion_val.init(self.X, self.y, self.T, self.samples_val)
        else:
            self.n_samples_val = self.n_samples
            self.samples_val = self.samples
            self.feature_values_val = self.feature_values
        
        return 0
    
    cdef int node_reset(self, SIZE_t start, SIZE_t end, 
                        SIZE_t start_val, SIZE_t end_val) nogil except -1:
                        
        self.start = start
        self.end = end
        self.start_val = start_val
        self.end_val = end_val
        
        # Calculate \hat{\theta_P}
        with gil: # `fsolve` in criterion.node_reset() requuires gil
            self.criterion.node_reset(start, end)
            if self.honest:
                self.criterion_val.node_reset(start_val, end_val)
        
        return 0
    
    cdef int node_split(self, SplitRecord* split) nogil except -1:
        cdef SIZE_t* samples = self.samples
        cdef SIZE_t start = self.start
        cdef SIZE_t end = self.end
        cdef SIZE_t* samples_val = self.samples_val
        cdef SIZE_t start_val = self.start_val
        cdef SIZE_t end_val = self.end_val

        cdef SIZE_t* features = self.features
        cdef SIZE_t n_features = self.n_features

        cdef DTYPE_t* Xf = self.feature_values
        cdef DTYPE_t* Xf_val = self.feature_values_val
        cdef SIZE_t max_features = self.max_features
        cdef SIZE_t min_samples_leaf = self.min_samples_leaf
        cdef UINT32_t* random_state = &self.rand_r_state

        # Two variables below are for test to print out `samples` with for loop
        cdef SIZE_t n_samples = self.n_samples
        cdef SIZE_t idx


        cdef SplitRecord best, current
        cdef double current_proxy_improvement = -INFINITY
        cdef double best_proxy_improvement = -INFINITY
        cdef double current_threshold = 0.0

        cdef SIZE_t f_i = n_features
        cdef SIZE_t f_j
        cdef SIZE_t p
        cdef SIZE_t p_val
        cdef SIZE_t i

        cdef SIZE_t n_visited_features = 0

        cdef SIZE_t partition_end

        _init_split(&best, end, end_val)
        
        # While loop to iterate features up to `max_features`
        while (n_visited_features < max_features):
            n_visited_features += 1

            f_j = rand_int(0, f_i, random_state)

            current.feature = features[f_j]

            for i in range(start, end):
                    Xf[i] = self.X[samples[i], current.feature]
            
            sort(Xf + start, samples + start, end - start)
            
            if self.honest:
                    for i in range(start_val, end_val):
                        Xf_val[i] = self.X[samples_val[i], current.feature]
                    
                    sort(Xf_val + start_val, samples_val + start_val, end_val - start_val)
            
            f_i -= 1
            features[f_i], features[f_j] = features[f_j], features[f_i]

            # Iterate all possible splits that could occur in a feature
            self.criterion.reset()
            if self.honest:
                self.criterion_val.reset()
            
            p = start + <int>floor((.5 - self.min_balancedness_tol) * (end - start)) - 1
            p_val = start_val

            while p < end and p_val < end_val:
                while (p + 1 < end and Xf[p + 1] <= Xf[p] + FEATURE_THRESHOLD):
                    p += 1
                p += 1
                
                current_threshold = Xf[p] / 2.0 + Xf[p - 1] / 2.0

                if ((current_threshold == Xf[p]) or
                            (current_threshold == INFINITY) or
                            (current_threshold == -INFINITY)):
                            current_threshold = Xf[p - 1]
                
                if self.honest:
                    while (p_val < end_val and Xf_val[p_val] <= current_threshold):
                        p_val += 1
                else:
                    p_val = p 
                
                if p < end and p_val < end_val:
                    current.pos = p
                    current.pos_val = p_val

                    if (end - current.pos) < (.5 - self.min_balancedness_tol) * (end - start):
                        break
                    if (current.pos_val - start_val) < (.5 - self.min_balancedness_tol) * (end_val - start_val):
                        continue
                    if (end_val - current.pos_val) < (.5 - self.min_balancedness_tol) * (end_val - start_val):
                        break

                    if (current.pos - start) < min_samples_leaf:
                        continue
                    if (end - current.pos) < min_samples_leaf:
                        break

                    if (current.pos_val - start_val) < min_samples_leaf:
                        continue
                    if (end_val - current.pos_val) < min_samples_leaf:
                        break

                    """
                    printf("seed %d: curr feature = %d, curr pos=%d, y = [", random_state[0], current.feature, current.pos)
                    for idx in range(n_samples):
                        # printf("%d, ",samples[idx])
                        # printf("%.0f, ", self.y[idx,0]) # something's wrong
                        printf("%.1f, ", self.y[samples[idx],0])
                    printf("]\n")
                    """
                    
                    with gil:
                        self.criterion.update(current.pos)
                        if self.honest:
                            self.criterion_val.update(current.pos_val)
                    current_proxy_improvement = self.criterion.get_proxy_delta_tilde()
                    
                    """
                    with gil:
                        if (random_state[0] == 0):
                            printf("seed %ld: current_proxy_improvement=%.2f\n", random_state[0], current_proxy_improvement)
                    """

                    if current_proxy_improvement > best_proxy_improvement:
                        best_proxy_improvement = current_proxy_improvement
                        current.threshold = current_threshold
                        best = current
        
        if best.pos < end and best.pos_val < end_val:
            partition_end = end
            p = start

            while p < partition_end:
                if self.X[samples[p], best.feature] <= best.threshold:
                    p += 1
                else:
                    partition_end -= 1
                    samples[p], samples[partition_end] = samples[partition_end], samples[p]

            if self.honest:
                partition_end = end_val
                p = start_val

                while p < partition_end:
                    if self.X[samples_val[p], best.feature] <= best.threshold:
                        p += 1
                    else:
                        partition_end -= 1

                        samples_val[p], samples_val[partition_end] = samples_val[partition_end], samples_val[p]
            self.criterion.reset()
            with gil:
                self.criterion.update(best.pos)
            
            if self.honest:
                self.criterion_val.reset()
                with gil:
                    self.criterion_val.update(best.pos_val)

        split[0] = best
        return 0
    
    cdef void node_value_val(self, double* dest) nogil:
        self.criterion_val.node_value(dest)


cdef inline void sort(DTYPE_t* Xf, SIZE_t* samples, SIZE_t n) nogil:
    if n == 0:
      return
    cdef int maxd = 2 * <int>log(n)
    introsort(Xf, samples, n, maxd)


cdef inline void swap(DTYPE_t* Xf, SIZE_t* samples,
        SIZE_t i, SIZE_t j) nogil:
    # Helper for sort
    Xf[i], Xf[j] = Xf[j], Xf[i]
    samples[i], samples[j] = samples[j], samples[i]


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
cdef void introsort(DTYPE_t* Xf, SIZE_t *samples,
                    SIZE_t n, int maxd) nogil:
    cdef DTYPE_t pivot
    cdef SIZE_t i, l, r

    while n > 1:
        if maxd <= 0:   # max depth limit exceeded ("gone quadratic")
            heapsort(Xf, samples, n)
            return
        maxd -= 1

        pivot = median3(Xf, n)

        # Three-way partition.
        i = l = 0
        r = n
        while i < r:
            if Xf[i] < pivot:
                swap(Xf, samples, i, l)
                i += 1
                l += 1
            elif Xf[i] > pivot:
                r -= 1
                swap(Xf, samples, i, r)
            else:
                i += 1

        introsort(Xf, samples, l, maxd)
        Xf += r
        samples += r
        n -= r


cdef inline void sift_down(DTYPE_t* Xf, SIZE_t* samples,
                           SIZE_t start, SIZE_t end) nogil:
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
            swap(Xf, samples, root, maxind)
            root = maxind


cdef void heapsort(DTYPE_t* Xf, SIZE_t* samples, SIZE_t n) nogil:
    cdef SIZE_t start, end

    # heapify
    start = (n - 2) // 2
    end = n
    while True:
        sift_down(Xf, samples, start, end)
        if start == 0:
            break
        start -= 1

    # sort by shrinking the heap, putting the max element immediately after it
    end = n - 1
    while end > 0:
        swap(Xf, samples, 0, end)
        sift_down(Xf, samples, 0, end)
        end = end - 1
