#cython: cdivision=True
#cython: boundscheck=False

import numpy as np
cimport numpy as np
cimport cython
from cython.parallel import prange

# data type of numpy arrays (double precision)
DTYPE = np.float
# corresponding compile-time type
ctypedef np.float_t DTYPE_t


cdef DTYPE_t chisquare(DTYPE_t x, DTYPE_t y) nogil:
    cdef DTYPE_t d
    if (x + y) > 0:
        d = ((x - y) * (x - y)) / (x + y)
    else:
        d = 0.0
    return d


cdef DTYPE_t euclidean(DTYPE_t x, DTYPE_t y) nogil:
    return (x - y) * (x - y)


def v2v_euclidean(np.ndarray[DTYPE_t, ndim=1] x1, np.ndarray[DTYPE_t, ndim=1] x2):
    cdef int d = len(x1)
    cdef int _d = len(x2)
    assert d == _d, "Dimension mismatch"
    cdef int i
    cdef DTYPE_t dist = 0.0
    for i in range(d):
        dist += euclidean(x1[i], x2[i])
    return dist


def v2v_chisquare(np.ndarray[DTYPE_t, ndim=1] x1, np.ndarray[DTYPE_t, ndim=1] x2):
    cdef int d = len(x1)
    cdef int _d = len(x2)
    assert d == _d, "Dimension mismatch"
    cdef int i
    cdef DTYPE_t dist = 0.0
    for i in range(d):
        dist += chisquare(x1[i], x2[i])
    return dist


def v2m_euclidean(np.ndarray[DTYPE_t, ndim=1] x, np.ndarray[DTYPE_t, ndim=2] m):
    """ Euclidean distances between vector x and row vectors in m
    """
    cdef int d = len(x)
    cdef int n = m.shape[0]
    cdef int _d = m.shape[1]
    assert _d == d, "Matrix dimension mismatch"
    cdef np.ndarray[DTYPE_t, ndim=1] res = np.zeros((n,))
    cdef int i
    cdef int j
    for i in range(n):
        for j in range(d):
            res[i] += euclidean(m[i, j], x[j])
    return res


def v2m_chisquare(np.ndarray[DTYPE_t, ndim=1] x, np.ndarray[DTYPE_t, ndim=2] m):
    """ Chisquare distances between vector x and row vectors in m
    """
    cdef int d = len(x)
    cdef int n = m.shape[0]
    cdef int _d = m.shape[1]
    assert _d == d, "Matrix dimension mismatch"
    cdef np.ndarray[DTYPE_t, ndim=1] res = np.zeros((n,))
    cdef int i
    cdef int j
    for i in range(n):
        for j in range(d):
            res[i] += chisquare(m[i, j], x[j])
    return res


def m2m_euclidean(np.ndarray[DTYPE_t, ndim=2] m1, np.ndarray[DTYPE_t, ndim=2] m2):
    """ Parallelized Euclidean distances between row vectors in m1 and m2
    """
    cdef int n1 = m1.shape[0]
    cdef int d = m1.shape[1]
    cdef int n2 = m2.shape[0]
    cdef int _d = m2.shape[1]
    assert _d == d, "Matrix dimension mismatch"
    cdef np.ndarray[DTYPE_t, ndim=2] res = np.zeros((n1, n2))
    cdef int i
    cdef int j
    cdef int k
    for i in prange(n1, nogil=True, schedule='runtime'):
        for j in range(n2):
            for k in range(d):
                res[i,j] += euclidean(m1[i, k], m2[j, k])
    return res


def m2m_chisquare(np.ndarray[DTYPE_t, ndim=2] m1, np.ndarray[DTYPE_t, ndim=2] m2):
    """ Parallelized Chisquare distances between row vectors in m1 and m2
    """
    cdef int n1 = m1.shape[0]
    cdef int d = m1.shape[1]
    cdef int n2 = m2.shape[0]
    cdef int _d = m2.shape[1]
    assert _d == d, "Matrix dimension mismatch"
    cdef np.ndarray[DTYPE_t, ndim=2] res = np.zeros((n1, n2))
    cdef int i
    cdef int j
    cdef int k
    for i in prange(n1, nogil=True, schedule='runtime'):
        for j in range(n2):
            for k in range(d):
                res[i,j] += chisquare(m1[i, k], m2[j, k])
    return res


def gram_euclidean(np.ndarray[DTYPE_t, ndim=2] m):
    """ Parallelized Euclidean distances between all row vectors of m
    """
    cdef int n = m.shape[0]
    cdef int d = m.shape[1]
    cdef np.ndarray[DTYPE_t, ndim=2] res = np.zeros((n, n))
    cdef int i
    cdef int j
    cdef int k
    for i in prange(n, nogil=True, schedule='runtime'):
        for j in range(i):
            for k in range(d):
                res[i,j] += euclidean(m[i, k], m[j, k])
            res[j,i] = res[i,j]
    return res


def gram_chisquare(np.ndarray[DTYPE_t, ndim=2] m):
    """ Parallelized Chisquare distances between all row vectors of m
    """
    cdef int n = m.shape[0]
    cdef int d = m.shape[1]
    cdef np.ndarray[DTYPE_t, ndim=2] res = np.zeros((n, n))
    cdef int i
    cdef int j
    cdef int k
    for i in prange(n, nogil=True, schedule='runtime'):
        for j in range(i):
            for k in range(d):
                res[i,j] += chisquare(m[i, k], m[j, k])
            res[j,i] = res[i,j]
    return res
