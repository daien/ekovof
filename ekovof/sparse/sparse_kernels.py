""" Efficient computation of kernels between between sparse vectors of double floats

Notes
-----
The sparse vectors are assumed to be
- either colum vectors in CSC format
- either row vectors in CSR format
(cf. scipy.sparse, passing from one to the other is just transposing and the
internal structure -- data, indices and indptr -- doesn't change)

WARNING: we do not check this for efficiency reasons...
"""

import numpy as np
from scipy.sparse import isspmatrix_csc, isspmatrix_csr
import numexpr as ne

from ..utils import get_heuristic_gamma
from ..generic_kernel import GenericKernel, DTYPE
from . import sparse_distances


AVAILABLE_KERNELS = ("linear", "intersection", "totvar", "chisquare", "l2")
RBF_KERNELS = ("totvar", "chisquare", "l2")


class SparseKernel(GenericKernel):
    """ Kernel operations between sparse vectors and matrices

    Vectors are either columns (CSC format) or rows (CSR format).
    """

    def __init__(self, name, **okwargs):
        """ Build a kernel object that operates on sparse vectors

        Parameters
        ----------
        name: string,
              the name of the kernel, one of
                  "linear"      : K(x,y) = sum_i x_i * y_i
                  "intersection": K(x,y) = sum_i min(x_i,y_i)
                  "totvar"      : K(x,y) = exp( - \gamma sum_i |x_i - y_i| )
                  "chisquare"   : K(x,y) = exp( - \gamma sum_i (x_i - y_i)^2 / (x_i + y_i) )
                  "l2"          : K(x,y) = exp( - \gamma sum_i (x_i - y_i)^2 )

        okwargs: other key-word arguments (cf. GenericKernel for defaults)

        """
        assert name in AVAILABLE_KERNELS, "Unknown kernel %s" % name
        super(SparseKernel, self).__init__(name, **okwargs)
        self.use_rbf = okwargs.get('use_rbf', name in RBF_KERNELS)
        if name not in RBF_KERNELS and self.use_rbf:
            raise ValueError('Cannot use RBF with non-RBF kernel {}'.format(name))
        # get the module used to compute the sparse distances (FIXME ugly)
        spdist = None
        exec('import sparse_distances.sparse_%s as spdist' % name)
        # define "raw" distance functions between sparse vectors (from C library)
        self._v2v_f = spdist.v2v  # raw dist. between vectors
        self._v2m_f = spdist.v2m  # raw dist. between a vector and a matrix
        self._m2m_f = spdist.m2m  # raw dist. between a (test) matrix and a (train) matrix
        self._gram_f = spdist.gram  # raw gram matrix of a list of vectors
        if name in ('linear', 'intersection'):
            self.is_additive = True

    @classmethod
    def is_kernel_for(cls, name):
        # TODO fix this (needed for factory function in get_dense_kernel)
        return False

    def v2v(self, x1, x2):
        """ Kernel value between two sparse vectors
        """
        # indices are not necessarily sorted, so sort them here (required)
        x1.sort_indices()
        x2.sort_indices()
        # note: does nothing if already sorted
        d = self._v2v_f(x1.indptr, x1.indices, x1.data,
                        x2.indptr, x2.indices, x2.data)
        # eventually return rbf'd version
        return np.exp(-self.gamma * d) if self.use_rbf else d

    def v2m(self, x, m):
        """ Kernel evaluations between a sparse vector and a sparse matrix.

        if 'self.center == True', then the result array is centered in the
        feature space induced by the kernel by using previously-computed
        (during Gram matrix computation) parameters

        Note: if self.libsvm_fmt == True, then the first element of the array
        is always '1' to comply with libsvm's format, the kernel values come
        after
        """
        if isspmatrix_csr(x):
            assert isspmatrix_csr(m), "Type mismatch"
            _n, _d = x.shape
            assert _n == 1, "Not a vector"
            n, d = m.shape
            assert _d == d, "Dimension mismatch"
        elif isspmatrix_csc(x):
            assert isspmatrix_csc(m), "Type mismatch"
            _d, _n = x.shape
            assert _n == 1, "Not a vector"
            d, n = m.shape
            assert _d == d, "Dimension mismatch"
        else:
            raise ValueError("x is not a CSR or CSC vector")
        # indices are not necessarily sorted, so sort them here (required)
        x.sort_indices()
        m.sort_indices()
        # note: does nothing if already sorted
        offset = (self.libsvm_fmt and 1) or 0
        kern_row = np.ones(offset + n, dtype=DTYPE)
        kern_row[offset:] = self._v2m_f(
            n,
            x.indptr, x.indices, x.data,
            m.indptr, m.indices, m.data)
        # eventually rbf it
        if self.use_rbf:
            kern_row[offset:] = np.exp(-self.gamma * kern_row[offset:])
        # eventually center (in place) the kernel row in the feature space
        if self.center:
            self._center_rows(kern_row[offset:])
        return kern_row

    def m2m(self, mtest, m):
        """ Kernel matrix between mtest and all m

        if 'self.center == True', then center the Gram matrix in the feature
        space induced by the kernel, which assumes the centering parameters
        were previously computed

        Note: if self.libsvm_fmt == True, then the first element of the array
        is the row number (from 1 for libsvm), the kernel values come after
        """
        if isspmatrix_csr(mtest):
            assert isspmatrix_csr(m), "Type mismatch"
            nte, _d = mtest.shape
            ntr, d = m.shape
            assert _d == d, "Dimension mismatch"
        elif isspmatrix_csc(mtest):
            assert isspmatrix_csc(m), "Type mismatch"
            _d, nte = mtest.shape
            d, ntr = m.shape
            assert _d == d, "Dimension mismatch"
        else:
            raise ValueError("mtest is not a CSR or CSC vector")
        # indices are not necessarily sorted, so sort them here (required)
        mtest.sort_indices()
        m.sort_indices()
        # note: does nothing if already sorted
        # pre-allocate result array (required by C library)
        kern_vals = np.zeros((nte, ntr), dtype=DTYPE, order='C')
        # compute kernel values in kern_vals
        self._m2m_f(kern_vals, mtest.indptr, mtest.indices, mtest.data,
            m.indptr, m.indices, m.data, self.num_threads)
        # eventually rbf the gram matrix
        if self.use_rbf:
            self.gamma = get_heuristic_gamma(kern_vals, self.gamma)
            #kern_vals = exp(-self.gamma*kern_vals)
            kern_vals = ne.evaluate("exp(-%f*kern_vals)" % (self.gamma))
        # eventually center (in place) the kernel matrix in the feature space
        if self.center:
            self._center_rows(kern_vals)
        # make 'id' col to contain line number (index starts from 1 for libsvm)
        if self.libsvm_fmt:
            id_col = np.arange(1, nte + 1, dtype=DTYPE)[:, np.newaxis]
            # add 'id' to kernel matrix (needed for training of libsvm)
            kern_mat = np.hstack((id_col, kern_vals))
        else:
            # no additional first column
            kern_mat = kern_vals  # reference
        return kern_mat

    def gram(self, m):
        """Return the kernel gram matrix between from the sparse matrix 'm'

        if 'self.center == True', then center the Gram matrix in the feature
        space induced by the kernel, and additionally set the Kernel parameters
        (array of the column means and 1/n*mean of all the kernel evaluations)
        needed to center in the same way the future kernel evaluations

        Note: if self.libsvm_fmt == True, then the first element of the array
        is the row number (from 1 for libsvm), the kernel values come after
        """
        if isspmatrix_csr(m):
            num_samps, d = m.shape
        elif isspmatrix_csc(m):
            d, num_samps = m.shape
        else:
            raise ValueError("mtest is not a CSR or CSC vector")
        # CSC indices are not necessarily sorted, so sort them here (required)
        m.sort_indices()
        # note: does nothing if already sorted
        # pre-allocate result array (required by C library)
        kern_vals = np.zeros((num_samps, num_samps), dtype=DTYPE, order='C')
        # compute kernel values in kern_vals
        self._gram_f(kern_vals, m.indptr, m.indices, m.data, self.num_threads)
        # eventually compute and define gamma, and rbf the gram matrix
        if self.use_rbf:
            self.gamma = get_heuristic_gamma(kern_vals, self.gamma)
            #kern_vals = exp(-self.gamma*kern_vals)
            kern_vals = ne.evaluate("exp(-%f*kern_vals)" % (self.gamma))
        # eventually center (in place) the kernel matrix in the feature space
        if self.center:
            self._center_gram(kern_vals)  # additionnally sets centering params
        # make 'id' col to contain line number (index starts from 1 for libsvm)
        if self.libsvm_fmt:
            id_col = np.arange(1, num_samps + 1, dtype=DTYPE)[:, np.newaxis]
            # add 'id' to kernel matrix (needed for training of libsvm)
            kern_mat = np.hstack((id_col, kern_vals))
        else:
            # no additional first column
            kern_mat = kern_vals  # reference
        return kern_mat
