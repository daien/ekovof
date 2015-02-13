""" Module containing the definition of the generic interface of Kernel objects
"""


import numpy as np

from .utils import center_gram, center_rows, safe_len, mpmap


DTYPE = np.float64


class AbstractClassError(Exception):
    pass


class GenericKernel(object):
    """ Abstract class for kernel operations between vectors and matrices

    Classes implementing this interface can define the following methods:

        v2v(self, vec1, vec2): (*must* be implemented by subclasses)
            kernel evaluation between two vectors

        v2m(self, vec, mat):
            array of all kernel evaluations between a vector and a matrix

        m2m(self, mat1, mat2):
            pairwise kernel evaluations between all vectors in mat1 and all
            vectors in mat2

        gram(self, mat):
            kernel gram matrix between all vectors in mat

    and set the following attributes:

        use_rbf: boolean, indicate whether it is a (Gaussian) RBF kernel

        gamma: float, optional: bandwidth parameters for RBF kernels

    Parameters
    ----------
    name: string,
          the name of the kernel,

    libsvm_fmt: boolean, optional, default: False,
                whether to add an extra first column of numerical sample
                index to comply with libsvm's format

    center: boolean, optional, default: False,
            whether to center in feature space the kernel

    gamma: float, optional, default: None,
           the bandwidth parameter for RBF kernels

    constant_diag: boolean, optional, default: False,
                   fix diagonal of Gram to 1

    num_threads: int, optional, default: 1,,
                 number of threads to use for m2m and gram methods
                 (use 0 for 1 thread per core)

    """

    # API methods

    def __init__(self, name, libsvm_fmt=False, center=False, gamma=None,
                 use_rbf=False, constant_diag=False, num_threads=1):
        self.name = name
        self.libsvm_fmt = libsvm_fmt
        self.center = center
        self.gamma = gamma
        self.use_rbf = use_rbf
        self.constant_diag = constant_diag
        self.num_threads = num_threads
        self.cms_ = None  # array of column means over the Gram matrix
        self.mcm_ = None  # 1/n*mean of all the elements of the Gram matrix
        self.is_additive = False  # whether the kernel is additive or not

    @classmethod
    def is_kernel_for(cls, name):
        """ Abstract method to determine the link between name and class

        Needed for '__subclasses__()' to work in factory function
        """
        raise AbstractClassError(
            "Abstract kernel: use SparseKernel or DenseKernel instead")

    def v2v(self, vec1, vec2):
        """ Return kernel value between two vectors

        Parameters
        ----------
        vec1: (d, ) numpy array,
              a d-dimensional vector

        vec2: (d, ) numpy array,
              a d-dimensional vector

        Returns
        -------
        kern_val: float,
                  the kernel evaluation between vec1 and vec2

        Notes
        -----
        Abstract method that must be implemented by children classes.
        """
        raise AbstractClassError(
            "Abstract kernel: use SparseKernel or DenseKernel instead")

    def v2m(self, vec, mat):
        """ Return array of all kernel evaluations between a vector and a matrix

        Parameters
        ----------
        vec: (d, ) numpy array,
             a d-dimensional vector

        mat: (m, d) numpy array,
             m d-dimensional vectors stacked row-wise

        Returns
        -------
        kern_row: (m, ) numpy array,
                  the kernel evaluations between vec and all lines of mat
                  (contains additional first element if libsvm_fmt attribute
                  is set, and is centered if center attribute is set)

        Notes
        -----
        Default version: calls v2v repeatedly on the rows of mat
        """
        m = len(mat)
        offset = (self.libsvm_fmt and 1) or 0
        kern_row = np.ones(offset + m, dtype=DTYPE)
        kern_row[offset:] = [self.v2v(vec, vecy) for vecy in mat]
        # eventually center (in place) the kernel row in the feature space
        if self.center:
            self._center_rows(kern_row[offset:])
        return kern_row

    def m2m(self, mat1, mat2):
        """ Return the pairwise kernel evaluations between all vectors in mat1
        and all vectors in mat2

        Parameters
        ----------
        mat1: (m1, d) numpy array,
              m1 d-dimensional vectors stacked row-wise

        mat2: (m2, d) numpy array,
              m2 d-dimensional vectors stacked row-wise

        Returns
        -------
        kern_mat: (m1, m2) numpy array,
                  the kernel evaluations between all lines of mat1 and mat2
                  (contains additional first column if libsvm_fmt attribute
                  is set, and rows are centered if center attribute is set)

        Notes
        -----
        Default version: calls v2m repeatedly on the rows of mat1
        """
        # build the kernel matrix
        f = lambda i: self.v2m(mat1[i], mat2)
        kern_mat = np.array(
            mpmap(f, range(safe_len(mat1)), ncpus=self.num_threads),
            dtype=DTYPE)
        # v2m has already centered and added an extra 1st col
        # update 'id' col to contain line number (starts from 1 for libsvm)
        if self.libsvm_fmt:
            kern_mat[:, 0] = np.arange(1, kern_mat.shape[0] + 1)
        # eventually center (in place) the kernel rows in the feature space
        if self.center:
            offset = (self.libsvm_fmt and 1) or 0
            for _row in kern_mat:
                self._center_rows(_row[offset:])
        return kern_mat

    def gram(self, mat):
        """Return the kernel gram matrix between all vectors in mat

        Parameters
        ----------
        mat: (m, d) numpy array,
             m d-dimensional vectors stacked row-wise

        Returns
        -------
        kern_mat: (m, m) numpy array,
                  the kernel evaluations between all lines of mat1 and mat2
                  (contains additional first column if libsvm_fmt attribute
                  is set, and is centered if center attribute is set)

        Notes
        -----
        Default version: calls v2v repeatedly
        """
        n = safe_len(mat)
        # compute the kernel values
        f = lambda (i, j): (i, j, self.v2v(mat[i], mat[j]))
        if self.constant_diag:
            # don't compute the diag
            ijs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        else:
            ijs = [(i, j) for i in range(n) for j in range(i, n)]
        kern_vals = mpmap(f, ijs, ncpus=self.num_threads)
        # fill the kernel matrix
        kern_mat = np.ones((n, n), dtype=DTYPE)
        for i, j, kval in kern_vals:
            kern_mat[i, j] = kval
            kern_mat[j, i] = kval
        # eventually center (in place) the kernel matrix in the feature space
        if self.center:
            self._center_gram(kern_mat)  # additionnally sets centering params
        # make 'id' col to contain line number (index starts from 1 for libsvm)
        if self.libsvm_fmt:
            kern_mat = np.c_[np.arange(1, kern_mat.shape[0] + 1), kern_mat]
        return kern_mat

    # Internals

    def _center_gram(self, kern_mat):
        """ Center (in place) the Gram (or kernel) matrix in the feature space

        Mathematical operation: K <- PKP where P = eye(n) - 1/n ones((n,n))

        Additionally sets the self.cms_ (column means of the original kernel
        matrix) and self.mcm_ (mean of the original column means), which are
        parameters needed to center in the same way the future kernel
        evaluations

        """
        self.cms_, self.mcm_ = center_gram(kern_mat)

    def _center_rows(self, kern_rows):
        """ Center (in place) kernel rows in the feature space

        Assumes self.cms_ and self.mcm_ are already defined

        """
        if self.cms_ is None or self.mcm_ is None:
            raise ValueError('Training Gram matrix must be precomputed before '
                             'rows can be centered')
        center_rows(kern_rows, self.cms_, self.mcm_)
