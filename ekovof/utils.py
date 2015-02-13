""" Various utilities useful when using kernels
"""


import logging
from multiprocessing import cpu_count
import numpy as np
import numexpr as ne


def get_kernel_object(name, sparse=True, **kwargs):
    """ Factory function to get a kernel object
    """
    if sparse:
        from .sparse.sparse_kernels import SparseKernel
        return SparseKernel(name, **kwargs)
    else:
        from .dense.dense_kernels import get_dense_kernel
        return get_dense_kernel(name, **kwargs)


def get_heuristic_gamma(dist_vals, gamma=None, dist_is_libsvm_fmt=False):
    """ Return a heuristic gamma value for Gaussian RBF kernels

    Uses the median or mean of distance values
    """
    if gamma is None:
        gamma = -1

    if dist_is_libsvm_fmt:
        # make a view of dist_vals without the 1st column (ID)
        if dist_vals.ndim != 2:
            raise ValueError(
                'dist_vals is {}-D not in libsvm_fmt!'.format(dist_vals.ndim))
        dist_vals = dist_vals[:, 1:]

    if gamma <= 0:
        if np.prod(dist_vals.shape) > 1e6:
            # median uses a sorted copy of dist_vals: use mean when too large
            _sigma2 = dist_vals.mean()
        else:
            # Note: if NaN in dist_vals, median forgets about it,
            # whereas mean is NaN => force (ugly) check here
            if not np.alltrue(np.isfinite(dist_vals)):
                raise ValueError('NaN in the dist_vals')
            _sigma2 = np.median(dist_vals)
            if _sigma2 == 0:
                # may happen if many zeros (i.e many similar points)
                _sigma2 = dist_vals.mean()
        if _sigma2 == 0:
            logging.warning("constant kernel matrix: use gamma = 1")
            gamma = 1.0
        elif np.isfinite(_sigma2):
            if gamma < 0:
                gamma /= - float(_sigma2)
            else:
                gamma = 1.0 / _sigma2
        else:
            raise ValueError(
                'Invalid kernel values'
                ' yielding incorrect _sigma2 ({})'.format(_sigma2))

    return gamma


def safe_sparse_dot(a, b, dense_output=True):
    """Dot product that handles the sparse matrix case correctly

    Note: from sklearn.utils.extmath
    """
    from scipy import sparse
    if sparse.issparse(a) or sparse.issparse(b):
        ret = a * b  # matrix multiplication
        if dense_output and hasattr(ret, "toarray"):
            ret = ret.toarray()
        return ret
    else:
        return np.dot(a, b)


def safe_len(a):
    """ Length of array-like a (number of rows for 2D arrays)
    """
    try:
        return a.shape[0]
    except:
        return len(a)


def center_gram(kern_mat, is_sym=True):
    """ Center (in place) the Gram (kernel) matrix in the feature space

    Mathematical operation: K <- PKP where P = eye(n) - 1/n ones((n,n))

    Parameters
    ----------
    kern_mat: (nr, nc) numpy array,
              positve semi-definite kernel matrix

    is_sym: boolean, optional, default: True,
            assume the matrix is symmetric

    Returns
    -------
    cms: (1, nc) numpy array,
         column means of the original kernel matrix

    mcm: double,
         mean of the original column means, which, like cms, are parameters
         needed to center in the same way the future kernel evaluations

    """
    # number of rows and cols
    nr, nc = kern_mat.shape
    assert not is_sym or nr == nc, "Matrix cannot be symmetric if not square!"
    # mean of the columns of the original matrix (as (1, nc) row vector)
    cms = np.mean(kern_mat, 0)[np.newaxis, :]
    # mean of the rows (as (nr, 1) column vector)
    if is_sym:
        rms = cms.T
    else:
        rms = np.mean(kern_mat, 1)[:, np.newaxis]
    # mean of the means over columns (mean of the full matrix)
    mcm = np.mean(cms)
    # center the matrix (using array broadcasting)
    kern_mat += mcm
    kern_mat -= cms
    kern_mat -= rms
    return cms, mcm


def center_rows(kern_rows, cms, mcm):
    """ Center (in place) a kernel row in the feature space

    WARNING: assumes kernel row NOT IN LIBSVM FORMAT!

    Parameters
    ----------
    kern_rows: (m, n) numpy array,
               rows of kernel evaluations k(x,x_i) of m test samples x
               with a (training) set {x_i}, i=1...n

    cms: (1, nc) numpy array,
         column means of the original kernel matrix

    mcm: double,
         mean of the original column means
    """
    if kern_rows.ndim == 2:
        # multiple rows at once
        rows_mean = np.mean(kern_rows, axis=-1)[:, np.newaxis]
    else:
        # only one row: 1D vector
        rows_mean = np.mean(kern_rows)
        cms = cms.squeeze()  # to broadcast correctly
    kern_rows += mcm
    kern_rows -= cms
    kern_rows -= rows_mean


def kpca(centered_kern_mat, k=2):
    """ Perform kernel PCA

    The kernel has to be centered and not in libsvm format

    Parameters
    ----------
    centered_kern_mat: (n, n) numpy array,
                       CENTERED Gram array
                       (NOT in libsvm format and assumed centered!)

    k: int, optional, default: 2,
       number of (largest) principal components kept

    Returns
    -------
    eigvals: (k,) numpy array,
             k largest eigenvalues of the kernel matrix (same as covariance
             operator) sorted in ascending order

    neigvects: (n, k) numpy array,
               corresponding k NORMALIZED eigenvectors of the kernel matrix

    m_proj: (k, n) numpy array,
            columns of projections onto the k principal components

    Notes
    -----
    The residual (reconstruction error) can be obtained by doing:
    r = 1.0/n * (K.trace() - eigvals.sum()) = mean of the smallest n-k eigvals

    To project a new vector K[xtest, :] do: dot(K[xtest,:], neigvects).T

    """
    from scipy.linalg import eigh
    n = centered_kern_mat.shape[0]
    # k largest (eigen-value, eigen-vector) pairs of the kernel matrix
    eigvals, eigvects = eigh(centered_kern_mat, eigvals=(n - k, n - 1))
    # Note: ascending order
    sqrt_eigvals = np.sqrt(eigvals)
    # project the data onto the principal (normalized) eigen-vectors
    m_proj = eigvects * sqrt_eigvals
    # Note: equivalent to: dot(centered_kern_mat, eigvects/sqrt_eigvals)
    # normalize eigenvectors (useful for projecting new vectors)
    eigvects /= sqrt_eigvals
    # checks
    assert np.all(np.isfinite(eigvals)), \
        "Some Nans or Infs in the eigenvalues"
    assert np.all(np.isfinite(eigvects)), \
        "Some Nans or Infs in the normalized eigenvectors"
    assert np.all(np.isfinite(m_proj)), \
        "Some Nans or Infs in the projections"
    # return the results
    return eigvals, eigvects, m_proj.T


def kpca_proj(K_rows, neigvects):
    """ Project a sample using pre-computed kPCA (normalized) eigen-vectors

    Parameters
    ----------
    K_row: (m, n) numpy array,
           column vectors containing K[x, xtrains] where x is the sample to
           project and xtrains are the vectors used during kPCA

    neigvects: (n, k) numpy array,
               normalized principal eigen-vectors as returned by kpca

    Returns
    -------
    proj_col: (k, m) numpy array,
              column vectors corresponding to the projections on the principal
              components
    """
    return np.dot(K_rows, neigvects).T


def _get_kmat(dm2, w, gamma, sim_b):
    if sim_b:
        return ne.evaluate("w * dm2")
    else:
        return ne.evaluate("w * exp(-gamma*dm2)")


def combine_kernels(dist_matrices_arg, gamma=None,
                    is_sim_l=None, weights_l=None, libsvm_fmt=False):
    """ Returns a kernel matrix that is the sum of the kernel matrices for the
    different distances in dist_matrices_arg

    is_sim_l: list of booleans,
              stating if the distance matrix is actually a similarity (hence
              directly added) or not (in which case its RBF'd using gamma
              before adding it)

    if libsvm_fmt is True, then distances are assumed with an extra 1st column
    and the kernel is returned with an extra ID column for libsvm

    Note: the combination is just the average of kernels (no weights)
    """
    if gamma is None:
        gamma = -1

    if isinstance(dist_matrices_arg, (tuple, list)):
        # multiple matrices
        N = len(dist_matrices_arg)
        dist_matrices = dist_matrices_arg
    elif isinstance(dist_matrices_arg, np.ndarray):
        # only one matrix
        N = 1
        dist_matrices = [dist_matrices_arg]
    else:
        raise ValueError(
            "Invalid type for 'dist_matrices_arg' ({})".format(
                type(dist_matrices_arg)))

    if is_sim_l is None:
        is_sim_l = [False] * N
    if weights_l is None:
        weights_l = [1.0 / N] * N
    if len(is_sim_l) != N or len(weights_l) != N:
        raise ValueError(
            "Invalid combination parameter length "
            "(N={}, is_sim_l={}, weights_l={})".format(N, is_sim_l, weights_l))

    kernel_matrix = None
    for w, sim_b, dist_matrix in zip(weights_l, is_sim_l, dist_matrices):
        if w > 0:
            if (not sim_b) and gamma <= 0:
                gamma = get_heuristic_gamma(
                    dist_matrix, dist_is_libsvm_fmt=libsvm_fmt)
            if kernel_matrix is None:
                kernel_matrix = _get_kmat(dist_matrix, w, gamma, sim_b)
            else:
                kernel_matrix += _get_kmat(dist_matrix, w, gamma, sim_b)

    if libsvm_fmt:
        # set first column for libsvm
        kernel_matrix[:, 0] = np.arange(
            1, kernel_matrix.shape[0] + 1, dtype=kernel_matrix.dtype)

    return kernel_matrix


def mpmap(func, inputs, ncpus=0, with_progressbar=True):
    """Apply function 'func' to all inputs (any iterable)

    Use 'ncpus' processes -- defaults to ncores.

    Return list of results (in the order of 'inputs')

    Note: only worth it if more than 2 cpus
    (need one process to serialize/deserialize across channels)
    """
    ncpus = int(ncpus)
    tot_cpus = cpu_count()
    if ncpus == 0 or ncpus > tot_cpus:
        ncpus = tot_cpus
    elif ncpus < 0:
        ncpus = max(1, tot_cpus + ncpus)
    # not more processes than inputs
    ncpus = min(ncpus, len(inputs))
    # activate parallelism only if possible
    try:
        import pprocess
    except ImportError:
        logging.warning("Could not find pprocess module: no parallelism")
        ncpus = 1
    # launch the computations
    if ncpus >= 2:
        # version with reusable processes
        results = pprocess.Map(limit=ncpus, reuse=1, continuous=0)
        calc = results.manage(pprocess.MakeReusable(func))
        for arg in inputs:
            calc(arg)
        # store the results (same order as 'inputs')
        full_res = []
        if with_progressbar:
            from progressbar import ProgressBar
            idx_inputs = ProgressBar()(xrange(len(inputs)))
        else:
            idx_inputs = xrange(len(inputs))
        for _i in idx_inputs:
            full_res.append(results[_i])
    else:
        # use normal map
        full_res = map(func, inputs)
    return full_res
