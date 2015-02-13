"""
Testing kernels from the `ekovof` module

"""

import numpy as np
from scipy import sparse

from ..utils import get_kernel_object, get_heuristic_gamma, combine_kernels
from ..sparse.sparse_kernels import AVAILABLE_KERNELS, RBF_KERNELS

from nose.tools import assert_true, assert_raises
from nose.plugins.skip import SkipTest


###############################################################################
# Set up test requirements
###############################################################################


# decorator to skip failed tests
def skip_failed(func):
    def new_func(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except:
            raise SkipTest()


dmat_size = 5

n_features = 42

n_samples_X_vec = 3
n_samples_Y_vec = 5

set_size1 = 7
set_size2 = 9

n_samples_X_set = 11
n_samples_Y_set = 13

np.random.seed(1)

vec1 = np.random.random((n_features, )).astype('f8')
vec2 = np.random.random((n_features, )).astype('f8')
X_vecs = np.random.random((n_samples_X_vec, n_features)).astype('f8')
Y_vecs = np.random.random((n_samples_Y_vec, n_features)).astype('f8')

svec1 = sparse.csr_matrix(vec1)
svec2 = sparse.csr_matrix(vec2)
sX_vecs = sparse.csr_matrix(X_vecs)
sY_vecs = sparse.csr_matrix(Y_vecs)

set1 = np.random.random((set_size1, n_features)).astype('f8')
set2 = np.random.random((set_size2, n_features)).astype('f8')
X_sets = [np.random.random((np.random.randint(1, 7), n_features)).astype('f8')
          for i in range(n_samples_X_set)]
Y_sets = [np.random.random((np.random.randint(11, 15), n_features)).astype('f8')
          for i in range(n_samples_Y_set)]


def safe_len(array_like):
    if sparse.issparse(array_like):
        # len is ambiguous for sparse matrices: explicitly get number of rows
        return array_like.shape[0]
    else:
        return len(array_like)


def check_kernel(kern, X, Y, x, y):
    # check gram (sets gamma if necessary)
    vals = kern.gram(X)
    assert_true(np.alltrue(np.isfinite(vals)))
    assert_true(not kern.use_rbf or kern.gamma > 0)
    if kern.libsvm_fmt:
        assert_true(vals.shape == (safe_len(X), safe_len(X) + 1))
        assert_true(np.alltrue(vals[:, 0] == np.arange(1, safe_len(X) + 1)))
    # check m2m
    vals = kern.m2m(Y, X)  # order is important when centering!
    assert_true(np.alltrue(np.isfinite(vals)))
    if kern.libsvm_fmt:
        assert_true(vals.shape == (safe_len(Y), safe_len(X) + 1))
        assert_true(np.alltrue(vals[:, 0] == np.arange(1, safe_len(Y) + 1)))
    # check v2v
    vals = kern.v2v(x, y)
    assert_true(np.alltrue(np.isfinite(vals)))
    # check v2m
    vals = kern.v2m(y, X)  # order is important when centering!
    assert_true(np.alltrue(np.isfinite(vals)))
    if kern.libsvm_fmt:
        assert_true(vals[0] == 1 and vals.shape == (safe_len(X) + 1, ))


###############################################################################
# Test utility functions
###############################################################################


def test_heuristic_gamma():
    ones_1d = np.ones((10, ))
    ones_2d = np.ones((10, 5))
    gamma = get_heuristic_gamma(ones_1d)
    assert_true(gamma == 1)
    gamma = get_heuristic_gamma(ones_2d)
    assert_true(gamma == 1)
    gamma = get_heuristic_gamma(ones_2d, dist_is_libsvm_fmt=True)
    assert_true(gamma == 1)
    assert_raises(ValueError,
                  get_heuristic_gamma, ones_1d, dist_is_libsvm_fmt=True)
    for gamma in (None, 0, -1, 1):
        gamma = get_heuristic_gamma(ones_2d, gamma=gamma)
        assert_true(gamma == 1)
    gamma = get_heuristic_gamma(ones_2d, gamma=-2)
    assert_true(gamma == 2)
    gamma = get_heuristic_gamma(ones_2d, gamma=2)
    assert_true(gamma == 2)
    assert_raises(ValueError,
                  get_heuristic_gamma, np.array([1, np.nan, 1]))
    assert_raises(ValueError,
                  get_heuristic_gamma, np.array([1, np.inf, 1]))


def _dmat():
    return 2 * np.ones((dmat_size, dmat_size)) - np.eye(dmat_size)


def _smat():
    return np.eye(dmat_size)


def test_combine_kernels():
    # with a single distance matrix
    assert_true(np.alltrue(np.isfinite(combine_kernels(_dmat()))))
    # with a single similarity matrix
    ckmat = combine_kernels(_smat(), is_sim_l=[True])
    assert_true(np.alltrue(np.isfinite(ckmat)))
    # with multiple distance matrices
    dmats = [_dmat() for i in range(3)]
    ckmat = combine_kernels(dmats)
    assert_true(np.alltrue(np.isfinite(ckmat)))
    # with a multiple similarity matrices
    ckmat = combine_kernels([_smat(), _smat()], is_sim_l=[True, True])
    assert_true(np.alltrue(np.isfinite(ckmat)))
    # with a mixed dist/sim matrices
    ckmat = combine_kernels([_dmat(), _smat()], is_sim_l=[False, True])
    assert_true(np.alltrue(np.isfinite(ckmat)))
    # with weights
    ws = range(len(dmats))
    ckmat = combine_kernels(dmats, weights_l=ws)
    assert_true(np.alltrue(np.isfinite(ckmat)))
    # with invalid weights
    assert_raises(ValueError, combine_kernels, dmats, weights_l=range(10))


###############################################################################
# Test dense kernels
###############################################################################


dense_kern_vecs = (
    "linear",
    "rbf_chisquare",
    "rbf_euclidean",
)


dense_kern_sets = (
    "match",
#    "postfiltermatch",  # special input
#    "topmatch",  # not implemented yet
    "softmatch",
    "allpairs",
#    "weightedpairs",  # special input
)


def test_linear_dense_kernel_dense_inputs():
    # get the inputs
    x = vec1
    y = vec2
    X = X_vecs
    Y = Y_vecs
    # try loading kernel
    kern = get_kernel_object('linear', sparse=False)
    # check gram
    vals = kern.gram(X)
    assert_true(np.alltrue(vals == np.dot(X, X.T)))
    # check m2m
    vals = kern.m2m(Y, X)
    assert_true(np.alltrue(vals == np.dot(Y, X.T)))
    # check v2v
    vals = kern.v2v(x, y)
    assert_true(np.alltrue(vals == np.dot(x, y)))
    # check v2m
    vals = kern.v2m(y, X)
    assert_true(np.alltrue(vals == np.dot(y, X.T)))


def test_linear_dense_kernel_sparse_inputs():
    # try loading kernel
    kern = get_kernel_object('linear', sparse=False)
    # check gram
    vals = kern.gram(sX_vecs)
    assert_true(np.allclose(vals, np.dot(X_vecs, X_vecs.T)))
    # check m2m
    vals = kern.m2m(sY_vecs, sX_vecs)
    assert_true(np.allclose(vals, np.dot(Y_vecs, X_vecs.T)))
    # check v2v
    vals = kern.v2v(svec1, svec2)
    assert_true(np.allclose(vals, np.dot(vec1, vec2)))
    # check v2m
    vals = kern.v2m(svec2, sX_vecs)
    assert_true(np.allclose(vals, np.dot(vec2, X_vecs.T)))


# use generators to try each kernel and configuration
def check_dense_kernel(name, on_sets,
                       libsvm_fmt=False, center=False, num_threads=1):
    # get the inputs
    if on_sets:
        x = set1
        y = set2
        X = X_sets
        Y = Y_sets
    else:
        x = vec1
        y = vec2
        X = X_vecs
        Y = Y_vecs
    # try loading kernel
    kern = get_kernel_object(name, sparse=False,
                             libsvm_fmt=libsvm_fmt,
                             center=center,
                             num_threads=num_threads)
    assert_true(kern.name == name)
    assert_true(kern.libsvm_fmt == libsvm_fmt)
    assert_true(kern.center == center)
    assert_true(kern.num_threads == num_threads)
    # perform various checks (cf definition at module top-level)
    check_kernel(kern, X, Y, x, y)


def test_dense_kernel_vecs_factory():
    for name in dense_kern_vecs:
        yield check_dense_kernel, name, False
        yield check_dense_kernel, name, False, True  # with libsvm_fmt
        yield check_dense_kernel, name, False, False, True  # with center
        yield check_dense_kernel, name, False, True, True  # with both
        yield check_dense_kernel, name, False, False, False, 2  # in parallel


def test_dense_kernel_sets_factory():
    for name in dense_kern_sets:
        yield check_dense_kernel, name, True
        yield check_dense_kernel, name, True, True  # with libsvm_fmt
        yield check_dense_kernel, name, True, False, True  # with center
        yield check_dense_kernel, name, True, True, True  # with both
        yield check_dense_kernel, name, True, False, False, 2  # in parallel


###############################################################################
# Test sparse kernels
###############################################################################


def check_sparse_kernel(name, on_sets,
                        libsvm_fmt=False, center=False, num_threads=1):
    # get the inputs
    x = svec1
    y = svec2
    X = sX_vecs
    Y = sY_vecs
    # try loading kernel
    kern = get_kernel_object(name,
                             libsvm_fmt=libsvm_fmt,
                             center=center,
                             num_threads=num_threads)
    assert_true(kern.name == name)
    assert_true(kern.libsvm_fmt == libsvm_fmt)
    assert_true(kern.center == center)
    assert_true(kern.num_threads == num_threads)
    assert_true(kern.use_rbf == (name in RBF_KERNELS))
    # perform various checks (cf definition at module top-level)
    check_kernel(kern, X, Y, x, y)


def test_sparse_kernel_vecs_factory():
    for name in AVAILABLE_KERNELS:
        yield check_sparse_kernel, name, False
        yield check_sparse_kernel, name, False, True  # with libsvm_fmt
        yield check_sparse_kernel, name, False, False, True  # with center
        yield check_sparse_kernel, name, False, True, True  # with both
        yield check_sparse_kernel, name, False, False, False, 2  # in parallel
