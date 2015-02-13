"""
Efficient computation of kernels between between dense vectors of floats (double precision)
===========================================================================================

"""

from functools import wraps

import numpy as np
import numexpr as ne
from numpy import exp
from scipy import sparse


from .dense_distances import v2v_euclidean, v2m_euclidean, m2m_euclidean, \
        gram_euclidean, v2v_chisquare, v2m_chisquare, m2m_chisquare, gram_chisquare
from ..utils import get_heuristic_gamma, safe_sparse_dot
from ..generic_kernel import GenericKernel, DTYPE
from ..post_filtering import match_post_filter


def get_dense_kernel(name, **kwargs):
    """ Factory function to return the kernel corresponding to 'name'
    """
    for cls in GenericKernel.__subclasses__():
        if cls.is_kernel_for(name):
            return cls(name, **kwargs)
    raise ValueError("Unknown dense kernel (%s)" % name)


def get_point_kern(point_kern=None):
    """ Return the point kernel object (default: LinearKernel)

    Used by kernels between sets (i.e. whose v2v relies on point_kern.m2m)
    """
    if point_kern is None:
        point_kern = LinearKernel('linear')
    else:
        if not isinstance(point_kern, GenericKernel):
            raise ValueError(
                'point_kern is not a GenericKernel ({})'.format(
                    type(point_kern)))
    # wrap m2m such that it checks for 2D inputs
    point_kern.m2m = _check_2D_inputs(point_kern.m2m)  # TODO needed? costly?
    return point_kern


def _check_2D_inputs(func):
    """ Decorator to check inputs are 2D
    """
    @wraps(func)
    def check2D_func(*args, **kwargs):
        for i, arg in enumerate(args):
            if not (isinstance(arg, np.ndarray) or sparse.issparse(arg)):
                raise ValueError(
                    "Argument {} is not an array but a {}".format(
                        i, type(arg)))
            if arg.ndim != 2:
                raise ValueError(
                    "Argument {} has ndim == {} != 2".format(i, arg.ndim))
        return func(*args, **kwargs)
    return check2D_func


# =============================================================================
# Linear kernel
# =============================================================================


class LinearKernel(GenericKernel):
    """ Linear kernel (inner product) between dense vectors
    """

    def __init__(self, name, **okwargs):
        super(LinearKernel, self).__init__(name, **okwargs)
        self.is_additive = True

    @classmethod
    def is_kernel_for(cls, name):
        return name == 'linear'

    def v2v(self, vec1, vec2):
        """Inner product between two vectors
        """
        # Note: safe_sparse_dot does not work because no 1D sparse vector...
        if sparse.issparse(vec1):
            return vec1.multiply(vec2).sum()
        elif sparse.issparse(vec2):
            return vec2.multiply(vec1).sum()
        else:
            return np.dot(vec1, vec2)

    def v2m(self, vec, mat):
        """ Inner products between a vector and the rows of a matrix
        """
        kern_vals = safe_sparse_dot(vec, mat.T)
        # eventually center (in place) the kernel row in the feature space
        if self.center:
            self._center_rows(kern_vals)
        if self.libsvm_fmt:
            kern_vals = np.r_[1, kern_vals]
        return kern_vals

    def m2m(self, mat1, mat2):
        """ Return the pairwise kernel evaluations between all row vectors in
        mat1 and all row vectors in mat2
        """
        # build the kernel matrix
        kern_mat = safe_sparse_dot(mat1, mat2.T)
        # eventually center (in place) the kernel matrix in the feature space
        if self.center:
            self._center_rows(kern_mat)
        # make 'id' col to contain line number (index starts from 1 for libsvm)
        if self.libsvm_fmt:
            kern_mat = np.c_[np.arange(1, kern_mat.shape[0] + 1), kern_mat]
        return kern_mat

    def gram(self, mat):
        """Return the kernel gram matrix between all row vectors in mat

        """
        # build the kernel matrix
        kern_mat = safe_sparse_dot(mat, mat.T)
        # eventually center (in place) the kernel matrix in the feature space
        if self.center:
            self._center_gram(kern_mat)  # additionnally sets centering params
        # make 'id' col to contain line number (index starts from 1 for libsvm)
        if self.libsvm_fmt:
            kern_mat = np.c_[np.arange(1, kern_mat.shape[0] + 1), kern_mat]
        return kern_mat


# =============================================================================
# RBF kernels
# =============================================================================


def _efficient_exp(dist_mat, gamma):
    try:
        return ne.evaluate("exp(-%f*dist_mat)" % gamma)
    except:
        return np.exp(- gamma * dist_mat)


class RBFKernel(GenericKernel):
    """ Gaussian RBF kernel between dense vectors
    """

    def __init__(self, name, use_rbf=True, **okwargs):
        super(RBFKernel, self).__init__(name, **okwargs)
        self.use_rbf = use_rbf
        if name in ("rbf", "rbf_euclidean"):
            self._v2v = v2v_euclidean
            self._v2m = v2m_euclidean
            self._m2m = m2m_euclidean
            self._gram = gram_euclidean
        elif name == "rbf_chisquare":
            self._v2v = v2v_chisquare
            self._v2m = v2m_chisquare
            self._m2m = m2m_chisquare
            self._gram = gram_chisquare
        else:
            raise ValueError(
                "Unknown distance function for RBF kernels ({})".format(name))

    @classmethod
    def is_kernel_for(cls, name):
        return name.startswith("rbf")

    def v2v(self, vec1, vec2):
        """Return kernel value between two vectors
        """
        d = self._v2v(vec1, vec2)
        # eventually return rbf'd version
        return exp(-self.gamma * d) if self.use_rbf else d

    def v2m(self, vec, mat):
        """Return array of all kernel evaluations between a vector and a matrix
        """
        n = mat.shape[0]
        offset = (self.libsvm_fmt and 1) or 0
        kern_row = np.ones(offset + n, dtype=DTYPE)
        kern_row[offset:] = self._v2m(vec, mat)
        # eventually rbf it
        if self.use_rbf:
            kern_row[offset:] = exp(-self.gamma * kern_row[offset:])
        # eventually center (in place) the kernel row in the feature space
        if self.center:
            self._center_rows(kern_row[offset:])
        return kern_row

    def m2m(self, mtest, m):
        """ Return the pairwise kernel evaluations between all vectors in mtest
        and all vectors in m
        """
        n1 = mtest.shape[0]
        n2 = m.shape[0]
        kern_mat = self._m2m(mtest, m)
        # eventually rbf the gram matrix
        if self.use_rbf:
            self.gamma = get_heuristic_gamma(kern_mat, self.gamma)
            kern_mat = _efficient_exp(kern_mat, self.gamma)
        # eventually center (in place) the kernel matrix in the feature space
        if self.center:
            self._center_rows(kern_mat)
        # eventually put indexes for libsvm format
        if self.libsvm_fmt:
            id_col = np.arange(1, n1 + 1).reshape((n1, 1))
            kern_mat = np.hstack((id_col, kern_mat))
        return kern_mat

    def gram(self, m):
        """Return the kernel gram matrix between all vectors in m
        """
        n = m.shape[0]
        kern_mat = self._gram(m)
        # eventually rbf the gram matrix
        if self.use_rbf:
            self.gamma = get_heuristic_gamma(kern_mat, self.gamma)
            kern_mat = _efficient_exp(kern_mat, self.gamma)
        # eventually center (in place) the kernel matrix in the feature space
        if self.center:
            self._center_gram(kern_mat)  # additionnally sets centering params
        # eventually put indexes for libsvm format
        if self.libsvm_fmt:
            id_col = np.arange(1, n + 1).reshape((n, 1))
            kern_mat = np.hstack((id_col, kern_mat))
        return kern_mat


# =============================================================================
# Match kernel
# =============================================================================


def points_match_score(X, Y, point_kern):
    """ Average of best matches found in Y for each point in X and vice-versa

    Parameters
    ----------
    X: (n, d) array,
       the points for which we look for a match

    Y: (m, d) array,
       the points to which we want to match

    point_kern: GenericKernel instance,
                kernel between points (with a m2m method)

    Returns
    -------
    score: float,
           the matching score of X with Y and Y with X

    Notes
    -----
    We use a simple linear kernel as base kernel between points.
    Therefore, it is important that each row in X and Y is l2-normalized.
    """
    # get all dot products
    XYT = point_kern.m2m(X, Y)
    # get the match score from X to Y
    mxy = np.mean(np.max(XYT, axis=1))
    # get the match score from Y to X
    myx = np.mean(np.max(XYT, axis=0))
    # return the mean of the match scores in both directions
    return 0.5 * (mxy + myx)


class MatchKernel(GenericKernel):
    """ Match Kernel between sets of dense vectors

    Parameters
    ----------
    point_kern: GenericKernel instance, optional, default: None (use linear)
                kernel between points (with a m2m method)

    """

    def __init__(self, name, point_kern=None, **okwargs):
        super(MatchKernel, self).__init__(name, **okwargs)
        self.use_rbf = False
        self.point_kern = get_point_kern(point_kern)

    @classmethod
    def is_kernel_for(cls, name):
        return name == "match"

    def v2v(self, X, Y):
        return points_match_score(X, Y, self.point_kern)


# =============================================================================
# PostFilterMatchKernel
# =============================================================================


def get_top_n_post_filter_match_score(Px, Py, XYT, top_n, dims, mq):
    n, m = XYT.shape
    # get the top_n matches and their scores
    mxy_idxs = np.argsort(XYT, axis=1)[:, -top_n:]
    mxy_scores = np.array([XYT[i, mxy_idxs[i]] for i in range(n)]).ravel()
    mxy_idxs = mxy_idxs.ravel()
    # duplicate the positions of X top_n times
    Px = Px[np.vstack([np.arange(n) for p in range(top_n)]).T.ravel()]
    # filter out the matches from X to Y
    bx_idxs = match_post_filter(Px, Py, mxy_idxs, dims=dims, mq=mq)
    # sum over the kept pairs and normalize by original number of matches
    mxy = mxy_scores[bx_idxs].sum() / float(n * top_n)  # TODO try real mean?
    return mxy


class PostFilterMatchKernel(GenericKernel):
    """ Match Kernel between sets of dense vectors

    Parameters
    ----------
    point_kern: GenericKernel instance, optional, default: None (use linear)
                kernel between points (with a m2m method)

    dims: iterable, optional, default: None (all),
          list of indexes of subset of position dimensions to use for filtering

    mq: int, optional, default: 3,
        number of neighbor to use for bandwidth estimation

    top_n: int, optional, default: 1,
           use the top_n best matches only

    """

    def __init__(self, name, point_kern=None,
                 dims=None, mq=3, top_n=1, **okwargs):
        super(PostFilterMatchKernel, self).__init__(name, **okwargs)
        self.use_rbf = False
        self.point_kern = get_point_kern(point_kern)
        self.dims = dims
        self.mq = mq
        self.top_n = top_n

    @classmethod
    def is_kernel_for(cls, name):
        return name == "postfiltermatch"

    def v2v(self, Xp, Yp):
        """ Average of the top_n best matches passing a post-filtering step

        Parameters
        ----------
        Xp: pair of (n, d) and (n, dp) arrays,
            the points for which we look for a match and their positions

        Yp: pair of (m, d) and (m, dp) arrays,
            the points we want to match with and their positions

        Returns
        -------
        score: float,
               the matching score of X with Y and Y with X obtained
               after post-filtering the matches
        """
        X, Px = Xp
        Y, Py = Yp
        # get all dot products
        XYT = self.point_kern.m2m(X, Y)
        # get the score from X to Y
        mxy = get_top_n_post_filter_match_score(
            Px, Py, XYT, self.top_n, self.dims, self.mq)
        # get the score from X to Y
        myx = get_top_n_post_filter_match_score(
            Py, Px, XYT.T, self.top_n, self.dims, self.mq)
        # return the mean of the match scores in both directions
        mscore = 0.5 * (mxy + myx)
        #print mxy, myx, mscore
        return mscore


# =============================================================================
# TopMatchKernel
# =============================================================================


# TODO
def points_topn_match_score(X, Y, point_kern):
    """ Average of top_n best matches in Y for each point in X and vice-versa

    Parameters
    ----------
    X: (n, d) array,
       the points for which we look for a match

    Y: (m, d) array,
       the points to which we want to match

    point_kern: GenericKernel instance,
                kernel between points (with a m2m method)

    Returns
    -------
    score: float,
           the matching score of X with Y and Y with X

    Notes
    -----
    We use a simple linear kernel as base kernel between points.
    Therefore, it is important that each row in X and Y is l2-normalized.
    """
    # get all dot products
    XYT = point_kern.m2m(X, Y)
    # get the match score from X to Y
    mxy = np.mean(np.max(XYT, axis=1))
    # get the match score from Y to X
    myx = np.mean(np.max(XYT, axis=0))
    # return the mean of the match scores in both directions
    return 0.5 * (mxy + myx)


class TopMatchKernel(GenericKernel):
    """ Match Kernel using the top n best matches instead of just the max

    Parameters
    ----------
    point_kern: GenericKernel instance, optional, default: None (use linear)
                kernel between points (with a m2m method)

    top_n: int, optional, default: 3,
           use the top_n best matches only

    """

    def __init__(self, name, point_kern=None, top_n=3, **okwargs):
        raise ValueError('topmatch kernel not implemented yet')
        super(TopMatchKernel, self).__init__(name, **okwargs)
        self.use_rbf = False
        self.point_kern = get_point_kern(point_kern)
        self.top_n = top_n

    @classmethod
    def is_kernel_for(cls, name):
        return name == "topmatch"

    def v2v(self, X, Y):
        return points_topn_match_score(X, Y, self.point_kern, self.top_n)


# =============================================================================
# SoftMatch kernel
# =============================================================================


class SoftMatch(GenericKernel):
    """ Soft-max of the similarity of all pairs between sets of dense vectors

    exp(-gamma*(smd)), where the soft-max dissimilarity is
        smd = (sm(X, X) + sm(Y, Y)) / 2 - sm(X, Y)
    and the soft-max similarity is
        sm(X, Y) = log(mean(exp(point_kern.m2m(X,Y))))
    """

    def __init__(self, name, point_kern=None, gamma=1., **okwargs):
        super(SoftMatch, self).__init__(name, **okwargs)
        self.gamma = gamma
        self.use_rbf = True
        self.point_kern = get_point_kern(point_kern)

    @classmethod
    def is_kernel_for(cls, name):
        return name == "softmatch"

    def v2v(self, X, Y):
        """ Return the mean of the similarity between all pairs of rows of X and Y
        """
        # softmax of all dot products  (TODO: use sum instead of mean?)
        logm = lambda K: np.log(np.mean(np.exp(K)))
        k11 = logm(self.point_kern.gram(X))
        k22 = logm(self.point_kern.gram(Y))
        k12 = logm(self.point_kern.m2m(X, Y))
        kval = np.exp(-self.gamma * (0.5 * (k11 + k22) - k12))
        return kval


# =============================================================================
# AllPairs kernel
# =============================================================================


class AllPairs(GenericKernel):
    """ Mean of the similarity of all pairs between sets of dense vectors
    """

    def __init__(self, name, point_kern=None, **okwargs):
        super(AllPairs, self).__init__(name, **okwargs)
        self.use_rbf = False
        self.point_kern = get_point_kern(point_kern)

    @classmethod
    def is_kernel_for(cls, name):
        return name == "allpairs"

    def v2v(self, X, Y):
        """ Return the mean of the similarity between all pairs of rows of X and Y
        """
        # average all dot products
        kval = np.mean(self.point_kern.m2m(X, Y))
        return kval


# =============================================================================
# WeightedPairs kernel
# =============================================================================


class WeightedPairs(GenericKernel):
    """ Weighted mean of the similarity of all pairs
    between sets of dense vectors

    Each sample is a pair Xw == (X, w) where X contains the set of features
    and w the per-row weights (1D vector)
    """

    def __init__(self, name, point_kern=None, **okwargs):
        super(WeightedPairs, self).__init__(name, **okwargs)
        self.use_rbf = False
        self.point_kern = get_point_kern(point_kern)

    @classmethod
    def is_kernel_for(cls, name):
        return name == "weightedpairs"

    def v2v(self, Xw, Yw):
        """ Return the mean of the similarity between all pairs of rows of X and Y
        """
        X, wx = Xw
        Y, wy = Yw
        pkvals = self.point_kern.m2m(X, Y)
        # TODO check dims instead of dummy float conversion?
        kval = float(np.dot(wx, np.dot(pkvals, wy)))
        return kval
