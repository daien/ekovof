""" Module to post-filter match by estimating a simple transformation and
removing matches deviating too much from this transformation
"""

import sys
import numpy as np

from sklearn.cluster.mean_shift_ import mean_shift, estimate_bandwidth


def ucounts(values):
    """ Count occurences of each unique element in values

    Parameters
    ----------
    values: list or array,
            the values to count

    Returns
    -------
    uvals: numpy array,
           the unique values

    counts: numpy array,
            the per-unique value counts
    """
    svalues = np.sort(values)
    diff = np.concatenate(([1], np.diff(svalues)))
    idx = np.concatenate((np.where(diff)[0], [len(svalues)]))
    uvals = svalues[idx[:-1]]
    counts = np.diff(idx)
    return uvals, counts


def get_meanshift_mode_members(data, mq=3, show_modes=False):
    """ Indexes of the rows of data which are closest to the dominant mode

    We adaptively estimates the bandwidth as the average distance to the
    mq^th neighbor

    Parameters
    ----------
    data: (n_points, n_dims) array,
          the data sampled from the distribution of which we try to find the mode

    mq: int, optional, default: 3
        the neighbor number used to estimate the bandwidth

    Returns
    -------
    mode_idxs: (n_modal_points, ) array,
               the indexes (rows of data) of the points in the dominant mode
    """
    n = data.shape[0]
    mq = min(mq, n)
    quantile = float(mq) / n
    try:
        bandwidth = estimate_bandwidth(data, quantile=quantile)
        modes, assignments = mean_shift(data, bandwidth=bandwidth)
        # find the dominant mode
        mcounts = ucounts(assignments)[1]
        dmi = np.argmax(mcounts)
        # retrieve the samples assigned to the mode
        mode_idxs = np.argwhere(assignments == dmi).squeeze()
        # might fail with TypError if mode of length 1!
        assert len(mode_idxs) > 0
        if show_modes:
            print "Found %d modes (bandwidth=%g)" % (len(modes), bandwidth)
            mcounts = ucounts(assignments)[1]
            plot_modes(mcounts, modes)
    except TypeError:
        # failure case: keep none of the matches
        sys.stderr.write("Warning: failed to compute modes, no matches kept\n")
        mode_idxs = []
    #print "Post-filtering kept %d/%d = %0.3f tubes" % \
    #        (len(mode_idxs), n, len(mode_idxs) / float(n))
    return mode_idxs


def get_histogram_mode_members(data, bins=3, show_modes=False):
    """ Indexes of the rows of data which are closest to the dominant mode

    Parameters
    ----------
    data: (n_points, n_dims) array,
          the data sampled from the distribution of which we try to find the mode

    bins: int or list, optional, default: 3
          number of bins for quantization along each dimension

    Returns
    -------
    mode_idxs: (n_modal_points, ) array,
               the indexes (rows of data) of the points in the dominant mode
    """
    n, n_dims = data.shape
    # the bin edges
    if isinstance(bins, int):
        bedges = [np.linspace(-1, 1, bins + 1) for d in range(n_dims)]
    else:
        bedges = [np.linspace(-1, 1, bins[d] + 1) for d in range(n_dims)]
    # get the maximum bin index + 1
    mbi = max(len(be) for be in bedges)
    # compute per-dim ass. and represent full ass. as sum(assign_d * 10**d)
    # TODO use multidim_digitize instead?
    assignments = np.digitize(data[:, 0], bedges[0]) - 1
    for d in range(1, n_dims):
        assignments += (np.digitize(data[:, d], bedges[d]) - 1) * mbi ** d
    uass, mcounts = ucounts(assignments)
    # find the dominant mode
    dass = uass[np.argmax(mcounts)]
    # retrieve the samples assigned to the mode
    mode_idxs = np.argwhere(assignments == dass).T[0]
    assert len(mode_idxs) > 0
    if show_modes:
        # the bin centers
        bcenters = [bedges[d][:-1] + np.diff(bedges[d], axis=0) *
                    0.5 for d in range(n_dims)]
        # get the per-dim indexes of the non-zero bins
        get_d = lambda xs, d: [int(
            np.base_repr(x, base=mbi, padding=n_dims)[-(d + 1)]) for x in xs]
        modes = np.array(
            [bcenters[d][get_d(uass, d)] for d in range(n_dims)]).T
        plot_modes(mcounts, modes, bins=bins)
        print "Post-filtering kept %d/%d = %0.3f tubes" % \
            (len(mode_idxs), n, len(mode_idxs) / float(n))
    return mode_idxs


def plot_modes(mcounts, modes, bins=None):
    """ Display the histogram of modes (up to 3dim data)
    """
    import pylab as pl
    ndims = modes.shape[1]
    if ndims == 1:
        # plot simple 1D histogram use to determine width
        modes = modes.squeeze()
        width = 2.0 / bins
        pl.bar(modes - 0.5 * width, mcounts, alpha=0.5, width=width)
    elif ndims == 2:
        # 2D histogram: show the heat-map
        pl.scatter(modes[:, 0], modes[:, 1], s=10 * mcounts)
    elif ndims == 3:
        # 3D plot
        from mpl_toolkits.mplot3d import Axes3D
        fig = pl.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(
            modes[:, 0], modes[:, 2], modes[:, 1], s=100 * mcounts, alpha=0.8)
        ax.set_xlabel('X')
        ax.set_ylabel('T')
        ax.set_zlabel('Y')
    else:
        raise ValueError("Cannot display data > 3D")


def match_post_filter(pos1, pos2, m12_idxs,
                      dims=None, algo="hist", mq=3, show_modes=False):
    """ Filter out matches from 1 to 2 with incoherent position difference

    Parameters
    ----------
    pos1: (n_pts_1, d),
          the d-dimensional positions of the first set of points

    pos2: (n_pts_2, d),
          the d-dimensional positions of the second set of points

    m12_idxs: (n_pts_1, ),
              the index in the second set of the match of each point
              from the first set

    dims: list, optional, default: None,
          the subset of dimensions to use (if None, use all)

    algo: string, optional, default: "hist",
          the mode-seeking algo: hist or meanshift

    mq: int, optional, default: 3
        if hist: the number of bins per dimension (can be a list)
        if meanshift: the rank of the neighbor to estimate the bandwidth

    Returns
    -------
    bidxs1: (n_modal_points, ) array,
            the indexes (rows of pos1) of the points in the dominant mode
    """
    # get the position differences between each match
    if dims is None or len(dims) == 0:
        deltas = pos1 - pos2[m12_idxs]
    else:
        deltas = pos1[:, dims] - pos2[m12_idxs][:, dims]
    # find the tubes 1 that fall in the bin of the dominant mode
    if algo == "hist":
        bidxs1 = get_histogram_mode_members(
            deltas, bins=mq, show_modes=show_modes)
    else:
        bidxs1 = get_meanshift_mode_members(
            deltas, mq=mq, show_modes=show_modes)
    return bidxs1
