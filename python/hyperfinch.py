import time
import argparse
import numpy as np
from sklearn import metrics
import scipy.sparse as sp
import warnings

try:
    from pyflann import *

    pyflann_available = True
except Exception as e:
    warnings.warn('pyflann not installed: {}'.format(e))
    pyflann_available = False
    pass

from hyperbolic_utils import frechet_mean, np_dist

FLANN_THRESHOLD = 70000


def clust_rank(mat, initial_rank=None, distance='cosine'):
    s = mat.shape[0]
    if initial_rank is not None:
        orig_dist = []
    elif s <= FLANN_THRESHOLD:
        orig_dist = metrics.pairwise.pairwise_distances(mat, mat, metric=distance)
        np.fill_diagonal(orig_dist, 1e12)
        initial_rank = np.argmin(orig_dist, axis=1)
    else:
        if not pyflann_available:
            raise MemoryError("You should use pyflann for inputs larger than {} samples.".format(FLANN_THRESHOLD))
        print('Using flann to compute 1st-neighbours at this step ...')
        flann = FLANN()
        result, dists = flann.nn(mat, mat, num_neighbors=2, algorithm="kdtree", trees=8, checks=128)
        initial_rank = result[:, 1]
        orig_dist = []
        print('Step flann done ...')

    # The Clustering Equation
    A = sp.csr_matrix((np.ones_like(initial_rank, dtype=np.float32), (np.arange(0, s), initial_rank)), shape=(s, s))
    A = A + sp.eye(s, dtype=np.float32, format='csr')
    A = A @ A.T

    A = A.tolil()
    A.setdiag(0)
    return A, orig_dist


def get_clust(a, orig_dist, min_sim=None):
    if min_sim is not None:
        a[np.where((orig_dist * a.toarray()) > min_sim)] = 0

    num_clust, u = sp.csgraph.connected_components(csgraph=a, directed=True, connection='weak', return_labels=True)
    return u, num_clust


def cool_mean(M, u, hyperbolic=False):
    if hyperbolic:
        # Sort M by cluster label
        _, nf = np.unique(u, return_counts=True)
        idx = np.argsort(u)
        M = M[idx, :]

        cnf = np.cumsum(nf)
        cnf = np.insert(cnf, 0, 0)
        cluster_means = []
        for i in range(len(cnf) - 1):
            cluster_mean = frechet_mean(M[cnf[i]:cnf[i + 1]])
            cluster_means.append(cluster_mean)
        M = np.array(cluster_means)
    else:
        _, nf = np.unique(u, return_counts=True)
        idx = np.argsort(u)
        M = M[idx, :]
        M = np.vstack((np.zeros((1, M.shape[1])), M))

        np.cumsum(M, axis=0, out=M)
        cnf = np.cumsum(nf)
        nf1 = np.insert(cnf, 0, 0)
        nf1 = nf1[:-1]

        M = M[cnf, :] - M[nf1, :]
        M = M / nf[:, None]
    return M


def get_merge(c, u, data, hyperbolic=False):
    if len(c) != 0:
        _, ig = np.unique(c, return_inverse=True)
        c = u[ig]
    else:
        c = u

    mat = cool_mean(data, c, hyperbolic=hyperbolic)
    return c, mat


def update_adj(adj, d):
    # Update adj, keep one merge at a time
    idx = adj.nonzero()
    v = np.argsort(d[idx])
    v = v[:2]
    x = [idx[0][v[0]], idx[0][v[1]]]
    y = [idx[1][v[0]], idx[1][v[1]]]
    a = sp.lil_matrix(adj.get_shape())
    a[x, y] = 1
    return a


def req_numclust(c, data, req_clust, distance, hyperbolic=False):
    iter_ = len(np.unique(c)) - req_clust
    c_, mat = get_merge([], c, data, hyperbolic=hyperbolic)
    for i in range(iter_):
        if hyperbolic:
            adj, orig_dist = clust_rank(mat, initial_rank=None, distance=np_dist)
        else:
            adj, orig_dist = clust_rank(mat, initial_rank=None, distance=distance)
        adj = update_adj(adj, orig_dist)
        u, _ = get_clust(adj, [], min_sim=None)
        c_, mat = get_merge(c_, u, data, hyperbolic=False)
    return c_


def FINCH(data, initial_rank=None, req_clust=None, distance='cosine', ensure_early_exit=True, verbose=True):
    """ FINCH clustering algorithm.
    :param data: Input matrix with features in rows.
    :param initial_rank: Nx1 first integer neighbor indices (optional).
    :param req_clust: Set output number of clusters (optional). Not recommended.
    :param distance: One of ['cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan'] Recommended 'cosine'.
    :param ensure_early_exit: [Optional flag] may help in large, high dim datasets, ensure purity of merges and helps early exit
    :param verbose: Print verbose output.
    :return:
            c: NxP matrix where P is the partition. Cluster label for every partition.
            num_clust: Number of clusters.
            req_c: Labels of required clusters (Nx1). Only set if `req_clust` is not None.

    The code implements the FINCH algorithm described in our CVPR 2019 paper
        Sarfraz et al. "Efficient Parameter-free Clustering Using First Neighbor Relations", CVPR2019
         https://arxiv.org/abs/1902.11266
    For academic purpose only. The code or its re-implementation should not be used for commercial use.
    Please contact the author below for licensing information.
    Copyright
    M. Saquib Sarfraz (saquib.sarfraz@kit.edu)
    Karlsruhe Institute of Technology (KIT)
    """
    # Cast input data to float32
    data = data.astype(np.float32)
    
    if distance == 'hyperbolic':
        hyperbolic = True
    else:
        hyperbolic = False

    min_sim = None
    if distance in ['cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan']:
        adj, orig_dist = clust_rank(data, initial_rank, distance)
    elif hyperbolic:
        adj, orig_dist = clust_rank(data, initial_rank, np_dist)
    initial_rank = None
    group, num_clust = get_clust(adj, [], min_sim)
    c, mat = get_merge([], group, data, hyperbolic=hyperbolic)

    if verbose:
        print('Partition 0: {} clusters'.format(num_clust))

    if ensure_early_exit:
        if len(orig_dist) != 0:
            min_sim = np.max(orig_dist * adj.toarray())

    exit_clust = 2
    c_ = c
    k = 1
    num_clust = [num_clust]

    while exit_clust > 1:
        if distance in ['cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan']:
            adj, orig_dist = clust_rank(data, initial_rank, distance)
        elif hyperbolic:
            adj, orig_dist = clust_rank(data, initial_rank, np_dist)
        u, num_clust_curr = get_clust(adj, orig_dist, min_sim)
        c_, mat = get_merge(c_, u, data, hyperbolic=hyperbolic)

        num_clust.append(num_clust_curr)
        c = np.column_stack((c, c_))
        exit_clust = num_clust[-2] - num_clust_curr

        if num_clust_curr == 1 or exit_clust < 1:
            num_clust = num_clust[:-1]
            c = c[:, :-1]
            break

        if verbose:
            print('Partition {}: {} clusters'.format(k, num_clust[k]))
        k += 1

    if req_clust is not None:
        if req_clust not in num_clust:
            ind = [i for i, v in enumerate(num_clust) if v >= req_clust]
            req_c = req_numclust(c[:, ind[-1]], data, req_clust, distance, hyperbolic=hyperbolic)
        else:
            req_c = c[:, num_clust.index(req_clust)]
    else:
        req_c = None

    return c, num_clust, req_c


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', required=True, help='Specify the path to your data csv file.')
    parser.add_argument('--output-path', default=None, help='Specify the folder to write back the results.')
    parser.add_argument('--dist-fn', default='cosine', help='Specify the distance function to use')
    args = parser.parse_args()
    data = np.genfromtxt(args.data_path, delimiter=",").astype(np.float32)
    start = time.time()
    c, num_clust, req_c = FINCH(data, initial_rank=None, req_clust=None, distance=args.dist_fn, ensure_early_exit=True, verbose=True)
    print('Time Elapsed: {:2.2f} seconds'.format(time.time() - start))

    # Write back
    if args.output_path is not None:
        print('Writing back the results on the provided path ...')
        np.savetxt(args.output_path + '/c.csv', c, delimiter=',', fmt='%d')
        np.savetxt(args.output_path + '/num_clust.csv', np.array(num_clust), delimiter=',', fmt='%d')
        if req_c is not None:
            np.savetxt(args.output_path + '/req_c.csv', req_c, delimiter=',', fmt='%d')
    else:
        print('Results are not written back as the --output-path was not provided')


if __name__ == '__main__':
    main()
