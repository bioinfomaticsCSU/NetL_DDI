
from scipy.io import loadmat
import numpy as np
import scipy.io
from sklearn.exceptions import UndefinedMetricWarning
import warnings
from scipy.sparse import csr_matrix, lil_matrix
import scipy.sparse as sparse
from scipy.sparse import csgraph
import logging
import theano
from theano import tensor as T

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
logger = logging.getLogger(__name__)
theano.config.exception_verbosity = 'high'


def load_adjacency_matrix(file, variable_name="network"):
    data = scipy.io.loadmat(file)
    logger.info("loading mat file %s", file)
    return data[variable_name]


def deepwalk_filter(evals, window):
    for i in range(len(evals)):
        x = evals[i]
        evals[i] = 1. if x >= 1 else x * (1 - x ** window) / (1 - x) / window
    evals = np.maximum(evals, 0)
    # logger.info("After filtering, max eigenvalue=%f, min eigenvalue=%f",
    #         np.max(evals), np.min(evals))
    return evals


def approximate_normalized_graph_laplacian(A, rank, which="LA"):
    n = A.shape[0]
    L, d_rt = csgraph.laplacian(A, normed=True, return_diag=True)
    # X = D^{-1/2} W D^{-1/2}
    X = sparse.identity(n) - L
    logger.info("Eigen decomposition...")
    # evals, evecs = sparse.linalg.eigsh(X, rank,
    #        which=which, tol=1e-3, maxiter=300)
    evals, evecs = sparse.linalg.eigsh(X, rank, which=which)
    # logger.info("Maximum eigenvalue %f, minimum eigenvalue %f", np.max(evals), np.min(evals))
    # logger.info("Computing D^{-1/2}U..")
    D_rt_inv = sparse.diags(d_rt ** -1)
    D_rt_invU = D_rt_inv.dot(evecs)
    return evals, D_rt_invU


def approximate_deepwalk_matrix(evals, D_rt_invU, window, vol, b):
    evals = deepwalk_filter(evals, window=window)
    X = sparse.diags(np.sqrt(evals)).dot(D_rt_invU.T).T
    m = T.matrix()
    mmT = T.dot(m, m.T) * (vol / b)
    f = theano.function([m], T.log(T.maximum(mmT, 1)))
    Y = f(X.astype(theano.config.floatX))
    # logger.info("Computed DeepWalk matrix with %d non-zero elements",
    #         np.count_nonzero(Y))
    return sparse.csr_matrix(Y)


def svd_deepwalk_matrix(X, dim):
    u, s, v = sparse.linalg.svds(X, dim, return_singular_vectors="u")
    # return U \Sigma^{1/2}
    return sparse.diags(np.sqrt(s)).dot(u.T).T


def direct_compute_deepwalk_matrix(A, window, b):
    n = A.shape[0]
    vol = float(A.sum())
    L, d_rt = csgraph.laplacian(A, normed=True, return_diag=True)
    # X = D^{-1/2} A D^{-1/2}
    X = sparse.identity(n) - L
    S = np.zeros_like(X)
    X_power = sparse.identity(n)
    for i in range(window):
        # logger.info("Compute matrix %d-th power", i+1)
        X_power = X_power.dot(X)
        S += X_power
    S *= vol / window / b
    D_rt_inv = sparse.diags(d_rt ** -1)
    M = D_rt_inv.dot(D_rt_inv.dot(S).T)
    m = T.matrix()
    f = theano.function([m], T.log(T.maximum(m, 1)))
    Y = f(M.todense().astype(theano.config.floatX))
    return sparse.csr_matrix(Y)


def Fun_norm(M):
    [_, num] = M.shape
    nM = np.zeros_like(M)
    Result_M = np.zeros_like(M)

    for i in range(num):
        nM[i,i] = np.sum(M[i,:])

    for i in range(num):
        rsum = nM[i,i]
        for j in range(num):
            csum = nM[j,j]
            if rsum==0 or csum==0:
                Result_M[i,j] = 0
            else:
                Result_M[i,j] = M[i,j]/np.sqrt(rsum*csum)

    return Result_M


def netmf_small(input, ddi_191878, need_minus_list, window, negative, dim):
    # logger.info("Running NetMF for a small window size...")
    # logger.info("Window size is set to be %d", window)

    ddis = np.load(ddi_191878, 'r')
    for i in need_minus_list:
        input[ddis[i][0], ddis[i][1]] = 0
        input[ddis[i][1], ddis[i][0]] = 0

    input = csr_matrix(input)
    # directly compute deepwalk matrix
    deepwalk_matrix = direct_compute_deepwalk_matrix(input, window=window, b=negative)
    # factorize deepwalk matrix with SVD
    deepwalk_embedding = svd_deepwalk_matrix(deepwalk_matrix, dim=dim)

    # input_sim = np.loadtxt('data/input/Drug_simMat.txt')
    # input_sim = Fun_norm(input_sim)
    # input_sim = csr_matrix(input_sim)
    # deepwalk_matrix_sim = direct_compute_deepwalk_matrix(input_sim, window=window, b=negative)
    # # factorize deepwalk matrix with SVD
    # deepwalk_embedding_sim = svd_deepwalk_matrix(deepwalk_matrix_sim, dim=dim)

    modified_embedding = np.zeros([ddis.shape[0], 2 * dim], float)

    for i in range(ddis.shape[0]):
        index1 = ddis[i][0]
        index2 = ddis[i][1]
        t1_embeddings = np.hstack((deepwalk_embedding[index1], deepwalk_embedding[index2]))
        # t2_embeddings = np.hstack((deepwalk_embedding_sim[index1], deepwalk_embedding_sim[index2]))
        modified_embedding[i, :] = t1_embeddings

    return modified_embedding


def find_all_index(arr, item):
    return [i for i, a in enumerate(arr) if a > item]


def load_data(num_CV):
    train_ratio = 0.6
    val_ratio = 0.2

    np.random.seed(num_CV)

    label = loadmat('data/input/group_191878.mat')['group']

    y = label
    y_t = lil_matrix(y.transpose())
    y_train = np.zeros((y.shape[0], y.shape[1]), int)
    y_val = np.zeros((y.shape[0], y.shape[1]), int)
    y_test = np.zeros((y.shape[0], y.shape[1]), int)

    # construct y_train and y_test;
    # spit data for each class;
    for ddc in range(y_t.shape[0]):
        # the ddc-th class
        # ddis stores indexes of nonzero elements in the ddc-th row of y_t
        ddis = list(y_t.rows[ddc])
        # split the drug-drug interactions belong to the ddc-th class into train set and test set;
        if len(ddis) >= 5:
            np.random.shuffle(ddis)

            train_count = round(len(ddis) * train_ratio)
            valid_count = round(len(ddis) * val_ratio)

            for i in ddis[0: train_count]:
                y_train[i, ddc] = 1
            for i in ddis[train_count: train_count + valid_count]:
                y_val[i, ddc] = 1
            for i in ddis[train_count + valid_count:]:
                y_test[i, ddc] = 1
    # calculate the sum of each row in matrix y_train;
    s_y_train = y_train.sum(axis=1)
    # calculate the sum of each row in matrix y_test;
    s_y_val = y_val.sum(axis=1)
    s_y_test = y_test.sum(axis=1)
    # find elements with at least one lable;
    # function find_all_index is to find the indexes of element>0;
    train_index = find_all_index(s_y_train, 0)
    val_index = find_all_index(s_y_val, 0)
    test_index = find_all_index(s_y_test, 0)

    t_index = list(set(train_index).intersection(set(val_index)))
    for t_element in t_index:
        y_train[t_element, :] = y_train[t_element, :] + y_val[t_element, :]
        y_val[t_element, :] = 0
        val_index.remove(t_element)

    t_index = list(set(train_index).intersection(set(test_index)))
    for t_element in t_index:
        y_train[t_element, :] = y_train[t_element, :] + y_test[t_element, :]
        y_test[t_element, :] = 0
        test_index.remove(t_element)

    t_index = list(set(val_index).intersection(set(test_index)))
    for t_element in t_index:
        y_val[t_element, :] = y_val[t_element, :] + y_test[t_element, :]
        y_test[t_element, :] = 0
        test_index.remove(t_element)

    print(len(train_index))
    print(len(val_index))
    print(len(test_index))

    # load adjacency matrix
    A = loadmat('data/input/ddinetwork.mat')['ddinetwork']
    listd = list(set(val_index).union(set(test_index)))
    feature = netmf_small(A, 'data/input/ddi_index_191878.npy', listd, 1, 1.0, 128)

    filename = "embedding_256_T1_" + str(num_CV) + ".npy"
    np.save(filename, feature)

    train_x = feature[train_index, :]
    train_y = y_train[train_index, :]
    val_x = feature[val_index, :]
    val_y = y_val[val_index, :]
    test_x = feature[test_index, :]
    test_y = y_test[test_index, :]

    return train_x, train_y, val_x, val_y, test_x, test_y
