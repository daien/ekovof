// 
// Distance functions between Compressed Sparse Column (resp. Row) CSC (resp. CSR) column (resp. row) vectors
//
// Base distance or similarity (between vector elements) is defined at compile time
// with macro BASE_DIST:
//
// BASE_DIST == 0 : linear, K(x,y) = sum_i x_i*y_i (not a distance, but a similarity)
// BASE_DIST == 1 : intersection, K(x,y) = sum_i min(x_i,y_i) (not a distance, but a similarity)
// BASE_DIST == 2 : total variation, d(x,y) = sum_i |x_i - y_i|
// BASE_DIST == 3 : chi-square, d(x,y) = sum_i (x_i - y_i)^2 / (x_i + y_i)
// BASE_DIST == 4 : l2-square, d(x,y) = sum_i (x_i - y_i)^2
//
// CSC/CSR format: (data, indices, indptr)
//   where the row/column indices for column/row i are stored in
//   ``indices[indptr[i]:indices[i+1]]`` and their corresponding values are
//   stored in ``data[indptr[i]:indptr[i+1]]``.
//

/** Return the distance between two CSC/CSR column/row vectors x and y
 *
 * WARNING: vectors are assumed to be NORMALIZED already !!!
 *
 * Args:
 *       -x_nnz     : number of non-zero elements in the sparse vector x
 *                    (indptr[1] for CSC column vectors)
 *       -x_indices : the list of indexes of the non-zero elements in the vector
 *                    (indices for CSC column vectors)
 *       -x_data    : the non-zero values of the vector
 *                    (data for CSC column vectors)
 *
 * Note: x_data[k] = x[x_indices[k]] 
 *     if x is the dense representation of the corresponding sparse CSC vector
 *
 */
double v2v(
    int x_nnz, int *x_indices, double *x_data, 
    int y_nnz, int *y_indices, double *y_data);


/** Compute distances between (sparse) vector x and (sparse) matrix m:
 *
 * Args:
 *       - out_values : output array where the distance values [ d(x,m[0]), ..., d(x,m[n-1]) ] are stored
 *       - n          : number of output values (= number of columns in m)
 *       - x_*        : CSC column vector (for BoFs, shape = (voc_size,1)) or equivalent CSR row vector
 *       - m_*        : CSC matrix (for BoFs, shape = (voc_size, num_tr_samps)) or equivalent CSR row matrix
 *                      (elements are stored column wise)
 *
 */
void v2m(
    double *out_values, int n,
    int *x_indptr, int x_indptr_dim,
    int *x_indices, int x_indices_dim,
    double *x_data, int x_data_dim,
    int *m_indptr, int m_indptr_dim,
    int *m_indices, int m_indices_dim,
    double *m_data, int m_data_dim);


/** Compute distances between a (sparse) matrix of column vectors mtest and a (sparse) matrix m:
 *
 * Args:
 *       - out_mat_test : flattened (C-contiguous) 2D-array of dim num_test_samps x num_samps,
 *                        [ d(x_test_1, x_1), ..., d(x_test_1, x_num_samps), ...,
 *                          d(x_test_num_test_samps, x_1), ..., d(x_test_num_test_samps, x_num_samps) ]
 *       - m*  : sparse CSC/CSR matrix storing vectors in its columns/rows
 *       - num_threads: number of threads to use for parallel computation
 *
 */
void m2m(
    double *out_mat_test, int num_test_samps, int num_train_samps,
    int *mtest_indptr, int mtest_indptr_dim,
    int *mtest_indices, int mtest_indices_dim,
    double *mtest_data, int mtest_data_dim,
    int *m_indptr, int m_indptr_dim,
    int *m_indices, int m_indices_dim,
    double *m_data, int m_data_dim,
    int num_threads);


/** Compute (symmetric) gram matrix
 *
 * uses multiple threads (with OpenMP)
 *
 * Args:
 *       - out_mat : flattened (C-contiguous) symmetrix 2D-array of dim num_samps x num_samps,
 *                   where the gram matrix K will be saved
 *                   => K[i,j] = out_mat[i*num_samps + j]
 *       - m* : sparse CSC/CSR matrix storing vectors in its columns/rows
 *       - num_threads: number of threads to use for parallel computation
 *
 */
void gram(
    double *out_mat, int num_samps,
    int *m_indptr, int m_indptr_dim,
    int *m_indices, int m_indices_dim,
    double *m_data, int m_data_dim,
    int num_threads);

