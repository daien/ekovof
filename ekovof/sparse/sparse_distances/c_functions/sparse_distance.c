#ifndef BASE_DIST
#error "BASE_DIST not defined for 'sparse_distance.c' file"
#elif BASE_DIST < 0
#error "BASE_DIST should be positive or 0"
#elif BASE_DIST > 4
#error "BASE_DIST should be less or equal than 4"
#endif

#include "sparse_distance.h"
#include <stdio.h>
#include <sched.h>
#include <omp.h>

// ------------------------------------------------------------------------------------
// Local distance functions that are inlined in the caller's code
// Note: BASE_DIST 0 and 1 are 0 if one element is 0

// BASE_DIST == 0 : x_i * y_i // no need to put it here

// BASE_DIST == 1 : min(x_i, y_i)
static inline double intersection(double x, double y)
{
  return (x < y) ? x : y;
}

// BASE_DIST == 2 : |x_i - y_i|
static inline double totvar(double x, double y)
{
  return (x < y) ? (y-x) : (x-y);
}

// BASE_DIST == 3 : (x_i - y_i)^2 / (x_i + y_i)
static inline double chisquare(double x, double y)
{
  return (x + y) ? ((x-y)*(x-y)/(x+y)) : 0.0;
}

// BASE_DIST == 4 : (x_i - y_i)^2
static inline double l2(double x, double y)
{
  return (x-y)*(x-y);
}

// ------------------------------------------------------------------------------------
// Vector and matrix distance computations

// vector to vector
double v2v(
    int x_nnz, int *x_indices, double *x_data, 
    int y_nnz, int *y_indices, double *y_data)
{

  int big = 1<<30;

  double out_val = 0.0;

 // number of non-zero elements processed in x, y respectively
 int nx = 0;
 int ny = 0;

 while ( 1 )
 { // compare the indices of current non-zero vw count in both bofs

   // current vw index
   int cur_x_ind, cur_y_ind;

   // check if we processed all elements of x
   if (nx < x_nnz)
     cur_x_ind = x_indices[nx];
   else
     cur_x_ind = big; // all non-zero x elements processed

   // check if we processed all elements of y
   if (ny < y_nnz)
     cur_y_ind = y_indices[ny];
   else
     cur_y_ind = big; // all non-zero y elements processed

   // check if we finished processing all non-zero elements of *both* vectors
   if (cur_x_ind == big && cur_y_ind == big)
     return out_val;

   if (cur_x_ind < cur_y_ind)
   {// we are further in y than in x => 0 at this index in y 
#if BASE_DIST == 2
     out_val += totvar(x_data[nx], 0.0); // total variation
#elif BASE_DIST == 3
     out_val += x_data[nx]; // chi-square
#elif BASE_DIST == 4
     out_val += l2(x_data[nx], 0.0); // l2
#endif
     nx++; // => linear or min = 0 and advance in x only
   }
   else if (cur_x_ind > cur_y_ind)
   {// reverse case wrt above
#if BASE_DIST == 2
     out_val += totvar(0.0, y_data[ny]); // total variation
#elif BASE_DIST == 3
     out_val += y_data[ny]; // chi-square
#elif BASE_DIST == 4
     out_val += l2(0.0, y_data[ny]); // l2
#endif
     ny++;
   }
   else
   { // same index => both have non-zero element at this index
#if BASE_DIST == 0
     out_val += x_data[nx] * y_data[ny]; // use linear
#elif BASE_DIST == 1
     out_val += intersection(x_data[nx], y_data[ny]); // use intersection
#elif BASE_DIST == 2
     out_val += totvar(x_data[nx], y_data[ny]); // use total variation
#elif BASE_DIST == 3
     out_val += chisquare(x_data[nx], y_data[ny]); // use chi-square
#elif BASE_DIST == 4
     out_val += l2(x_data[nx], y_data[ny]); // use l2
#endif
     // advance in both
     nx++;
     ny++;
   }

 }

}

// vector to matrix
void v2m(
    double *out_values, int n,
    int *x_indptr, int x_indptr_dim,
    int *x_indices, int x_indices_dim,
    double *x_data, int x_data_dim,
    int *m_indptr, int m_indptr_dim,
    int *m_indices, int m_indices_dim,
    double *m_data, int m_data_dim)
{

  // iterate over all columns of m
  // parallelization with OpenMP: not efficient here because tasks are too small in general
  int i;
  for (i=0; i < n; i++)
  { // compute K(x, m[i])

    // in python, list of indices (in the dense vector) of the non-zero elements for m column i:
    // m_indices[m_indptr[i]:m_indptr[i+1]] (similar for data)

    int      mi_nnz = m_indptr[i+1] - m_indptr[i];
    int *mi_indices = m_indices + m_indptr[i];
    double *mi_data = m_data + m_indptr[i];

    out_values[i] = v2v(
      x_indptr[1], x_indices, x_data,
      mi_nnz, mi_indices, mi_data);
  }

}

// matrix to matrix
void m2m(
    double *out_mat_test, int num_test_samps, int num_train_samps,
    int *mtest_indptr, int mtest_indptr_dim,
    int *mtest_indices, int mtest_indices_dim,
    double *mtest_data, int mtest_data_dim,
    int *m_indptr, int m_indptr_dim,
    int *m_indices, int m_indices_dim,
    double *m_data, int m_data_dim,
    int num_threads)
{

  // parallelization with OpenMP
  if (num_threads > 0)
    omp_set_num_threads(num_threads);
  else
    omp_set_num_threads(omp_get_num_procs()); // requests 1 thread per core

  int i;
  #pragma omp parallel
  {
    #pragma omp for
    for (i=0; i < num_test_samps; i++)
    { // distances for test sample i

      int mti_nnz      = mtest_indptr[i+1] - mtest_indptr[i];
      int *mti_indices = mtest_indices + mtest_indptr[i];
      double *mti_data = mtest_data + mtest_indptr[i];

      // compute K(mtest[i], m[j])
      int j;
      for (j=0; j < num_train_samps; j++) {
        out_mat_test[i*num_train_samps+j] = v2v(
          mti_nnz, mti_indices, mti_data,
          m_indptr[j+1] - m_indptr[j], m_indices + m_indptr[j], m_data + m_indptr[j]);

        //fprintf(stderr,"num_test_samps=%d num_train_samps=%d i=%d j=%d k=%f\n",
        //  num_test_samps, num_train_samps, i, j, out_mat_test[i*num_test_samps+j]);
        //fflush(stderr);
      }
    }
  }

}

// Gram matrix
void gram(
    double *out_mat, int num_samps,
    int *m_indptr, int m_indptr_dim,
    int *m_indices, int m_indices_dim,
    double *m_data, int m_data_dim,
    int num_threads)
{

  // parallelization with OpenMP
  if (num_threads > 0)
    omp_set_num_threads(num_threads);
  else
    omp_set_num_threads(omp_get_num_procs()); // requests 1 thread per core

  int i;
  #pragma omp parallel
  {
    #pragma omp for
    for (i=0; i < num_samps; i++)
    { // fill lower triangular matrix (including diagonal)

      //fprintf(stderr,"thread_id = %u, #threads = %u, #procs_available = %u, in parallel: %s\n",
      //  omp_get_thread_num(), omp_get_num_threads(), omp_get_num_procs(), omp_in_parallel()?"yes":"no");
      //fflush(stderr);

      int      mi_nnz = m_indptr[i+1] - m_indptr[i];
      int *mi_indices = m_indices + m_indptr[i];
      double *mi_data = m_data + m_indptr[i];

      // compute K(m[i], m[j])
      int j;
      for (j=0; j <= i; j++)
        out_mat[i*num_samps+j] = v2v(
          mi_nnz, mi_indices, mi_data,
          m_indptr[j+1] - m_indptr[j], m_indices + m_indptr[j], m_data + m_indptr[j]);
    }
  }

  // symmetrically fill upper triangular matrix (excluding diagonal)
  int k, p;
  for (k=1; k < num_samps; k++)
    for (p=0; p < k; p++)
      out_mat[p*num_samps+k] = out_mat[k*num_samps+p]; // K[p,k]

}
