#ifndef BASE_DIST
#error "BASE_DIST not declared for 'sparse_distance.i' SWIG interface file"
#endif
#if BASE_DIST == 0
%module sparse_linear
#elif BASE_DIST == 1
%module sparse_intersection
#elif BASE_DIST == 2
%module sparse_totvar
#elif BASE_DIST == 3
%module sparse_chisquare
#elif BASE_DIST == 4
%module sparse_l2
#else
#error "Invalid BASE_DIST in SWIG"
#endif

%{
#define SWIG_FILE_WITH_INIT
#include "sparse_distance.h"
#include <stdio.h>
#include <sched.h>
#include <omp.h>
%}

%include "numpy.i"

%init %{
  import_array();
%}

// By default, functions release Python's global lock
%exception {
  Py_BEGIN_ALLOW_THREADS
  $action
  Py_END_ALLOW_THREADS
}

%apply (double* ARGOUT_ARRAY1, int DIM1) {(double* out_values, int n)}
%apply (double* INPLACE_ARRAY2, int DIM1, int DIM2) {(double* out_mat_arr, int num_samps1, int num_samps2)}
%apply (double* INPLACE_ARRAY2, int DIM1, int DIM2) {(double* out_mat_test, int num_test_samps, int num_train_samps)}

%apply (int* IN_ARRAY1, int DIM1) {
  (int *mtest_indptr, int mtest_indptr_dim),
  (int *mtest_indices, int mtest_indices_dim),
  (int *m_indptr, int m_indptr_dim),
  (int *m_indices, int m_indices_dim),
  (int *x_indptr, int x_indptr_dim),
  (int *x_indices, int x_indices_dim),
  (int *y_indptr, int y_indptr_dim),
  (int *y_indices, int y_indices_dim)
}

%apply (double* IN_ARRAY1, int DIM1) {
  (double *mtest_data, int mtest_data_dim),
  (double *m_data, int m_data_dim),
  (double *x_data, int x_data_dim),
  (double *y_data, int y_data_dim)
}

// ----------------------------------------------------------------
// wrap C functions so that it can take the 'true' python arguments
// ----------------------------------------------------------------

// '%ignore' because of the '%include' below

%rename (v2v) my_v2v;
%ignore v2v;
%inline %{
  double my_v2v(
      int *x_indptr, int x_indptr_dim,
      int *x_indices, int x_indices_dim,
      double *x_data, int x_data_dim,
      int *y_indptr, int y_indptr_dim,
      int *y_indices, int y_indices_dim,
      double *y_data, int y_data_dim)
  {

    int x_nnz = x_indptr[1];
    int y_nnz = y_indptr[1];

    // check that the sparse vector is well formed
    if (x_nnz != x_indices_dim && x_nnz != x_data_dim) {
      PyErr_Format(PyExc_ValueError, 
          "First vector is badly formed (nnz:%d, indices_dim:%d, data_dim:%d)",
          x_nnz, x_indices_dim, x_data_dim);
      return -2.0;
    }
    if (y_nnz != y_indices_dim && y_nnz != y_data_dim) {
      PyErr_Format(PyExc_ValueError, 
          "Second vector is badly formed (nnz:%d, indices_dim:%d, data_dim:%d)",
          y_nnz, y_indices_dim, y_data_dim);
      return -2.0;
    }

    // select the appropriate base distance

    // return the result
    return v2v(
        x_nnz, x_indices, x_data,
        y_nnz, y_indices, y_data);

  }
%}

%rename (gram) my_gram;
%ignore gram;
%inline %{
  void my_gram(
      double *out_mat_arr, int num_samps1, int num_samps2,
      int *m_indptr, int m_indptr_dim,
      int *m_indices, int m_indices_dim,
      double *m_data, int m_data_dim,
      int num_threads)
  {
    // check that the matrix is well formed
    if (num_samps1 != num_samps2) {
      PyErr_Format(PyExc_ValueError, 
          "Pre-allocated kernel matrix is not symmetric! (%d != %d)",
          num_samps1, num_samps2);
      //return -1.0;
    }

    // do the job
    gram(
        out_mat_arr, num_samps1,
        m_indptr, m_indptr_dim,
        m_indices, m_indices_dim,
        m_data, m_data_dim,
        num_threads);

  }
%}

%include "sparse_distance.h"

