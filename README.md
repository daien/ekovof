# Efficient Kernels On Vectors Of Floats

Computation of kernels between between vectors (potentially sparse).

Includes vector-to-vector (v2v), vector-to-matrix (v2m), matrix-to-matrix (m2m,
gram), centering in RKHS (cf. the GenericKernel class for the API of all
kernels), and other various utilities like kernel PCA.

Note: as we use C code for the critical distance computations, we enforce that
all vector elements must be double precision floating points.

OpenMP is used for parallelization.

The sparse module is using swig to wrap C code.

The dense module is using cython.

Tests depend on the nose testing framework.

# Author

Adrien Gaidon


# License

MIT License
