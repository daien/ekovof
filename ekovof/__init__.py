"""
Efficient Kernels On Vectors Of Floats
======================================

Efficient computation of kernels between between vectors (potentially sparse)

Note: as we use C code for the critical distance computations, we enforce that
all vector elements must be double precision floating points

"""


from .utils import get_heuristic_gamma, get_kernel_object
