"""
Efficient Kernels On Vectors Of Floats
======================================

Module to compute kernels between sparse vectors

Uses interface to a "swig'd" C library (approx. 1500x more efficient)

WARNING: the functions here use sparse CSC or CSR matrix formats with double
         float precision (cf. doc of scipy.sparse python module)!

AUTHOR: Adrien Gaidon, LEAR, INRIA, 2009
"""
