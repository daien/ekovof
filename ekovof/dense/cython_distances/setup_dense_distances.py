from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

# Note: numy.get_include is important to use the correct numpy version (in a non-std location)
#   else we get "RuntimeWarning: numpy.dtype size changed, may indicate binary incompatibility"
ext_modules = [Extension(
    "dense_distances",
    ["dense_distances.pyx"],
    include_dirs=[numpy.get_include()],
    extra_compile_args=['-fopenmp'],
    extra_link_args=['-fopenmp'])]

setup(
    name = 'dense_distances',
    cmdclass = {'build_ext': build_ext},
    ext_modules = ext_modules
)

