from setuptools import setup
from Cython.Build import cythonize
import numpy

# Define the Cython extension module
setup(
    ext_modules=cythonize("gottagofasttest2.pyx"),
    include_dirs=[numpy.get_include()],
)