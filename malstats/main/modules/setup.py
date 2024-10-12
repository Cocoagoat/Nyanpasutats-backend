from setuptools import setup
from Cython.Build import cythonize
import numpy

# Define the Cython extension module
setup(
    ext_modules=cythonize("your_module.pyx"),  # Replace with your .pyx file
    include_dirs=[numpy.get_include()],        # Add NumPy headers for compilation
)