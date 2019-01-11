from distutils.core import setup
from Cython.Build import cythonize
import numpy as np

setup(name='optimized',
      ext_modules=cythonize("optimized.pyx"),
      include_dirs=[np.get_include()]
)