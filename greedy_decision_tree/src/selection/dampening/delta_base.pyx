# cython: infer_types=False, language_level=3, cdivision=True, boundscheck=False, wraparound=False, profile=True, linetrace=True
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
# distutils: language = c++

cimport numpy as np

cdef class DeltaBase:
  pass