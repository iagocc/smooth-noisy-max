# cython: infer_types=False, language_level=3, cdivision=True, boundscheck=False, wraparound=False, profile=False, linetrace=False
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
# distutils: language = c++

from src.selection.dampening.delta_base cimport DeltaBase

from libcpp.set cimport set as cset
from libcpp.pair cimport pair
from libcpp.map cimport map

cimport numpy as np

cdef class DeltaInfoGain(DeltaBase):
  cdef np.ndarray x
  cdef int attr
  cdef np.ndarray class_attr
  cdef map[pair[int, int], cset[pair[int, int]]] last_cands
  cdef map[pair[int, int], cset[pair[int, int]]] cands
  cdef map[pair[int, int], int] size_table
  cdef double sens

  cdef double global_sensitivity(self)
  cdef void build_size_table(self)
  cdef void init_candidates(self, int nlabels) noexcept nogil
  cdef double candidates(self, int t, int j, int c, int n) noexcept nogil
  cdef double f(self, double x) noexcept nogil
  cdef double g(self, double x) noexcept nogil
  cdef double h(self, double x, double y) noexcept nogil
  cdef double calc(self, int t) noexcept nogil