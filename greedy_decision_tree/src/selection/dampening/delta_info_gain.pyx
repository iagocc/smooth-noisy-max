# cython: infer_types=False, language_level=3, cdivision=True, boundscheck=False, wraparound=False, profile=False, linetrace=False
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION CYTHON_TRACE_NOGIL=1
# distutils: language = c++

from src.selection.dampening.delta_base cimport DeltaBase

from libcpp.vector cimport vector
from libcpp.set cimport set as cset
from libcpp.pair cimport pair
from libcpp.map cimport map
from libcpp.algorithm cimport max_element
from libc.math cimport log2
from cython.operator import dereference

import numpy as np
cimport numpy as np

cdef class DeltaInfoGain(DeltaBase):
  def __cinit__(self, np.ndarray x, int attr, np.ndarray class_attr, map[pair[int, int], cset[pair[int, int]]] last_candidates):
    self.x = x
    self.attr = attr
    self.class_attr = class_attr
    self.last_cands = last_candidates
    self.sens = 0
    self.build_size_table()
    self.init_candidates(class_attr.shape[0])

  cdef double global_sensitivity(self):
    return self.h(self.x.shape[0], 0)

  cdef void build_size_table(self):
    cdef np.ndarray unqs
    cdef size_t j, c
    unqs = np.unique(self.x[:, self.attr])
    for j, jv in enumerate(unqs):
      mask = self.x[:, self.attr] == jv
      for c, cv in enumerate(self.class_attr):
        c_count = np.count_nonzero(self.x[mask, -1] == cv)
        self.size_table.insert(pair[pair[int, int], int](pair[int,int](j,c), c_count))
    pass

  cdef void init_candidates(self, int nlabels) noexcept nogil:
    cdef int j, c, tj, tjc, cc = 0
    cdef pair[int, int] jc = pair[int, int](j, c)
    cdef cset[pair[int, int]] cands
    cdef pair[pair[int, int], int] sz_par
    cdef pair[int, int] jc_pair

    for sz_par in self.size_table:
      jc_pair = sz_par.first
      j = jc_pair.first
      c = jc_pair.second

      tj = 0
      for cc in range(nlabels):
        tj +=  dereference(self.size_table.find(pair[int,int](j, cc))).second

      tjc = dereference(self.size_table.find(pair[int,int](j, c))).second
      cands.insert(pair[int, int](tj, tjc))
      self.last_cands.insert(pair[pair[int, int], cset[pair[int, int]]](jc, cands))

  """
    Notation for the paper's algorithm
    t is the distance
    j is the self.attrs's attribute values
    c is the label value
  """
  cdef double candidates(self, int t, int j, int c, int n) noexcept nogil:
    cdef cset[pair[int, int]] cands
    cdef pair[int, int] jc = pair[int, int](j, c) 
    cdef int c1, c2 = 0
    cdef pair[int, int] cand_pair
    cdef cset[pair[int, int]] lc
    cdef double sens, max_sens = 0

    lc = dereference(self.last_cands.find(jc)).second
    for cand_pair in lc:
      c1 = cand_pair.first
      c2 = cand_pair.second

      sens = self.h(c1, c2)
      if sens > max_sens:
        max_sens = sens
      
      if c1 > 0 and c2 > 0:
        cands.insert(pair[int, int](c1 - 1, c2 - 1))
      if c1 < n:
        cands.insert(pair[int, int](c1 + 1, c2))

    self.cands.insert(pair[pair[int, int], cset[pair[int, int]]](jc, cands))
    return max_sens
    

  cdef double calc(self, int t) noexcept nogil:
    cdef int j, c
    cdef pair[pair[int, int], int] sz_par
    cdef pair[int, int] jc_pair
    cdef int n = self.x.shape[0]
    cdef double sens = 0

    for sz_par in self.size_table:
        jc_pair = sz_par.first
        j = jc_pair.first
        c = jc_pair.second

        sens = self.candidates(t, j, c, n)
        if sens > self.sens:
          self.sens = sens
    
    return self.sens

  cdef double f(self, double x) noexcept nogil:
    if(x <= 1.):
        return 0.

    return x*log2((x+1)/x) + log2(x+1)

  cdef double g(self, double x) noexcept nogil:
    if(x <= 2.):
        return 0.

    return x*log2((x-1)/x) - log2(x-1)

  cdef double h(self, double x, double y) noexcept nogil:
    cdef double fr, gr = 0
    fr = self.f(x) - self.f(y)
    gr = self.g(y) - self.g(x)

    if fr > gr:
      return fr
    return gr
