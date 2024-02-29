# cython: infer_types=False, language_level=3, cdivision=True, boundscheck=False, wraparound=False, profile=False, linetrace=False
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION CYTHON_TRACE_NOGIL=1
# distutils: language = c++

from src.selection.dampening.delta_base cimport DeltaBase
from src.selection.dampening.delta_info_gain cimport DeltaInfoGain

from libcpp.algorithm cimport minmax
from libcpp.pair cimport pair

cimport numpy as np

np.import_array()

cdef class DampeningFunc:
  cdef np.ndarray data
  cdef DeltaInfoGain Delta

  def __cinit__(self, np.ndarray data, DeltaBase Delta):
    self.data = data
    self.Delta = Delta

  cpdef double evaluate(self, double[:] u, size_t r):
    cdef size_t i = 0
    cdef double bi = 0 
    cdef double bip = 0 
    cdef double gs = self.Delta.global_sensitivity()
    self.Delta.sens = 0 # reseting

    cdef int sign = 1

    if u[r] < 0:
        sign = -1
        u[r] *= sign

    bi = self.b(i, r)
    while True:
      bip = self.b(i + 1, r)

      # print(f"[LD] r[{r}] Iteration: {i} | {bi:.4f} <= {u[r]:.4f} < {bip:.4f} | delta_b={bip - bi} gs={gs}", end="\r")

      # This stop criteria follows the original implementation of local dampening
      if (bip - bi) >= gs - 1e-6:
        return (((u[r] - bi) / (gs)) + i) * sign

      if bi <= u[r] < bip:
        break

      i += 1
      bi = bip

    return (((u[r] - bi) / (bip - bi)) + i) * sign

  cdef double b(self, size_t i, size_t r) nogil:
    cdef size_t j = 0
    cdef double sum_d = 0
    cdef double last_delta = 0
    cdef double new_delta = 0
    cdef pair[double, double] bounds

    if i == 0:
      return 0

    if i <= 0:
      return -self.b(-i, r)

    last_delta = 0
    for j in range(i+1):
        bounds = minmax(self.Delta.calc(j), last_delta)
        new_delta = bounds.second # get the max
        sum_d += new_delta
        # Cache
        last_delta = new_delta
        self.Delta.last_cands = self.Delta.cands
        self.Delta.cands.clear()

    return sum_d
