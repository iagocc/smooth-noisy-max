from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules=cythonize(
        [
            "./src/selection/dampening/delta_base.pyx",
            "./src/selection/dampening/dampening_func.pyx",
            "./src/selection/dampening/delta_info_gain.pyx",
        ],
        annotate=True,
    ),
    include_dirs=[numpy.get_include()],
    extra_compile_args=["-O3"],
)
