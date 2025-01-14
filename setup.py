from setuptools import setup
from setuptools.extension import Extension
import numpy as np
from glob import glob


pyx_files = glob("grf_cython/*.pyx", recursive=True)

from Cython.Build import cythonize
pyx_extensions = cythonize([Extension(name=f"grf_cython.{file.split('/')[-1][:-4]}",
                                        sources=[file],
                                        include_dirs=[np.get_include()],

                                        # to handle error messages
                                        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
                                        extra_compile_args=["-Wno-unreachable-code"],
                                        ) for file in pyx_files],
                            language_level="3")

setup(ext_modules=pyx_extensions,
      zip_safe=False,
      version="1.0.0")
