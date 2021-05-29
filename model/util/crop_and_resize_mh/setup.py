#----------------------------------------------------------------------------------
# note I copy the original setup file from https://github.com/pybind/python_example
# and I modified some code to accept my roi pooling cpp code

from setuptools import setup

# Available at setup time due to pyproject.toml
from pybind11.setup_helpers import Pybind11Extension, build_ext
from pybind11 import get_cmake_dir

import sys

__version__ = "0.0.1"

# The main interface is through Pybind11Extension.
# * You can add cxx_std=11/14/17, and then build_ext can be removed.
# * You can set include_pybind11=false to add the include directory yourself,
#   say from a submodule.
#
# Note:
#   Sort input source files if you glob sources to ensure bit-for-bit
#   reproducible builds (https://github.com/pybind/python_example/pull/53)

ext_modules = [
    Pybind11Extension("crop_and_resize_mh",
        ["src/crop_and_resize.cpp"],
        # Example: passing in the version to the compiled code
        define_macros = [('VERSION_INFO', __version__)],
        ),
]

setup(
    name="crop_and_resize_mh",
    version=__version__,
    author="victor miao",
    author_email="miaohua1982@gmail.com",
    url="https://github.com/miaohua1982/simple_fasterrcnn_pytorch",
    description="A c++ version roi alignment pool algorithm project using pybind11",
    long_description="",
    ext_modules=ext_modules,
    extras_require={"test": "pytest"},
    # Currently, build_ext only provides an optional "highest supported C++
    # level" feature, but in the future it may provide more features.
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)
