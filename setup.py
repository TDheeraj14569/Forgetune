import os
import sys
from setuptools import setup, find_packages

try:
    from pybind11.setup_helpers import Pybind11Extension, build_ext
    has_pybind11 = True
except ImportError:
    has_pybind11 = False

ext_modules = []
cmdclass = {}

if has_pybind11:
    ext_modules = [
        Pybind11Extension(
            "forgetune._forgetune_cpp",
            [
                "src/bindings.cpp",
                "src/ir.cpp",
                "src/graph_verifier.cpp",
                "src/interpreter.cpp",
                "src/pass_manager.cpp",
                "src/optimizer.cpp",
                "src/fusion.cpp",
                "src/codegen.cpp",
                "src/json_io.cpp",
                "src/memory_planner.cpp",
                "tests/cpp/test_main.cpp",
            ],
            include_dirs=["include"],
            define_macros=[("FORGETUNE_EXCLUDE_MAIN", "1")],
            cxx_std=17,
        ),
    ]
    cmdclass["build_ext"] = build_ext

setup(
    name="forgetune",
    version="2.0.0",
    package_dir={"": "python"},
    packages=find_packages(where="python"),
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)
