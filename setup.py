import os
import sys
from setuptools import setup, find_packages
from setuptools.command.build_ext import build_ext

class SmartBuildExt(build_ext):
    def run(self):
        ext_suffix = ".pyd" if sys.platform == "win32" else ".so"
        target_path = os.path.join("python", "forgetune", f"_forgetune_cpp{ext_suffix}")
        if os.path.isfile(target_path):
            print(f"[setup.py] Found native extension: {target_path}")
            return
        build_script = os.path.join(os.path.dirname(__file__), "scripts", "build_native.py")
        if os.path.isfile(build_script):
            try:
                import subprocess
                print(f"[setup.py] Compiling native extension via scripts/build_native.py...")
                subprocess.run([sys.executable, build_script], check=True)
                return
            except Exception as e:
                print(f"[setup.py] scripts/build_native.py invocation failed: {e}")
        try:
            super().run()
        except Exception as e:
            print(f"[setup.py] Native C++ compilation deferred: {e}. Run 'python scripts/build_native.py' to build.")

setup(
    name="forgetune",
    version="2.0.0",
    package_dir={"": "python"},
    packages=find_packages(where="python"),
    package_data={"forgetune": ["*.pyd", "*.so", "*.dll"]},
    cmdclass={"build_ext": SmartBuildExt},
)
