import os
import sys
import shutil
import glob
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext

class SmartBuildExt(build_ext):
    def run(self):
        ext_suffix = ".pyd" if sys.platform == "win32" else ".so"
        target_path = os.path.join("python", "forgetune", f"_forgetune_cpp{ext_suffix}")
        
        # Check source freshness: if any C++ header or source is newer than target_path, must rebuild
        needs_rebuild = not os.path.isfile(target_path)
        if not needs_rebuild:
            target_mtime = os.path.getmtime(target_path)
            for pattern in ["src/*.cpp", "include/**/*.hpp", "include/**/*.h", "tests/cpp/*.cpp"]:
                for src_file in glob.glob(pattern, recursive=True):
                    if os.path.isfile(src_file) and os.path.getmtime(src_file) > target_mtime:
                        needs_rebuild = True
                        print(f"[setup.py] Source '{src_file}' is newer than '{target_path}', rebuilding...")
                        break
                if needs_rebuild:
                    break

        if needs_rebuild:
            build_script = os.path.join(os.path.dirname(__file__), "scripts", "build_native.py")
            if os.path.isfile(build_script):
                import subprocess
                print(f"[setup.py] Compiling native extension via scripts/build_native.py...")
                res = subprocess.run([sys.executable, build_script], capture_output=True, text=True)
                if res.returncode != 0:
                    print(res.stdout)
                    print(res.stderr)
                    raise RuntimeError(f"Native build failed with exit code {res.returncode}")
                print(f"[setup.py] Native extension compiled successfully.")
            else:
                super().run()

        # Copy target_path to build_lib if building a wheel
        if getattr(self, "build_lib", None) and os.path.isfile(target_path):
            dest_dir = os.path.join(self.build_lib, "forgetune")
            os.makedirs(dest_dir, exist_ok=True)
            dest_path = os.path.join(dest_dir, f"_forgetune_cpp{ext_suffix}")
            shutil.copy2(target_path, dest_path)
            print(f"[setup.py] Copied extension to {dest_path}")

setup(
    name="forgetune",
    version="2.0.0",
    package_dir={"": "python"},
    packages=find_packages(where="python"),
    package_data={"forgetune": ["*.pyd", "*.so", "*.dll"]},
    ext_modules=[Extension("forgetune._forgetune_cpp", sources=["src/bindings.cpp"])],
    cmdclass={"build_ext": SmartBuildExt},
)
