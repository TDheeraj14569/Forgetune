import os
import sys
import shutil
import glob
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext

def is_build_metadata_valid(meta_path: str) -> bool:
    """Checks whether the compiled native extension build metadata matches the current host environment."""
    import json
    import platform
    import sysconfig
    import shutil
    if not os.path.isfile(meta_path):
        return False
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        cur_abi = sysconfig.get_config_var("SOABI") or f"{sys.version_info.major}.{sys.version_info.minor}"
        if (meta.get("platform") != sys.platform or
            meta.get("machine") != platform.machine() or
            meta.get("python_abi") != cur_abi or
            meta.get("python_version", "").split(".")[:2] != [str(sys.version_info.major), str(sys.version_info.minor)]):
            return False
        if meta.get("build_flags") is not None and not isinstance(meta.get("build_flags"), list):
            return False
        if "pybind11_version" in meta and meta.get("pybind11_version"):
            try:
                import pybind11
                if meta["pybind11_version"] != getattr(pybind11, "__version__", "unknown"):
                    return False
            except ImportError:
                pass
        comp_path = meta.get("compiler_path")
        if comp_path:
            is_known_cmd = comp_path in ("zig", "clang", "gcc", "cl", "g++", "clang++")
            is_valid_file = os.path.isfile(comp_path) or shutil.which(comp_path) is not None
            if not (is_valid_file or is_known_cmd):
                return False
            meta_ver = meta.get("compiler_version")
            if meta_ver and meta_ver != "unknown":
                import subprocess
                try:
                    p = subprocess.run([comp_path, "version" if "zig" in os.path.basename(comp_path).lower() else "--version"], capture_output=True, text=True, timeout=5)
                    if p.returncode == 0 and p.stdout:
                        cur_ver = p.stdout.strip().splitlines()[0] if "zig" not in os.path.basename(comp_path).lower() else p.stdout.strip()
                        if cur_ver and cur_ver != meta_ver:
                            return False
                except Exception:
                    pass
        return True
    except Exception:
        return False

class SmartBuildExt(build_ext):
    def run(self):
        ext_suffix = ".pyd" if sys.platform == "win32" else ".so"
        target_path = os.path.join("python", "forgetune", f"_forgetune_cpp{ext_suffix}")
        meta_path = os.path.join("python", "forgetune", "_forgetune_cpp.build_meta.json")

        # Validate existence and build metadata
        needs_rebuild = not os.path.isfile(target_path) or not is_build_metadata_valid(meta_path)
        if needs_rebuild and os.path.isfile(target_path):
            print(f"[setup.py] Missing or stale build metadata in '{meta_path}' for current environment, rebuilding...")

        # Check source freshness: if any C++ header, source, or build script is newer than target_path, must rebuild
        if not needs_rebuild:
            target_mtime = os.path.getmtime(target_path)
            for pattern in ["src/*.cpp", "include/**/*.hpp", "include/**/*.h", "tests/cpp/*.cpp", "scripts/build_native.py", "CMakeLists.txt"]:
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
                print("[setup.py] Compiling native extension via scripts/build_native.py...")
                res = subprocess.run([sys.executable, build_script], capture_output=True, text=True)
                if res.returncode != 0:
                    print(res.stdout)
                    print(res.stderr)
                    raise RuntimeError(f"Native build failed with exit code {res.returncode}")
                print("[setup.py] Native extension compiled successfully.")
            else:
                super().run()

        # Copy target_path and metadata to build_lib if building a wheel
        if getattr(self, "build_lib", None) and os.path.isfile(target_path):
            dest_dir = os.path.join(self.build_lib, "forgetune")
            os.makedirs(dest_dir, exist_ok=True)
            dest_path = os.path.join(dest_dir, f"_forgetune_cpp{ext_suffix}")
            shutil.copy2(target_path, dest_path)
            print(f"[setup.py] Copied extension to {dest_path}")
            if os.path.isfile(meta_path):
                dest_meta = os.path.join(dest_dir, "_forgetune_cpp.build_meta.json")
                shutil.copy2(meta_path, dest_meta)
                print(f"[setup.py] Copied build metadata to {dest_meta}")

if __name__ == "__main__":
    setup(
        package_dir={"": "python"},
        packages=find_packages(where="python"),
        package_data={"forgetune": ["*.pyd", "*.so", "*.dll"]},
        ext_modules=[Extension("forgetune._forgetune_cpp", sources=["src/bindings.cpp"])],
        cmdclass={"build_ext": SmartBuildExt},
    )
