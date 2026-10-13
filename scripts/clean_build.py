"""
Clean rebuild script for ForgeTune.
Removes only project-generated build artifacts and runs clean rebuild.
"""

import os
import sys
import shutil
import subprocess

def clean_build():
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    os.chdir(root_dir)

    print("=================================================================")
    print(" ForgeTune: Clean Build")
    print(f" Root: {root_dir}")
    print("=================================================================")

    # Safely remove generated artifacts directories
    dirs_to_clean = [
        os.path.join(root_dir, "build"),
        os.path.join(root_dir, "bin"),
    ]

    for d in dirs_to_clean:
        if os.path.exists(d):
            print(f"Removing build artifact directory: {d}")
            try:
                shutil.rmtree(d)
            except Exception as e:
                print(f"Warning: could not completely remove {d}: {e}")

    pyd_file = os.path.join(root_dir, "python", "forgetune", "_forgetune_cpp.pyd")
    if os.path.exists(pyd_file):
        print(f"Removing build artifact: {pyd_file}")
        try:
            os.remove(pyd_file)
        except Exception as e:
            print(f"Warning: could not remove {pyd_file}: {e}")

    # Recreate build and bin directories
    os.makedirs(os.path.join(root_dir, "build"), exist_ok=True)
    os.makedirs(os.path.join(root_dir, "bin"), exist_ok=True)

    print("Clean build directory initialized successfully.")
    print("Now invoking verification runner scripts/verify_all.py...")

    python_exe = os.path.join(root_dir, ".venv", "Scripts", "python.exe")
    res = subprocess.run([python_exe, "scripts/verify_all.py"])
    sys.exit(res.returncode)

if __name__ == "__main__":
    clean_build()
