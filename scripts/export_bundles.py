"""
Export script to generate clean release bundles:
1. forgetune_cpp_src.txt (all C++ headers, sources, and tests)
2. forgetune_clean.zip (all tracked clean repository sources, docs, tests, and C++ source dump)
Copies both files to repo root and to Desktop.
"""

import os
import shutil
import subprocess
import zipfile

def main():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    os.chdir(repo_root)

    print("1. Generating forgetune_cpp_src.txt...")
    cpp_files = []
    for folder in ["include", "src", os.path.join("tests", "cpp")]:
        for root, _, filenames in os.walk(folder):
            for fn in sorted(filenames):
                if fn.endswith((".h", ".hpp", ".c", ".cpp")):
                    cpp_files.append(os.path.join(root, fn))

    cpp_txt_path = "forgetune_cpp_src.txt"
    with open(cpp_txt_path, "w", encoding="utf-8") as out_f:
        for fpath in cpp_files:
            rel_path = os.path.relpath(fpath, repo_root).replace("\\", "/")
            out_f.write(f"\n=== FILE: {rel_path} ===\n")
            with open(fpath, "r", encoding="utf-8", errors="replace") as in_f:
                out_f.write(in_f.read())
            out_f.write("\n")

    cpp_txt_size = os.path.getsize(cpp_txt_path) / 1024
    print(f"   Created {cpp_txt_path} ({cpp_txt_size:.1f} KB)")

    print("2. Generating forgetune_clean.zip...")
    res = subprocess.run(["git", "ls-files"], capture_output=True, text=True, check=True)
    tracked_files = [f.strip() for f in res.stdout.splitlines() if f.strip()]

    # Always include forgetune_cpp_src.txt
    if cpp_txt_path not in tracked_files:
        tracked_files.append(cpp_txt_path)

    # Exclude binaries or zip if accidentally tracked
    valid_files = []
    for f in tracked_files:
        if f.endswith((".zip", ".exe", ".dll", ".pyd", ".obj", ".o", ".a", ".lib")):
            continue
        if os.path.isfile(f):
            valid_files.append(f)

    zip_path = "forgetune_clean.zip"
    if os.path.exists(zip_path):
        os.remove(zip_path)

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for f in sorted(valid_files):
            zf.write(f, arcname=f)

    zip_size = os.path.getsize(zip_path) / 1024
    print(f"   Created {zip_path} with {len(valid_files)} files ({zip_size:.1f} KB)")

    print("3. Copying release bundles to Desktop...")
    desktop_dir = os.path.join(os.path.expanduser("~"), "Desktop")
    if os.path.isdir(desktop_dir):
        dest_zip = os.path.join(desktop_dir, "forgetune_clean.zip")
        dest_txt = os.path.join(desktop_dir, "forgetune_cpp_src.txt")
        shutil.copy2(zip_path, dest_zip)
        shutil.copy2(cpp_txt_path, dest_txt)
        print(f"   Copied to {dest_zip}")
        print(f"   Copied to {dest_txt}")

    print("\nBundles generated successfully!")

if __name__ == "__main__":
    main()
