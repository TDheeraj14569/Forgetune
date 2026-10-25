import os
import sys
import ctypes
import subprocess
import ziglang

zig_exe = os.path.join(os.path.dirname(ziglang.__file__), "zig.exe")
test_dir = os.path.abspath("build/ci_probe")
os.makedirs(test_dir, exist_ok=True)
cpp_src = os.path.join(test_dir, "probe.cpp")
dll_out = os.path.join(test_dir, "probe.dll")

with open(cpp_src, "w", encoding="utf-8") as f:
    f.write('extern "C" __declspec(dllexport) int ci_probe_add(int a, int b) { return a + b; }\n')

cmd = [zig_exe, "c++", "-target", "x86_64-windows-gnu", "-shared", "-o", dll_out, cpp_src, "-O2", "-w"]
subprocess.run(cmd, check=True)

print(f"Loading unsigned test DLL from: {dll_out}")
try:
    lib = ctypes.CDLL(dll_out)
    lib.ci_probe_add.argtypes = [ctypes.c_int, ctypes.c_int]
    lib.ci_probe_add.restype = ctypes.c_int
    res = lib.ci_probe_add(40, 2)
    assert res == 42, f"Expected 42, got {res}"
    print(f"[+] CI probe SUCCESS: {dll_out} loaded and executed cleanly (result={res})")
except Exception as e:
    print(f"[-] CI probe DENIED: {type(e).__name__}: {e}")
