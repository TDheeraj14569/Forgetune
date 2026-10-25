"""
ForgeTune V2.0 - Independent Process Deterministic Code Generation Verification.
Validates Gate 8:
- Direct evidence of deterministic code generation across independent OS processes.
- Independent subprocess execution with separate directories and isolated environments.
- Cryptographic SHA-256 byte comparisons establishing 100% bitwise identity.
- Multi-operator graph coverage (MatMul + Add + ReLU + Conv2D + Flatten).
- Negative testing proving intentional discrepancy detection.
"""

import hashlib
import os
import subprocess
import sys
import tempfile
import pytest

from forgetune.graph_adapter import Graph, Node

_ENGINE_CANDIDATES = [
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "bin", "forgetune_engine.exe")),
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "bin", "forgetune_test_engine.exe")),
]
ENGINE_EXE = next((p for p in _ENGINE_CANDIDATES if os.path.isfile(p)), None)


def _compute_sha256(file_path: str) -> str:
    """Computes SHA-256 digest of raw file bytes."""
    with open(file_path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _read_bytes(file_path: str) -> bytes:
    """Reads raw file bytes."""
    with open(file_path, "rb") as f:
        return f.read()


def test_independent_cli_subprocess_determinism(tmp_path):
    """
    Spawns two completely independent CLI subprocesses in separate isolated
    working directories, compiling identical graphs to separate outputs.
    Verifies bit-for-bit identity and matching SHA-256 digests.
    """
    assert ENGINE_EXE is not None and os.path.isfile(ENGINE_EXE), f"Engine executable not found at {ENGINE_EXE}"

    # Build representative multi-operator graph: MatMul -> Add -> ReLU
    g = Graph()
    x = g.add_node(Node(0, "INPUT", shape=[-1, 32]))
    w = g.add_node(Node(1, "CONST", shape=[32, 64], const_data=[0.05] * (32 * 64)))
    bias = g.add_node(Node(2, "CONST", shape=[64], const_data=[0.1] * 64))
    mm = g.add_node(Node(3, "MATMUL", inputs=[x, w], shape=[-1, 64]))
    add = g.add_node(Node(4, "ADD", inputs=[mm, bias], shape=[-1, 64]))
    relu = g.add_node(Node(5, "RELU", inputs=[add], shape=[-1, 64]))
    g.set_inputs([x])
    g.set_outputs([relu])
    graph_json = g.to_json()

    dir_a = tmp_path / "proc_a"
    dir_b = tmp_path / "proc_b"
    dir_a.mkdir()
    dir_b.mkdir()

    json_a = str(dir_a / "model.json")
    json_b = str(dir_b / "model.json")
    cpp_a = str(dir_a / "kernel.cpp")
    cpp_b = str(dir_b / "kernel.cpp")

    with open(json_a, "w", encoding="utf-8") as f:
        f.write(graph_json)
    with open(json_b, "w", encoding="utf-8") as f:
        f.write(graph_json)

    # Invoke Process A independently
    cmd_a = [
        ENGINE_EXE, "--compile", json_a, "-o", cpp_a,
        "--variant", "fused_ikj", "--candidate-id", "indep_cand_01", "--fusion"
    ]
    res_a = subprocess.run(cmd_a, capture_output=True, text=True, cwd=str(dir_a))
    assert res_a.returncode == 0, f"Process A failed: {res_a.stderr}"
    assert os.path.isfile(cpp_a)

    # Invoke Process B independently
    cmd_b = [
        ENGINE_EXE, "--compile", json_b, "-o", cpp_b,
        "--variant", "fused_ikj", "--candidate-id", "indep_cand_01", "--fusion"
    ]
    res_b = subprocess.run(cmd_b, capture_output=True, text=True, cwd=str(dir_b))
    assert res_b.returncode == 0, f"Process B failed: {res_b.stderr}"
    assert os.path.isfile(cpp_b)

    # Compare raw bytes and cryptographic digests
    bytes_a = _read_bytes(cpp_a)
    bytes_b = _read_bytes(cpp_b)
    sha_a = hashlib.sha256(bytes_a).hexdigest()
    sha_b = hashlib.sha256(bytes_b).hexdigest()

    assert sha_a == sha_b, f"SHA-256 mismatch between independent runs:\nRun A: {sha_a}\nRun B: {sha_b}"
    assert bytes_a == bytes_b, "Byte-for-byte discrepancy detected across independent CLI executions!"
    assert len(bytes_a) > 0, "Generated kernel is unexpectedly empty"


def test_independent_python_subprocess_determinism(tmp_path):
    """
    Spawns two completely separate Python interpreter processes, importing Forgetune
    in fresh runtimes with isolated environments, writing to disjoint paths.
    Verifies bitwise identity and SHA-256 equivalence.
    """
    dir_a = tmp_path / "py_proc_a"
    dir_b = tmp_path / "py_proc_b"
    dir_a.mkdir()
    dir_b.mkdir()

    out_a = str(dir_a / "kernel.cpp")
    out_b = str(dir_b / "kernel.cpp")

    worker_code = """
import sys
import os
out_path = sys.argv[1]
from forgetune.graph_adapter import Graph, Node
from forgetune.compiler import emit_cpp_source
from forgetune.autotuner import KernelCandidate

g = Graph()
x = g.add_node(Node(0, "INPUT", shape=[-1, 16]))
w = g.add_node(Node(1, "CONST", shape=[16, 32], const_data=[0.02] * (16 * 32)))
mm = g.add_node(Node(2, "MATMUL", inputs=[x, w], shape=[-1, 32]))
r = g.add_node(Node(3, "RELU", inputs=[mm], shape=[-1, 32]))
g.set_inputs([x])
g.set_outputs([r])

cand = KernelCandidate("cand_py_det", "loop_ikj", enable_fusion=True)
emit_cpp_source(g, cand, out_path)
"""

    env = os.environ.copy()
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    env["PYTHONPATH"] = f"{os.path.join(root_dir, 'python')};{root_dir}"

    # Independent Subprocess 1
    p1 = subprocess.run([sys.executable, "-c", worker_code, out_a], env=env, capture_output=True, text=True, cwd=str(dir_a))
    assert p1.returncode == 0, f"Python subprocess 1 failed: {p1.stderr}"
    assert os.path.isfile(out_a)

    # Independent Subprocess 2
    p2 = subprocess.run([sys.executable, "-c", worker_code, out_b], env=env, capture_output=True, text=True, cwd=str(dir_b))
    assert p2.returncode == 0, f"Python subprocess 2 failed: {p2.stderr}"
    assert os.path.isfile(out_b)

    bytes_a = _read_bytes(out_a)
    bytes_b = _read_bytes(out_b)
    sha_a = hashlib.sha256(bytes_a).hexdigest()
    sha_b = hashlib.sha256(bytes_b).hexdigest()

    assert sha_a == sha_b, f"Independent Python process SHA-256 mismatch:\n{sha_a} != {sha_b}"
    assert bytes_a == bytes_b, "Byte mismatch between independent Python subprocesses!"


def test_independent_determinism_multi_operator_v2_conv2d(tmp_path):
    """
    Tests determinism on a full V2 multi-operator pipeline containing:
    Conv2D -> ReLU -> Flatten -> MatMul across separate CLI runs.
    """
    assert ENGINE_EXE is not None and os.path.isfile(ENGINE_EXE)

    g = Graph()
    x = g.add_node(Node(0, "INPUT", shape=[-1, 1, 8, 8]))
    w_conv = g.add_node(Node(1, "CONST", shape=[2, 1, 3, 3], const_data=[0.1] * 18))
    conv = g.add_node(Node(2, "CONV2D", inputs=[x, w_conv], shape=[-1, 2, 6, 6]))
    relu = g.add_node(Node(3, "RELU", inputs=[conv], shape=[-1, 2, 6, 6]))
    flat = g.add_node(Node(4, "FLATTEN", inputs=[relu], shape=[-1, 72]))
    w_fc = g.add_node(Node(5, "CONST", shape=[72, 10], const_data=[0.01] * 720))
    fc = g.add_node(Node(6, "MATMUL", inputs=[flat, w_fc], shape=[-1, 10]))
    g.set_inputs([x])
    g.set_outputs([fc])
    graph_json = g.to_json()

    dir1 = tmp_path / "v2_run_1"
    dir2 = tmp_path / "v2_run_2"
    dir1.mkdir()
    dir2.mkdir()

    j1, j2 = str(dir1 / "model.json"), str(dir2 / "model.json")
    c1, c2 = str(dir1 / "kernel.cpp"), str(dir2 / "kernel.cpp")

    with open(j1, "w", encoding="utf-8") as f:
        f.write(graph_json)
    with open(j2, "w", encoding="utf-8") as f:
        f.write(graph_json)

    res1 = subprocess.run([ENGINE_EXE, "--compile", j1, "-o", c1, "--fusion", "--candidate-id", "v2_det_01"],
                          capture_output=True, text=True, cwd=str(dir1))
    assert res1.returncode == 0, f"Run 1 failed: {res1.stderr}"

    res2 = subprocess.run([ENGINE_EXE, "--compile", j2, "-o", c2, "--fusion", "--candidate-id", "v2_det_01"],
                          capture_output=True, text=True, cwd=str(dir2))
    assert res2.returncode == 0, f"Run 2 failed: {res2.stderr}"

    sha1 = _compute_sha256(c1)
    sha2 = _compute_sha256(c2)
    assert sha1 == sha2, f"V2 multi-op determinism mismatch: {sha1} vs {sha2}"
    assert _read_bytes(c1) == _read_bytes(c2)


def test_independent_determinism_detects_intentional_discrepancy(tmp_path):
    """
    Negative verification: Proves that the determinism verification harness
    definitively detects and flags any intentional discrepancy.
    Tests both:
      1. Semantic discrepancy (different candidate_id or tile size)
      2. Single-byte mutation in generated code
    """
    assert ENGINE_EXE is not None and os.path.isfile(ENGINE_EXE)

    g = Graph()
    x = g.add_node(Node(0, "INPUT", shape=[-1, 8]))
    w = g.add_node(Node(1, "CONST", shape=[8, 8], const_data=[1.0] * 64))
    mm = g.add_node(Node(2, "MATMUL", inputs=[x, w], shape=[-1, 8]))
    g.set_inputs([x])
    g.set_outputs([mm])
    graph_json = g.to_json()

    dir_base = tmp_path / "base"
    dir_diff = tmp_path / "diff"
    dir_base.mkdir()
    dir_diff.mkdir()

    j_base = str(dir_base / "model.json")
    j_diff = str(dir_diff / "model.json")
    c_base = str(dir_base / "kernel.cpp")
    c_diff = str(dir_diff / "kernel.cpp")

    with open(j_base, "w", encoding="utf-8") as f:
        f.write(graph_json)
    with open(j_diff, "w", encoding="utf-8") as f:
        f.write(graph_json)

    # Run base configuration
    subprocess.run([ENGINE_EXE, "--compile", j_base, "-o", c_base, "--candidate-id", "cand_A"], check=True)

    # Run intentionally perturbed configuration (different candidate_id)
    subprocess.run([ENGINE_EXE, "--compile", j_diff, "-o", c_diff, "--candidate-id", "cand_B"], check=True)

    sha_base = _compute_sha256(c_base)
    sha_diff = _compute_sha256(c_diff)
    bytes_base = _read_bytes(c_base)
    bytes_diff = _read_bytes(c_diff)

    # Must detect discrepancy
    assert sha_base != sha_diff, "Assertion failed: Test harness failed to detect candidate_id discrepancy!"
    assert bytes_base != bytes_diff, "Assertion failed: Byte comparison failed to detect difference!"

    # Test single-byte mutation detection
    mutated_bytes = bytearray(bytes_base)
    mutated_bytes[10] ^= 0x01  # Flip one bit
    mutated_sha = hashlib.sha256(mutated_bytes).hexdigest()

    assert mutated_sha != sha_base, "Assertion failed: SHA-256 did not detect single-bit mutation!"
    assert mutated_bytes != bytes_base, "Assertion failed: Byte equality failed to detect mutation!"
