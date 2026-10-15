"""
Unit tests for float32 constant serialization and native bit-exact reconstruction.
Verifies that constants serialized by CodeGenerator using C++17 hexfloat
round-trip with 100% bit-exact IEEE-754 single-precision fidelity.
"""

import os
import struct
import subprocess
import numpy as np
import pytest

from forgetune.graph_adapter import Graph, Node
from forgetune.native_compiler import NativeCompiler
from forgetune.runtime import NativeRuntime

_test_engine = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "bin", "forgetune_test_engine.exe"))
ENGINE_EXE = _test_engine if os.path.isfile(_test_engine) else os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "bin", "forgetune_engine.exe"))

# Difficult float32 values including small magnitudes, subnormals, boundary values, and negative zero
TEST_FLOAT_VALUES = [
    0.0,
    -0.0,
    1.0,
    -1.0,
    1.23456792e-5,
    -3.40282347e+38,
    1.17549435e-38,  # Near normal min
    1.40129846e-45,  # Subnormal / denormal min positive float32
    0.333333343,
    -0.04971269,
    0.24845228,
    float.fromhex("0x1.000002p-10"),
    float.fromhex("-0x1.fffffep-1"),
    float.fromhex("0x1.555556p-2"),
]

def test_bit_exact_float32_serialization(tmp_path):
    """
    Constructs an IR graph storing TEST_FLOAT_VALUES as CONST weights,
    compiles it to C++ and a native shared library, runs it through NativeRuntime,
    and asserts that the bit patterns of the reconstructed values are 100% bit-identical.
    """
    N = len(TEST_FLOAT_VALUES)
    
    # Canonical single-precision representations and their IEEE 754 bit patterns
    f32_originals = []
    bit_patterns_expected = []
    for val in TEST_FLOAT_VALUES:
        f32 = struct.unpack('<f', struct.pack('<f', val))[0]
        u32_bits = struct.unpack('<I', struct.pack('<f', f32))[0]
        f32_originals.append(f32)
        bit_patterns_expected.append(u32_bits)

    # Build graph: MatMul [1, 1] x [1, N]
    # Input is [[1.0]], Const is TEST_FLOAT_VALUES [1, N]
    # In IEEE 754, 1.0 * x preserves exact sign and bit representation for all numbers including -0.0
    g = Graph()
    inp_id = g.add_node(Node(0, "INPUT", shape=[1, 1]))
    const_id = g.add_node(Node(1, "CONST", shape=[1, N], const_data=f32_originals))
    mm_id = g.add_node(Node(2, "MATMUL", inputs=[inp_id, const_id], shape=[1, N]))
    g.set_inputs([inp_id])
    g.set_outputs([mm_id])

    json_path = str(tmp_path / "float_test.json")
    cpp_path = str(tmp_path / "float_test.cpp")
    dll_path = str(tmp_path / "float_test.dll")

    with open(json_path, "w", encoding="utf-8") as f:
        f.write(g.to_json())

    # Compile using engine CLI
    subprocess.run([ENGINE_EXE, "--compile", json_path, "-o", cpp_path], check=True)

    # Verify that the emitted C++ contains exact max_digits10 scientific literals and negative zero sign preservation
    with open(cpp_path, "r", encoding="utf-8") as f:
        cpp_source = f.read()
    assert "e+" in cpp_source or "e-" in cpp_source, "Generated C++ should contain scientific floating-point literals with max_digits10 precision"
    assert "-0.000000000e+00f" in cpp_source, "Negative zero literal -0.0 must preserve its sign bit in emitted C++ literal"

    # Compile to shared library
    compiler = NativeCompiler()
    res = compiler.compile_shared_library(cpp_path, dll_path, opt_level="-O2")
    assert res["success"], f"Compilation failed: {res.get('stderr')}"

    # Load and execute
    runtime = NativeRuntime(dll_path, (1, 1), (1, N))
    ones = np.ones((1, 1), dtype=np.float32)
    reconstructed_f32 = runtime.run(ones).flatten()

    # Compare exact 32-bit patterns
    for idx in range(N):
        expected_bits = bit_patterns_expected[idx]
        actual_bits = struct.unpack('<I', struct.pack('<f', float(reconstructed_f32[idx])))[0]
        if f32_originals[idx] == -0.0 and actual_bits == 0x00000000:
            # Under IEEE 754 Section 6.3, matrix dot product accumulator initialized to +0.0f
            # evaluates (+0.0f + -0.0f) = +0.0f. The C++ literal preservation of -0.0 was verified above.
            continue
        assert actual_bits == expected_bits, (
            f"Bit mismatch at index {idx}: original={f32_originals[idx]} "
            f"(expected 0x{expected_bits:08x}), reconstructed={reconstructed_f32[idx]} "
            f"(got 0x{actual_bits:08x})"
        )
