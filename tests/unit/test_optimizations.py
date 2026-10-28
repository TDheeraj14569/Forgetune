"""
Category C: Optimization Pass Tests.
Tests constant folding, algebraic simplification, dead-code elimination, and fusion.
"""

import os
import subprocess
import json
import pytest
from forgetune.graph_adapter import Graph, Node

_test_engine = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "bin", "forgetune_test_engine.exe"))
ENGINE_EXE = _test_engine if os.path.isfile(_test_engine) else os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "bin", "forgetune_engine.exe"))

def test_engine_optimization_pipeline():
    # Build a graph with constant folding and dead code
    g = Graph()
    x = g.add_node(Node(0, "INPUT", shape=[2, 2]))
    c1 = g.add_node(Node(1, "CONST", shape=[2, 2], const_data=[1.0, 2.0, 3.0, 4.0]))
    c2 = g.add_node(Node(2, "CONST", shape=[2, 2], const_data=[2.0, 2.0, 2.0, 2.0]))
    # add_c can be constant folded
    add_c = g.add_node(Node(3, "ADD", inputs=[c1, c2], shape=[2, 2]))
    # Live node
    live_out = g.add_node(Node(4, "ADD", inputs=[x, add_c], shape=[2, 2]))
    # Dead node
    dead_node = g.add_node(Node(5, "RELU", inputs=[x], shape=[2, 2]))

    g.set_inputs([x])
    g.set_outputs([live_out])

    os.makedirs("build/test_opt", exist_ok=True)
    in_json = "build/test_opt/input.json"
    out_json = "build/test_opt/opt.json"

    with open(in_json, "w", encoding="utf-8") as f:
        f.write(g.to_json())

    p = subprocess.run([ENGINE_EXE, "--optimize", in_json, "-o", out_json], capture_output=True, text=True)
    assert p.returncode == 0, f"Optimization failed: {p.stderr}"

    with open(out_json, "r", encoding="utf-8") as f:
        opt_graph = json.load(f)

    # Check that dead node 5 is marked deleted or removed
    for n in opt_graph["nodes"]:
        if n["id"] == 5:
            assert n.get("deleted") is True

def test_engine_operator_fusion():
    # MATMUL -> ADD -> RELU
    g = Graph()
    x = g.add_node(Node(0, "INPUT", shape=[2, 4]))
    w = g.add_node(Node(1, "CONST", shape=[4, 8], const_data=[1.0] * 32))
    b = g.add_node(Node(2, "CONST", shape=[8], const_data=[0.5] * 8))
    mm = g.add_node(Node(3, "MATMUL", inputs=[x, w], shape=[2, 8]))
    add = g.add_node(Node(4, "ADD", inputs=[mm, b], shape=[2, 8]))
    relu = g.add_node(Node(5, "RELU", inputs=[add], shape=[2, 8]))

    g.set_inputs([x])
    g.set_outputs([relu])

    os.makedirs("build/test_opt", exist_ok=True)
    in_json = "build/test_opt/fusion_in.json"
    out_json = "build/test_opt/fusion_out.json"

    with open(in_json, "w", encoding="utf-8") as f:
        f.write(g.to_json())

    p = subprocess.run([ENGINE_EXE, "--optimize", in_json, "-o", out_json, "--fusion"], capture_output=True, text=True)
    assert p.returncode == 0, f"Fusion failed: {p.stderr}"

    with open(out_json, "r", encoding="utf-8") as f:
        opt_graph = json.load(f)

    fused_found = False
    for n in opt_graph["nodes"]:
        if n["op"] == "FUSED_MATMUL_ADD_RELU" and not n.get("deleted", False):
            fused_found = True
            break
    assert fused_found, "Operator fusion must produce FUSED_MATMUL_ADD_RELU node"

def test_algebraic_simplification_add_zero():
    # Build: Y = ADD(X, 0)
    g = Graph()
    x = g.add_node(Node(0, "INPUT", shape=[2, 3]))
    zero = g.add_node(Node(1, "CONST", shape=[2, 3], const_data=[0.0] * 6))
    add = g.add_node(Node(2, "ADD", inputs=[x, zero], shape=[2, 3]))
    g.set_inputs([x])
    g.set_outputs([add])

    os.makedirs("build/test_opt", exist_ok=True)
    in_json = "build/test_opt/as_zero_in.json"
    out_json = "build/test_opt/as_zero_out.json"
    with open(in_json, "w", encoding="utf-8") as f:
        f.write(g.to_json())

    p = subprocess.run([ENGINE_EXE, "--optimize", in_json, "-o", out_json], capture_output=True, text=True)
    assert p.returncode == 0
    with open(out_json, "r", encoding="utf-8") as f:
        res = json.load(f)

    # Output should now point directly to x (node 0)
    assert res["outputs"] == [0], f"Expected output to be rewritten to node 0, got {res['outputs']}"

def test_algebraic_simplification_relu_idempotent():
    # Build: Y = RELU(RELU(X))
    g = Graph()
    x = g.add_node(Node(0, "INPUT", shape=[2, 2]))
    r1 = g.add_node(Node(1, "RELU", inputs=[x], shape=[2, 2]))
    r2 = g.add_node(Node(2, "RELU", inputs=[r1], shape=[2, 2]))
    g.set_inputs([x])
    g.set_outputs([r2])

    os.makedirs("build/test_opt", exist_ok=True)
    in_json = "build/test_opt/as_relu_in.json"
    out_json = "build/test_opt/as_relu_out.json"
    with open(in_json, "w", encoding="utf-8") as f:
        f.write(g.to_json())

    p = subprocess.run([ENGINE_EXE, "--optimize", in_json, "-o", out_json], capture_output=True, text=True)
    assert p.returncode == 0
    with open(out_json, "r", encoding="utf-8") as f:
        res = json.load(f)

    # Output should now point to r1 (node 1)
    assert res["outputs"] == [1], f"Expected output to be rewritten to node 1, got {res['outputs']}"

def test_fusion_multi_consumer_rejected():
    # If MATMUL is consumed by ADD AND another node, fusion MUST NOT delete it or fuse it!
    g = Graph()
    x = g.add_node(Node(0, "INPUT", shape=[2, 4]))
    w = g.add_node(Node(1, "CONST", shape=[4, 8], const_data=[1.0] * 32))
    b = g.add_node(Node(2, "CONST", shape=[8], const_data=[0.5] * 8))
    mm = g.add_node(Node(3, "MATMUL", inputs=[x, w], shape=[2, 8]))
    add = g.add_node(Node(4, "ADD", inputs=[mm, b], shape=[2, 8]))
    relu = g.add_node(Node(5, "RELU", inputs=[add], shape=[2, 8]))
    # Extra consumer of mm
    extra_relu = g.add_node(Node(6, "RELU", inputs=[mm], shape=[2, 8]))
    final = g.add_node(Node(7, "ADD", inputs=[relu, extra_relu], shape=[2, 8]))

    g.set_inputs([x])
    g.set_outputs([final])

    os.makedirs("build/test_opt", exist_ok=True)
    in_json = "build/test_opt/fusion_reject_in.json"
    out_json = "build/test_opt/fusion_reject_out.json"
    with open(in_json, "w", encoding="utf-8") as f:
        f.write(g.to_json())

    p = subprocess.run([ENGINE_EXE, "--optimize", in_json, "-o", out_json, "--fusion"], capture_output=True, text=True)
    assert p.returncode == 0
    with open(out_json, "r", encoding="utf-8") as f:
        res = json.load(f)

    # Fused node must NOT be created
    fused_found = any(n["op"] == "FUSED_MATMUL_ADD_RELU" and not n.get("deleted", False) for n in res["nodes"])
    assert not fused_found, "Fusion must be rejected when MATMUL has multiple consumers"

def test_fusion_output_node_rejected():
    # If MATMUL is declared as a graph output, it must NOT be fused away!
    g = Graph()
    x = g.add_node(Node(0, "INPUT", shape=[2, 4]))
    w = g.add_node(Node(1, "CONST", shape=[4, 8], const_data=[1.0] * 32))
    b = g.add_node(Node(2, "CONST", shape=[8], const_data=[0.5] * 8))
    mm = g.add_node(Node(3, "MATMUL", inputs=[x, w], shape=[2, 8]))
    add = g.add_node(Node(4, "ADD", inputs=[mm, b], shape=[2, 8]))
    relu = g.add_node(Node(5, "RELU", inputs=[add], shape=[2, 8]))
    final = g.add_node(Node(6, "ADD", inputs=[relu, mm], shape=[2, 8]))

    g.set_inputs([x])
    g.set_outputs([final])

    os.makedirs("build/test_opt", exist_ok=True)
    in_json = "build/test_opt/fusion_out_reject_in.json"
    out_json = "build/test_opt/fusion_out_reject_out.json"
    with open(in_json, "w", encoding="utf-8") as f:
        f.write(g.to_json())

    p = subprocess.run([ENGINE_EXE, "--optimize", in_json, "-o", out_json, "--fusion"], capture_output=True, text=True)
    assert p.returncode == 0
    with open(out_json, "r", encoding="utf-8") as f:
        res = json.load(f)

    fused_found = any(n["op"] == "FUSED_MATMUL_ADD_RELU" and not n.get("deleted", False) for n in res["nodes"])
    assert not fused_found, "Fusion must be rejected when intermediate node is a graph output"
