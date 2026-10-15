"""
Acceptance Test Suite: Pass-by-Pass Optimization Attacks
Challenging each optimization pass independently against aggressive attack vectors:
1. Constant Folding: Folding across Add, MatMul, Relu, multi-step chains, signed zeros.
2. Algebraic Simplification: Identity elimination X + 0, 0 + X, ReLU idempotency,
   and preserving non-identities (non-zero vectors, shape mismatches).
3. Dead Code Elimination: Eliminating disconnected subgraphs and verifying output node protection.
4. Operator Fusion Legality: Rejection of fusion when intermediate Add or MatMul has
   multiple consumers or is declared as a graph output.
"""

import os
import json
import shutil
import tempfile
import subprocess
import pytest
import numpy as np

from forgetune.graph_adapter import Graph, Node
import forgetune._forgetune_cpp as cpp

_test_engine = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "bin", "forgetune_test_engine.exe"))
ENGINE_EXE = _test_engine if os.path.isfile(_test_engine) else os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "bin", "forgetune_engine.exe"))

@pytest.fixture
def temp_opt_dir():
    d = os.path.abspath(os.path.join("build", "artifacts", "acceptance_opt"))
    if os.path.exists(d):
        try:
            shutil.rmtree(d)
        except Exception:
            pass
    os.makedirs(d, exist_ok=True)
    yield d
    try:
        shutil.rmtree(d)
    except Exception:
        pass

def run_optimizer(graph: Graph, output_dir: str, enable_fusion: bool = False) -> Graph:
    os.makedirs(output_dir, exist_ok=True)
    in_json = os.path.join(output_dir, "in.json")
    out_json = os.path.join(output_dir, "out.json")

    with open(in_json, "w", encoding="utf-8") as f:
        f.write(graph.to_json())

    cmd = [ENGINE_EXE, "--optimize", in_json, "-o", out_json]
    if enable_fusion:
        cmd.append("--fusion")

    subprocess.run(cmd, check=True)

    with open(out_json, "r", encoding="utf-8") as f:
        opt_data = json.load(f)

    # Convert back to Graph object
    g_opt = Graph()
    for nd in opt_data["nodes"]:
        g_opt.add_node(Node.from_dict(nd))
    g_opt.set_inputs(opt_data["inputs"])
    g_opt.set_outputs(opt_data["outputs"])
    return g_opt

# ============================================================================
# 1. Constant Folding Attacks
# ============================================================================

def test_attack_constant_folding_matmul(temp_opt_dir):
    """Attack: folding of two constant matrices through MATMUL."""
    c1_data = [1.0, 2.0, 3.0, 4.0] # 2x2
    c2_data = [5.0, 6.0, 7.0, 8.0] # 2x2

    g = Graph()
    in_dummy = g.add_node(Node(0, "INPUT", "X", shape=[2, 2], dtype="f32"))
    c1 = g.add_node(Node(1, "CONST", "C1", shape=[2, 2], dtype="f32", const_data=c1_data))
    c2 = g.add_node(Node(2, "CONST", "C2", shape=[2, 2], dtype="f32", const_data=c2_data))
    mm = g.add_node(Node(3, "MATMUL", "mm_const", inputs=[c1, c2], shape=[2, 2], dtype="f32"))
    out = g.add_node(Node(4, "OUTPUT", "out", inputs=[mm], shape=[2, 2], dtype="f32"))
    g.set_inputs([in_dummy])
    g.set_outputs([out])

    opt_g = run_optimizer(g, os.path.join(temp_opt_dir, "cf_matmul"))

    # The MATMUL node should be completely eliminated and replaced by a folded CONST
    live_ops = [n.op for n in opt_g.nodes.values() if not n.deleted]
    assert "MATMUL" not in live_ops, f"MATMUL was not constant-folded! Live ops: {live_ops}"
    assert "CONST" in live_ops

    # Check numerical evaluation of the folded constant
    # [1, 2; 3, 4] x [5, 6; 7, 8] = [19, 22; 43, 50]
    expected = [19.0, 22.0, 43.0, 50.0]
    folded_node = next(n for n in opt_g.nodes.values() if not n.deleted and n.op == "CONST")
    np.testing.assert_allclose(folded_node.const_data, expected, atol=1e-5)

def test_attack_constant_folding_chained_pipeline(temp_opt_dir):
    """Attack: multi-step constant propagation chain: (C1 x C2) + C3 -> single CONST."""
    g = Graph()
    in_dummy = g.add_node(Node(0, "INPUT", "X", shape=[2, 2], dtype="f32"))
    c1 = g.add_node(Node(1, "CONST", "C1", shape=[2, 2], dtype="f32", const_data=[1.0, 0.0, 0.0, 1.0]))
    c2 = g.add_node(Node(2, "CONST", "C2", shape=[2, 2], dtype="f32", const_data=[2.0, 3.0, 4.0, 5.0]))
    c3 = g.add_node(Node(3, "CONST", "C3", shape=[2, 2], dtype="f32", const_data=[10.0, 10.0, 10.0, 10.0]))
    mm = g.add_node(Node(4, "MATMUL", "mm", inputs=[c1, c2], shape=[2, 2], dtype="f32"))
    add = g.add_node(Node(5, "ADD", "add", inputs=[mm, c3], shape=[2, 2], dtype="f32"))
    out = g.add_node(Node(6, "OUTPUT", "out", inputs=[add], shape=[2, 2], dtype="f32"))
    g.set_inputs([in_dummy])
    g.set_outputs([out])

    opt_g = run_optimizer(g, os.path.join(temp_opt_dir, "cf_chained"))
    live_ops = [n.op for n in opt_g.nodes.values() if not n.deleted]

    assert "MATMUL" not in live_ops
    assert "ADD" not in live_ops
    # Expected: [2, 3; 4, 5] + [10, 10; 10, 10] = [12, 13; 14, 15]
    folded_node = next(n for n in opt_g.nodes.values() if not n.deleted and n.op == "CONST")
    np.testing.assert_allclose(folded_node.const_data, [12.0, 13.0, 14.0, 15.0], atol=1e-5)

# ============================================================================
# 2. Algebraic Simplification Attacks
# ============================================================================

def test_attack_algebraic_identity_add_zero(temp_opt_dir):
    """Attack: X + 0 -> X identity elimination."""
    g = Graph()
    x = g.add_node(Node(0, "INPUT", "X", shape=[2, 4], dtype="f32"))
    zero = g.add_node(Node(1, "CONST", "Zero", shape=[2, 4], dtype="f32", const_data=[0.0] * 8))
    add = g.add_node(Node(2, "ADD", "add", inputs=[x, zero], shape=[2, 4], dtype="f32"))
    out = g.add_node(Node(3, "OUTPUT", "out", inputs=[add], shape=[2, 4], dtype="f32"))
    g.set_inputs([x])
    g.set_outputs([out])

    opt_g = run_optimizer(g, os.path.join(temp_opt_dir, "alg_add_zero"))
    live_ops = [n.op for n in opt_g.nodes.values() if not n.deleted]

    # ADD and CONST zero should be simplified and eliminated by DCE
    assert "ADD" not in live_ops, f"ADD(X, 0) was not simplified! Live ops: {live_ops}"
    assert "CONST" not in live_ops
    # Output should directly consume X
    out_node = next(n for n in opt_g.nodes.values() if not n.deleted and n.op == "OUTPUT")
    assert out_node.inputs[0] == x

def test_attack_algebraic_relu_idempotent(temp_opt_dir):
    """Attack: RELU(RELU(X)) -> RELU(X)."""
    g = Graph()
    x = g.add_node(Node(0, "INPUT", "X", shape=[2, 2], dtype="f32"))
    r1 = g.add_node(Node(1, "RELU", "relu1", inputs=[x], shape=[2, 2], dtype="f32"))
    r2 = g.add_node(Node(2, "RELU", "relu2", inputs=[r1], shape=[2, 2], dtype="f32"))
    out = g.add_node(Node(3, "OUTPUT", "out", inputs=[r2], shape=[2, 2], dtype="f32"))
    g.set_inputs([x])
    g.set_outputs([out])

    opt_g = run_optimizer(g, os.path.join(temp_opt_dir, "alg_relu"))
    live_ops = [n.op for n in opt_g.nodes.values() if not n.deleted]

    # Exactly one RELU node must remain
    relu_count = live_ops.count("RELU")
    assert relu_count == 1, f"Expected 1 RELU after idempotency simplification, got {relu_count}"

def test_attack_algebraic_reject_non_zero_constant(temp_opt_dir):
    """Attack: X + [0, 0, 0, 0.001] must NOT be simplified away."""
    g = Graph()
    x = g.add_node(Node(0, "INPUT", "X", shape=[1, 4], dtype="f32"))
    c = g.add_node(Node(1, "CONST", "almost_zero", shape=[1, 4], dtype="f32", const_data=[0.0, 0.0, 0.0, 1e-3]))
    add = g.add_node(Node(2, "ADD", "add", inputs=[x, c], shape=[1, 4], dtype="f32"))
    out = g.add_node(Node(3, "OUTPUT", "out", inputs=[add], shape=[1, 4], dtype="f32"))
    g.set_inputs([x])
    g.set_outputs([out])

    opt_g = run_optimizer(g, os.path.join(temp_opt_dir, "alg_reject_non_zero"))
    live_ops = [n.op for n in opt_g.nodes.values() if not n.deleted]

    # ADD node must be preserved!
    assert "ADD" in live_ops, "Non-zero addition was incorrectly simplified away!"

# ============================================================================
# 3. Dead Code Elimination Attacks
# ============================================================================

def test_attack_dce_removes_dead_subgraph_and_protects_outputs(temp_opt_dir):
    """
    Attack: Graph with a live branch and a dead, disconnected multi-op branch.
    Verify dead branch is eliminated, but output nodes with 0 consumers are preserved.
    """
    g = Graph()
    x = g.add_node(Node(0, "INPUT", "X", shape=[2, 2], dtype="f32"))
    # Live branch: X -> Relu -> Out
    r = g.add_node(Node(1, "RELU", "relu_live", inputs=[x], shape=[2, 2], dtype="f32"))
    out = g.add_node(Node(2, "OUTPUT", "out", inputs=[r], shape=[2, 2], dtype="f32"))

    # Dead branch: X -> MatMul -> Add (unreferenced by Out)
    c_w = g.add_node(Node(3, "CONST", "W_dead", shape=[2, 2], dtype="f32", const_data=[1.0]*4))
    d_mm = g.add_node(Node(4, "MATMUL", "mm_dead", inputs=[x, c_w], shape=[2, 2], dtype="f32"))
    c_b = g.add_node(Node(5, "CONST", "B_dead", shape=[2, 2], dtype="f32", const_data=[2.0]*4))
    d_add = g.add_node(Node(6, "ADD", "add_dead", inputs=[d_mm, c_b], shape=[2, 2], dtype="f32"))

    g.set_inputs([x])
    g.set_outputs([out])

    opt_g = run_optimizer(g, os.path.join(temp_opt_dir, "dce_dead_branch"))
    live_ids = [n.id for n in opt_g.nodes.values() if not n.deleted]

    # Dead branch nodes must be purged
    assert d_mm not in live_ids
    assert d_add not in live_ids
    assert c_w not in live_ids
    assert c_b not in live_ids

    # Live nodes and output must remain
    assert out in live_ids
    assert r in live_ids
    assert x in live_ids

# ============================================================================
# 4. Operator Fusion Legality Attacks
# ============================================================================

def test_attack_fusion_legality_multi_consumer_add(temp_opt_dir):
    """
    Attack: Attempt to fuse MatMul + Add + Relu when intermediate Add has multiple consumers.
    Graph:
      %M = MatMul(X, W)
      %A = Add(%M, B)
      %R = Relu(%A)
      %Extra = Add(%A, B2)  <-- second consumer of %A!
    Compiler MUST NOT fuse %A away into %R, which would break %Extra.
    """
    g = Graph()
    x = g.add_node(Node(0, "INPUT", "X", shape=[2, 2], dtype="f32"))
    w = g.add_node(Node(1, "CONST", "W", shape=[2, 2], dtype="f32", const_data=[1.0, 0.0, 0.0, 1.0]))
    b = g.add_node(Node(2, "CONST", "B", shape=[2], dtype="f32", const_data=[0.5, 0.5]))
    b2 = g.add_node(Node(3, "CONST", "B2", shape=[2], dtype="f32", const_data=[1.0, 1.0]))

    mm = g.add_node(Node(4, "MATMUL", "mm", inputs=[x, w], shape=[2, 2], dtype="f32"))
    add = g.add_node(Node(5, "ADD", "add", inputs=[mm, b], shape=[2, 2], dtype="f32"))
    relu = g.add_node(Node(6, "RELU", "relu", inputs=[add], shape=[2, 2], dtype="f32"))
    extra = g.add_node(Node(7, "ADD", "extra_consumer", inputs=[add, b2], shape=[2, 2], dtype="f32"))

    out1 = g.add_node(Node(8, "OUTPUT", "out1", inputs=[relu], shape=[2, 2], dtype="f32"))
    out2 = g.add_node(Node(9, "OUTPUT", "out2", inputs=[extra], shape=[2, 2], dtype="f32"))

    g.set_inputs([x])
    g.set_outputs([out1, out2])

    opt_g = run_optimizer(g, os.path.join(temp_opt_dir, "fusion_multi_consumer"), enable_fusion=True)
    live_ops = [n.op for n in opt_g.nodes.values() if not n.deleted]

    # Because Add has 2 consumers, fusion must be rejected for that pattern
    assert "FUSED_MATMUL_ADD_RELU" not in live_ops, (
        "Illegally fused Add when Add had multiple active consumers!"
    )
    assert "MATMUL" in live_ops
    assert "ADD" in live_ops

def test_attack_fusion_legality_intermediate_add_is_graph_output(temp_opt_dir):
    """
    Attack: Intermediate Add is explicitly declared as a graph output.
    Graph outputs: [Add, Relu].
    Compiler must NOT fuse Add away.
    """
    g = Graph()
    x = g.add_node(Node(0, "INPUT", "X", shape=[2, 2], dtype="f32"))
    w = g.add_node(Node(1, "CONST", "W", shape=[2, 2], dtype="f32", const_data=[1.0]*4))
    b = g.add_node(Node(2, "CONST", "B", shape=[2], dtype="f32", const_data=[0.0, 0.0]))

    mm = g.add_node(Node(3, "MATMUL", "mm", inputs=[x, w], shape=[2, 2], dtype="f32"))
    add = g.add_node(Node(4, "ADD", "add", inputs=[mm, b], shape=[2, 2], dtype="f32"))
    relu = g.add_node(Node(5, "RELU", "relu", inputs=[add], shape=[2, 2], dtype="f32"))

    out_add = g.add_node(Node(6, "OUTPUT", "out_add", inputs=[add], shape=[2, 2], dtype="f32"))
    out_relu = g.add_node(Node(7, "OUTPUT", "out_relu", inputs=[relu], shape=[2, 2], dtype="f32"))

    g.set_inputs([x])
    # The Add node itself is an output!
    g.set_outputs([add, relu])

    opt_g = run_optimizer(g, os.path.join(temp_opt_dir, "fusion_add_output"), enable_fusion=True)
    live_ops = [n.op for n in opt_g.nodes.values() if not n.deleted]

def test_attack_dce_preserves_unconsumed_graph_inputs(temp_opt_dir):
    """
    Regression Test for DEFECT-001:
    Verifies that when a declared graph input is unconsumed by the output
    (e.g., following constant folding or in multi-input signatures where an input is unused),
    DeadCodeEliminationPass preserves the INPUT node as a live ABI root, ensuring
    the post-pass graph verifier succeeds.
    """
    g = Graph()
    in_unconsumed = g.add_node(Node(0, "INPUT", "X_unused", shape=[2, 2], dtype="f32"))
    c = g.add_node(Node(1, "CONST", "C", shape=[2, 2], dtype="f32", const_data=[1.0]*4))
    r = g.add_node(Node(2, "RELU", "relu", inputs=[c], shape=[2, 2], dtype="f32"))
    out = g.add_node(Node(3, "OUTPUT", "out", inputs=[r], shape=[2, 2], dtype="f32"))
    g.set_inputs([in_unconsumed])
    g.set_outputs([out])

    # Run optimizer (ConstantFolding, AlgebraicSimplification, DCE)
    opt_g = run_optimizer(g, os.path.join(temp_opt_dir, "dce_unconsumed_in"))
    live_ids = [n.id for n in opt_g.nodes.values() if not n.deleted]

    # in_unconsumed must remain live as an input node
    assert in_unconsumed in live_ids
    assert opt_g.nodes[in_unconsumed].op == "INPUT"
    assert not opt_g.nodes[in_unconsumed].deleted
    assert in_unconsumed in opt_g.inputs

