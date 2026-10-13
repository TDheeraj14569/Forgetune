"""
Acceptance Test Suite: Negative Mutation Testing & Graph Verifier Diagnostics
Systematically attacks the graph verifier and compiler frontend with corrupted,
malformed, cyclical, and dimensionally incompatible IR representations to ensure
the compiler fails closed with actionable diagnostic errors.
"""

import pytest
import forgetune._forgetune_cpp as cpp
from forgetune.graph_adapter import Graph, Node
from forgetune.errors import InvalidGraphError

def test_acceptance_mutation_cycle_detection():
    """Mutates graph to contain a circular dependency cycle."""
    g = Graph()
    in_id = g.add_node(Node(0, "INPUT", "X", shape=[2, 2], dtype="f32"))
    # Node 1 depends on Node 2, and Node 2 depends on Node 1
    r1 = g.add_node(Node(1, "RELU", "r1", inputs=[2], shape=[2, 2], dtype="f32"))
    r2 = g.add_node(Node(2, "RELU", "r2", inputs=[1], shape=[2, 2], dtype="f32"))
    out = g.add_node(Node(3, "OUTPUT", "out", inputs=[r2], shape=[2, 2], dtype="f32"))
    g.set_inputs([in_id])
    g.set_outputs([out])

    # In Python topological sort:
    with pytest.raises(InvalidGraphError) as exc_info:
        g.topological_order()
    assert "cycle" in str(exc_info.value).lower()

    # In C++ graph verifier:
    cpp_g = cpp.json_to_graph(g.to_json())
    vres = cpp.verify_graph(cpp_g, False)
    assert not vres.passed
    assert any("cycle" in e.lower() for e in vres.errors)

def test_acceptance_mutation_matmul_dimension_mismatch():
    """Mutates MatMul to have mismatched inner K dimensions: [2, 3] @ [4, 5]."""
    g = Graph()
    in_id = g.add_node(Node(0, "INPUT", "X", shape=[2, 3], dtype="f32"))
    w_id = g.add_node(Node(1, "CONST", "W", shape=[4, 5], dtype="f32", const_data=[1.0]*20))
    mm_id = g.add_node(Node(2, "MATMUL", "mm_bad", inputs=[in_id, w_id], shape=[2, 5], dtype="f32"))
    out_id = g.add_node(Node(3, "OUTPUT", "out", inputs=[mm_id], shape=[2, 5], dtype="f32"))
    g.set_inputs([in_id])
    g.set_outputs([out_id])

    cpp_g = cpp.json_to_graph(g.to_json())
    vres = cpp.verify_graph(cpp_g, False)
    assert not vres.passed
    assert any("dimension mismatch" in e.lower() or "k-dim" in e.lower() for e in vres.errors)

def test_acceptance_mutation_add_incompatible_broadcast_shapes():
    """Mutates Add to have incompatible shapes: [2, 3] + [3, 2]."""
    g = Graph()
    in_id = g.add_node(Node(0, "INPUT", "X", shape=[2, 3], dtype="f32"))
    c_id = g.add_node(Node(1, "CONST", "C", shape=[3, 2], dtype="f32", const_data=[1.0]*6))
    add_id = g.add_node(Node(2, "ADD", "add_bad", inputs=[in_id, c_id], shape=[2, 3], dtype="f32"))
    out_id = g.add_node(Node(3, "OUTPUT", "out", inputs=[add_id], shape=[2, 3], dtype="f32"))
    g.set_inputs([in_id])
    g.set_outputs([out_id])

    cpp_g = cpp.json_to_graph(g.to_json())
    vres = cpp.verify_graph(cpp_g, False)
    assert not vres.passed
    assert any("unsupported broadcast" in e.lower() for e in vres.errors)

def test_acceptance_mutation_const_buffer_size_mismatch():
    """Mutates CONST node: shape declares [2, 3] (6 elements), but buffer has 4 floats."""
    g = Graph()
    in_id = g.add_node(Node(0, "INPUT", "X", shape=[2, 2], dtype="f32"))
    c_bad = g.add_node(Node(1, "CONST", "bad_const", shape=[2, 3], dtype="f32", const_data=[1.0, 2.0, 3.0, 4.0]))
    out_id = g.add_node(Node(2, "OUTPUT", "out", inputs=[c_bad], shape=[2, 3], dtype="f32"))
    g.set_inputs([in_id])
    g.set_outputs([out_id])

    cpp_g = cpp.json_to_graph(g.to_json())
    vres = cpp.verify_graph(cpp_g, False)
    assert not vres.passed
    assert any("buffer size mismatch" in e.lower() for e in vres.errors)

def test_acceptance_mutation_missing_operand_reference():
    """Mutates node to refer to a non-existent operand ID (e.g. 999)."""
    g = Graph()
    in_id = g.add_node(Node(0, "INPUT", "X", shape=[2, 2], dtype="f32"))
    r = g.add_node(Node(1, "RELU", "r", inputs=[999], shape=[2, 2], dtype="f32"))
    out = g.add_node(Node(2, "OUTPUT", "out", inputs=[r], shape=[2, 2], dtype="f32"))
    g.set_inputs([in_id])
    g.set_outputs([out])

    cpp_g = cpp.json_to_graph(g.to_json())
    vres = cpp.verify_graph(cpp_g, False)
    assert not vres.passed
    assert any("non-existent node" in e.lower() for e in vres.errors)

def test_acceptance_mutation_empty_inputs_or_outputs():
    """Mutates graph to have empty input or output set."""
    g_no_in = Graph()
    c = g_no_in.add_node(Node(0, "CONST", "C", shape=[2, 2], dtype="f32", const_data=[1.0]*4))
    out = g_no_in.add_node(Node(1, "OUTPUT", "out", inputs=[c], shape=[2, 2], dtype="f32"))
    g_no_in.set_inputs([])
    g_no_in.set_outputs([out])

    cpp_g = cpp.json_to_graph(g_no_in.to_json())
    vres = cpp.verify_graph(cpp_g, False)
    assert not vres.passed
    assert any("no declared inputs" in e.lower() for e in vres.errors)

def test_acceptance_mutation_non_positive_dimensions():
    """Mutates shape to have non-positive or negative dimensions."""
    g = Graph()
    in_id = g.add_node(Node(0, "INPUT", "X", shape=[2, -4], dtype="f32"))
    out = g.add_node(Node(1, "OUTPUT", "out", inputs=[in_id], shape=[2, -4], dtype="f32"))
    g.set_inputs([in_id])
    g.set_outputs([out])

    cpp_g = cpp.json_to_graph(g.to_json())
    vres = cpp.verify_graph(cpp_g, False)
    assert not vres.passed
    assert any("non-positive dimension" in e.lower() for e in vres.errors)
