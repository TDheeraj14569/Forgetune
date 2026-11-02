"""
Category B: IR and Graph Tests.
Tests valid import, duplicate IDs, missing operands, cycles, and JSON roundtrip.
"""

import pytest
from forgetune.graph_adapter import Graph, Node
from forgetune.errors import InvalidGraphError

def test_duplicate_node_id_rejected():
    g = Graph()
    g.add_node(Node(0, "INPUT", shape=[2, 2]))
    with pytest.raises(InvalidGraphError):
        g.add_node(Node(0, "RELU", shape=[2, 2]))

def test_cyclic_graph_rejected():
    g = Graph()
    g.add_node(Node(0, "RELU", inputs=[1], shape=[2, 2]))
    g.add_node(Node(1, "RELU", inputs=[0], shape=[2, 2]))
    with pytest.raises(InvalidGraphError):
        g.topological_order()
    with pytest.raises(InvalidGraphError):
        g.validate()

def test_topological_sort_order():
    g = Graph()
    x = g.add_node(Node(0, "INPUT", shape=[2, 2]))
    r1 = g.add_node(Node(1, "RELU", inputs=[x], shape=[2, 2]))
    r2 = g.add_node(Node(2, "RELU", inputs=[r1], shape=[2, 2]))
    g.set_inputs([x])
    g.set_outputs([r2])

    order = g.topological_order()
    assert order == [0, 1, 2]
    # Verify central validate() method succeeds for valid graph
    g.validate()

    # Verify validate() rejects invalid operand dependency
    g_bad = Graph()
    in_bad = g_bad.add_node(Node(10, "INPUT", shape=[2, 2]))
    g_bad.add_node(Node(11, "RELU", inputs=[999], shape=[2, 2]))
    g_bad.set_inputs([in_bad])
    with pytest.raises(InvalidGraphError) as exc_info:
        g_bad.validate()
    assert "depends on non-existent or deleted node" in str(exc_info.value)

    # Verify validate() rejects non-positive static dimension
    g_bad_dim = Graph()
    g_bad_dim.add_node(Node(20, "INPUT", shape=[-2, 4]))
    g_bad_dim.set_inputs([20])
    with pytest.raises(InvalidGraphError) as exc_info:
        g_bad_dim.validate()
    assert "invalid shape dimension" in str(exc_info.value)

def test_json_serialization_and_deserialization():
    g = Graph()
    x = g.add_node(Node(0, "INPUT", shape=[2, 4]))
    c = g.add_node(Node(1, "CONST", shape=[4, 8], const_data=[1.0] * 32))
    mm = g.add_node(Node(2, "MATMUL", inputs=[x, c], shape=[2, 8]))
    g.set_inputs([x])
    g.set_outputs([mm])

    json_str = g.to_json()
    assert '"schema_version_major": 2' in json_str

    g2 = Graph.from_json(json_str)
    assert g2.inputs == [0]
    assert g2.outputs == [2]
    assert len(g2.nodes) == 3
    assert g2.nodes[1].const_data == [1.0] * 32

    # Backward compatibility: verify v1 schema JSON deserializes correctly
    v1_json = json_str.replace('"schema_version_major": 2', '"schema_version_major": 1')
    g_v1 = Graph.from_json(v1_json)
    assert g_v1.inputs == [0]
    assert len(g_v1.nodes) == 3
