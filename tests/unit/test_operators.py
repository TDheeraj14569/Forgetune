"""
Category A: Operator Unit Tests.
Tests MATMUL, ADD, RELU, CONST under valid and invalid conditions.
"""

import pytest
import numpy as np
from forgetune.graph_adapter import Graph, Node
from forgetune.errors import InvalidShapeError

def test_matmul_valid_dimensions():
    # [2, 4] @ [4, 3] -> [2, 3]
    g = Graph()
    x = g.add_node(Node(0, "INPUT", shape=[2, 4]))
    w = g.add_node(Node(1, "INPUT", shape=[4, 3]))
    mm = g.add_node(Node(2, "MATMUL", inputs=[x, w], shape=[2, 3]))
    g.set_inputs([x, w])
    g.set_outputs([mm])

    x_data = np.ones((2, 4), dtype=np.float32)
    w_data = np.full((4, 3), 2.0, dtype=np.float32)
    res = g.evaluate({x: x_data, w: w_data})[mm]

    assert res.shape == (2, 3)
    assert np.allclose(res, 8.0)

def test_matmul_invalid_dimensions():
    # [2, 4] @ [5, 3] should raise InvalidShapeError
    g = Graph()
    x = g.add_node(Node(0, "INPUT", shape=[2, 4]))
    w = g.add_node(Node(1, "INPUT", shape=[5, 3]))
    mm = g.add_node(Node(2, "MATMUL", inputs=[x, w], shape=[2, 3]))
    g.set_inputs([x, w])
    g.set_outputs([mm])

    with pytest.raises(InvalidShapeError):
        g.evaluate({x: np.ones((2, 4), dtype=np.float32), w: np.ones((5, 3), dtype=np.float32)})

def test_matmul_non_square():
    # M=3, K=5, N=2
    g = Graph()
    x = g.add_node(Node(0, "INPUT", shape=[3, 5]))
    w = g.add_node(Node(1, "INPUT", shape=[5, 2]))
    mm = g.add_node(Node(2, "MATMUL", inputs=[x, w], shape=[3, 2]))
    g.set_inputs([x, w])
    g.set_outputs([mm])

    np.random.seed(1)
    x_val = np.random.randn(3, 5).astype(np.float32)
    w_val = np.random.randn(5, 2).astype(np.float32)
    expected = x_val @ w_val
    actual = g.evaluate({x: x_val, w: w_val})[mm]
    assert np.allclose(actual, expected, atol=1e-5)

def test_add_exact_shape():
    g = Graph()
    a = g.add_node(Node(0, "INPUT", shape=[2, 3]))
    b = g.add_node(Node(1, "INPUT", shape=[2, 3]))
    add = g.add_node(Node(2, "ADD", inputs=[a, b], shape=[2, 3]))
    g.set_inputs([a, b])
    g.set_outputs([add])

    a_val = np.full((2, 3), 3.0, dtype=np.float32)
    b_val = np.full((2, 3), 4.5, dtype=np.float32)
    res = g.evaluate({a: a_val, b: b_val})[add]
    assert np.allclose(res, 7.5)

def test_add_rank1_vector_broadcasting():
    # [2, 3] + [3] -> [2, 3]
    g = Graph()
    a = g.add_node(Node(0, "INPUT", shape=[2, 3]))
    b = g.add_node(Node(1, "INPUT", shape=[3]))
    add = g.add_node(Node(2, "ADD", inputs=[a, b], shape=[2, 3]))
    g.set_inputs([a, b])
    g.set_outputs([add])

    a_val = np.ones((2, 3), dtype=np.float32)
    b_val = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    res = g.evaluate({a: a_val, b: b_val})[add]
    expected = a_val + b_val
    assert np.allclose(res, expected)

def test_add_rank2_row_broadcasting():
    # [2, 3] + [1, 3] -> [2, 3]
    g = Graph()
    a = g.add_node(Node(0, "INPUT", shape=[2, 3]))
    b = g.add_node(Node(1, "INPUT", shape=[1, 3]))
    add = g.add_node(Node(2, "ADD", inputs=[a, b], shape=[2, 3]))
    g.set_inputs([a, b])
    g.set_outputs([add])

    a_val = np.ones((2, 3), dtype=np.float32) * 5.0
    b_val = np.array([[100.0, 200.0, 300.0]], dtype=np.float32)
    res = g.evaluate({a: a_val, b: b_val})[add]
    expected = a_val + b_val
    assert np.allclose(res, expected)

def test_add_unsupported_shapes_rejected():
    import forgetune._forgetune_cpp as cpp

    # 1. Broad over wrong dim: [2, 3] + [2] -> illegal
    g1 = Graph()
    a1 = g1.add_node(Node(0, "INPUT", shape=[2, 3]))
    b1 = g1.add_node(Node(1, "INPUT", shape=[2]))
    g1.add_node(Node(2, "ADD", inputs=[a1, b1], shape=[2, 3]))
    g1.set_inputs([a1, b1])
    g1.set_outputs([2])
    cg1 = cpp.json_to_graph(g1.to_json())
    vres1 = cpp.verify_graph(cg1, False)
    assert not vres1.passed, "C++ verifier must reject [2, 3] + [2]"
    with pytest.raises(Exception):
        cpp.verify_graph(cg1, True)
    with pytest.raises(InvalidShapeError):
        g1.evaluate({a1: np.ones((2, 3), dtype=np.float32), b1: np.ones((2,), dtype=np.float32)})

    # 2. Incompatible dimension: [2, 3] + [4] -> illegal
    g2 = Graph()
    a2 = g2.add_node(Node(0, "INPUT", shape=[2, 3]))
    b2 = g2.add_node(Node(1, "INPUT", shape=[4]))
    g2.add_node(Node(2, "ADD", inputs=[a2, b2], shape=[2, 3]))
    g2.set_inputs([a2, b2])
    g2.set_outputs([2])
    cg2 = cpp.json_to_graph(g2.to_json())
    vres2 = cpp.verify_graph(cg2, False)
    assert not vres2.passed, "C++ verifier must reject [2, 3] + [4]"
    with pytest.raises(Exception):
        cpp.verify_graph(cg2, True)
    with pytest.raises(InvalidShapeError):
        g2.evaluate({a2: np.ones((2, 3), dtype=np.float32), b2: np.ones((4,), dtype=np.float32)})

    # 3. Transposed dimensions: [2, 3] + [3, 2] -> illegal
    g3 = Graph()
    a3 = g3.add_node(Node(0, "INPUT", shape=[2, 3]))
    b3 = g3.add_node(Node(1, "INPUT", shape=[3, 2]))
    g3.add_node(Node(2, "ADD", inputs=[a3, b3], shape=[2, 3]))
    g3.set_inputs([a3, b3])
    g3.set_outputs([2])
    cg3 = cpp.json_to_graph(g3.to_json())
    vres3 = cpp.verify_graph(cg3, False)
    assert not vres3.passed, "C++ verifier must reject [2, 3] + [3, 2]"
    with pytest.raises(Exception):
        cpp.verify_graph(cg3, True)
    with pytest.raises(InvalidShapeError):
        g3.evaluate({a3: np.ones((2, 3), dtype=np.float32), b3: np.ones((3, 2), dtype=np.float32)})

def test_relu_semantics():
    g = Graph()
    x = g.add_node(Node(0, "INPUT", shape=[1, 5]))
    r = g.add_node(Node(1, "RELU", inputs=[x], shape=[1, 5]))
    g.set_inputs([x])
    g.set_outputs([r])

    inp = np.array([[-3.0, -0.0, 0.0, 2.5, 100.0]], dtype=np.float32)
    res = g.evaluate({x: inp})[r]
    expected = np.array([[0.0, 0.0, 0.0, 2.5, 100.0]], dtype=np.float32)
    assert np.allclose(res, expected)

def test_const_node_storage():
    g = Graph()
    data = [1.5, 2.5, 3.5, 4.5]
    c = g.add_node(Node(0, "CONST", shape=[2, 2], const_data=data))
    r = g.add_node(Node(1, "RELU", inputs=[c], shape=[2, 2]))
    g.set_inputs([])
    g.set_outputs([r])

    res = g.evaluate({})[r]
    assert np.allclose(res, np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float32))
