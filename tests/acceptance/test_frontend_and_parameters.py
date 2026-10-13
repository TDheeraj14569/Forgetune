"""
Acceptance Test Suite: Frontend & Parameter Integrity Verification
Tests ahead-of-time graph extraction via torch.export, weight layout transposition,
parameter lifting fidelity, rejection of unsupported ATen operators, and
bit-exact IEEE 754 float32 JSON serialization.
"""

import json
import pytest
import torch
import torch.nn as nn
import numpy as np

from forgetune.export_frontend import export_to_forgetune_ir
from forgetune.graph_adapter import Graph, Node
from forgetune.errors import UnsupportedOpError, InvalidShapeError
import forgetune._forgetune_cpp as cpp

class MultiLayerTestNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(6, 10, bias=True)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(10, 4, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))

class UnsupportedConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

class UnsupportedSoftmaxNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 4)
        self.sm = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sm(self.fc(x))

class UnsupportedLayerNormNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln = nn.LayerNorm(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ln(x)

def test_acceptance_parameter_lifting_and_weight_transposition():
    """
    Validates that:
    1. Only the user input tensor is registered as an INPUT node in the IR.
    2. Model weights and biases are lifted into CONST nodes.
    3. Linear layer weights are transposed from PyTorch row-major [N, K] to ForgeTune [K, N].
    """
    torch.manual_seed(42)
    model = MultiLayerTestNet()
    model.eval()
    x = torch.randn(2, 6, dtype=torch.float32)

    ir_graph = export_to_forgetune_ir(model, (x,))

    # Check inputs: exactly 1 INPUT node with shape [2, 6]
    assert len(ir_graph.inputs) == 1
    input_node = ir_graph.nodes[ir_graph.inputs[0]]
    assert input_node.op == "INPUT"
    assert input_node.shape == [2, 6]

    # Verify weight transposition for fc1:
    # PyTorch fc1.weight has shape [10, 6]
    # In ForgeTune IR, transposed weight must have shape [6, 10]
    fc1_weight_torch = model.fc1.weight.detach().cpu().numpy() # [10, 6]
    expected_fc1_w_trans = fc1_weight_torch.T # [6, 10]

    transposed_const_nodes = [
        n for n in ir_graph.nodes.values()
        if n.op == "CONST" and n.shape == [6, 10]
    ]
    assert len(transposed_const_nodes) == 1, "Transposed weight [6, 10] not found in IR!"
    actual_transposed_data = np.array(transposed_const_nodes[0].const_data, dtype=np.float32).reshape(6, 10)
    np.testing.assert_allclose(actual_transposed_data, expected_fc1_w_trans, atol=1e-6)

    # Verify bias for fc1: shape [10]
    fc1_bias_torch = model.fc1.bias.detach().cpu().numpy()
    bias_nodes = [
        n for n in ir_graph.nodes.values()
        if n.op == "CONST" and n.shape == [10]
    ]
    assert len(bias_nodes) == 1
    np.testing.assert_allclose(bias_nodes[0].const_data, fc1_bias_torch, atol=1e-6)

def test_acceptance_unsupported_ops_rejected_with_informative_error():
    """
    Verifies that unsupported operations are rejected with actionable error messages.
    """
    # 1. Conv2d rejection
    conv_model = UnsupportedConvNet()
    x_4d = torch.randn(1, 1, 8, 8, dtype=torch.float32)
    with pytest.raises(Exception) as exc_info:
        export_to_forgetune_ir(conv_model, (x_4d,))
    err_str = str(exc_info.value).lower()
    assert "conv" in err_str or "unsupported" in err_str or "rank-2" in err_str

    # 2. Softmax rejection
    sm_model = UnsupportedSoftmaxNet()
    x_2d = torch.randn(2, 4, dtype=torch.float32)
    with pytest.raises(Exception) as exc_info:
        export_to_forgetune_ir(sm_model, (x_2d,))
    err_str = str(exc_info.value).lower()
    assert "softmax" in err_str or "unsupported" in err_str

    # 3. LayerNorm rejection
    ln_model = UnsupportedLayerNormNet()
    with pytest.raises(Exception) as exc_info:
        export_to_forgetune_ir(ln_model, (x_2d,))
    err_str = str(exc_info.value).lower()
    assert "layernorm" in err_str or "unsupported" in err_str or "native_layer_norm" in err_str

def test_acceptance_reject_rank_greater_than_2_inputs():
    """Verifies that rank-3 or rank-4 user inputs are cleanly rejected."""
    model = MultiLayerTestNet()
    x_3d = torch.randn(2, 3, 6, dtype=torch.float32)
    with pytest.raises(InvalidShapeError) as exc_info:
        export_to_forgetune_ir(model, (x_3d,))
    assert "rank-2" in str(exc_info.value)

def test_acceptance_bit_exact_float32_serialization():
    """
    Tests bit-exact JSON round-trip serialization of special float32 values:
    - Signed zero (-0.0)
    - Subnormal values
    - Extreme float32 magnitudes
    """
    special_floats = [
        0.0,
        -0.0,
        1.0,
        -1.0,
        1.401298464324817e-45,  # min positive subnormal
        1.1754943508222875e-38, # min positive normal
        3.4028234663852886e+38, # max float32
        -3.4028234663852886e+38,
        3.1415927410125732,    # pi float32
        -2.7182817459106445     # e float32
    ]

    g = Graph()
    in_id = g.add_node(Node(0, "INPUT", "X", shape=[1, len(special_floats)], dtype="f32"))
    c_id = g.add_node(Node(1, "CONST", "C_spec", shape=[1, len(special_floats)], dtype="f32", const_data=special_floats))
    add_id = g.add_node(Node(2, "ADD", "add", inputs=[in_id, c_id], shape=[1, len(special_floats)], dtype="f32"))
    out_id = g.add_node(Node(3, "OUTPUT", "out", inputs=[add_id], shape=[1, len(special_floats)], dtype="f32"))
    g.set_inputs([in_id])
    g.set_outputs([out_id])

    json_str = g.to_json()

    # Deserialize back via C++ JSON IO
    cpp_graph = cpp.json_to_graph(json_str)
    c_node = cpp_graph.get_node(c_id)
    deserialized_data = c_node.const_data

    # Check length
    assert len(deserialized_data) == len(special_floats)

    # Check bit-exact representation via float32 array
    np_orig = np.array(special_floats, dtype=np.float32)
    np_deser = np.array(deserialized_data, dtype=np.float32)

    assert (np_orig.view(np.uint32) == np_deser.view(np.uint32)).all(), (
        f"Bit-exact mismatch! Orig: {np_orig.view(np.uint32)} vs Deser: {np_deser.view(np.uint32)}"
    )
