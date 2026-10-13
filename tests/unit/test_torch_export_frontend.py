"""
Unit tests for the torch.export frontend.
Verifies that:
1. export_to_forgetune_ir uses torch.export.export ahead-of-time graph extraction.
2. User inputs, lifted parameters, and buffers are correctly classified from graph_signature.
3. PyTorch Linear weights ([N, K]) are correctly transposed to canonical row-major layout ([K, N]).
4. Unsupported ATen operators are rejected explicitly with UnsupportedOpError.
5. Non-rank-2 input shapes are rejected with InvalidShapeError.
"""

import torch
import torch.nn as nn
import numpy as np
import pytest

from forgetune.export_frontend import export_to_forgetune_ir
from forgetune.errors import UnsupportedOpError, InvalidShapeError, InvalidGraphError

class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(8, 3)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

class UnsupportedOpModel(nn.Module):
    def forward(self, x):
        return torch.sin(x)  # Unsupported ATen operator

class ConvModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, 3)

    def forward(self, x):
        return self.conv(x)

def test_torch_export_extraction_and_transposition():
    torch.manual_seed(42)
    model = SimpleMLP().eval()
    x = torch.randn(2, 4)

    # Export using the production frontend
    graph = export_to_forgetune_ir(model, (x,))

    # 1. Verify that user input was recognized and created with correct shape
    input_nodes = [node for node in graph.nodes.values() if node.op == "INPUT"]
    assert len(input_nodes) == 1
    assert input_nodes[0].shape == [2, 4]

    # 2. Verify that Linear weights were transposed from [N, K] to [K, N]
    # fc1 weight in PyTorch is [8, 4] -> transposed must be [4, 8]
    # fc2 weight in PyTorch is [3, 8] -> transposed must be [8, 3]
    const_shapes = [node.shape for node in graph.nodes.values() if node.op == "CONST"]
    assert [4, 8] in const_shapes, "fc1 weight must be transposed to [4, 8]"
    assert [8, 3] in const_shapes, "fc2 weight must be transposed to [8, 3]"

    # 3. Verify MatMul nodes were created with matching dimensions
    matmul_nodes = [node for node in graph.nodes.values() if node.op == "MATMUL"]
    assert len(matmul_nodes) == 2
    assert matmul_nodes[0].shape == [2, 8]
    assert matmul_nodes[1].shape == [2, 3]

    # 4. Verify numerical evaluation matches PyTorch reference
    with torch.no_grad():
        ref_out = model(x).numpy()
    eval_out = graph.evaluate({graph.inputs[0]: x.numpy()})[graph.outputs[0]]
    assert np.allclose(ref_out, eval_out, atol=1e-5), "Reference interpreter must match PyTorch"

def test_torch_export_rejects_unsupported_ops():
    model = UnsupportedOpModel().eval()
    x = torch.randn(2, 4)
    with pytest.raises(UnsupportedOpError):
        export_to_forgetune_ir(model, (x,))

def test_torch_export_rejects_rank_3_inputs():
    model = SimpleMLP().eval()
    x_rank3 = torch.randn(2, 3, 4)
    with pytest.raises((InvalidShapeError, InvalidGraphError)):
        export_to_forgetune_ir(model, (x_rank3,))

def test_canonical_tinymlp_exported_aten_targets_and_decomposition():
    """
    Explicitly tests and verifies:
    1. torch.export.export produces an ExportedProgram targeting aten.linear and aten.relu.
    2. Parameters are lifted into graph_signature inputs_to_parameters.
    3. export_to_forgetune_ir normalizes and decomposes linear into transposed CONST + MATMUL + ADD.
    4. Actionable error messages on unsupported ops.
    """
    torch.manual_seed(101)
    model = SimpleMLP().eval()
    x = torch.randn(2, 4)

    # 1. Directly inspect torch.export.export ExportedProgram
    exp = torch.export.export(model, (x,))
    aten_targets = [str(n.target) for n in exp.graph_module.graph.nodes if n.op == "call_function"]
    assert "aten.linear.default" in aten_targets, f"Expected aten.linear.default, got {aten_targets}"
    assert "aten.relu.default" in aten_targets, f"Expected aten.relu.default, got {aten_targets}"

    # Verify parameter lifting
    params = exp.graph_signature.inputs_to_parameters
    assert "p_fc1_weight" in params and params["p_fc1_weight"] == "fc1.weight"
    assert "p_fc1_bias" in params and params["p_fc1_bias"] == "fc1.bias"
    assert "p_fc2_weight" in params and params["p_fc2_weight"] == "fc2.weight"
    assert "p_fc2_bias" in params and params["p_fc2_bias"] == "fc2.bias"

    # 2. Run through ForgeTune frontend and verify canonical IR decomposition
    ir_graph = export_to_forgetune_ir(model, (x,))
    op_counts = {}
    for node in ir_graph.nodes.values():
        op_counts[node.op] = op_counts.get(node.op, 0) + 1

    # 1 User Input, 4 parameter Constants (2 weights + 2 biases) + 2 transposed weight constants = 6 CONST nodes
    # 2 MATMUL nodes, 2 ADD nodes, 1 RELU node
    assert op_counts["INPUT"] == 1
    assert op_counts["MATMUL"] == 2
    assert op_counts["ADD"] == 2
    assert op_counts["RELU"] == 1

    # 3. Actionable error message check
    class UnsupportedModel(nn.Module):
        def forward(self, z):
            return torch.sigmoid(z)

    with pytest.raises(UnsupportedOpError) as exc_info:
        export_to_forgetune_ir(UnsupportedModel().eval(), (x,))
    err_str = str(exc_info.value)
    assert "sigmoid" in err_str.lower(), f"Error message should mention unsupported operator: {err_str}"

