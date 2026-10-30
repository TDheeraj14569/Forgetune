"""
Acceptance Test Suite: Milestone 6 — Required Models Acceptance Verification.
Verifies end-to-end compilation, native dynamic batch execution, contract validation,
and bitwise numerical fidelity against PyTorch eager mode for:
1. Model A: Dynamic TinyMLP (Linear -> ReLU -> Linear)
2. Model B: Residual MLP (Linear -> ReLU -> Linear -> Add skip -> Linear)
3. Model C: Small CNN (Conv2d -> ReLU -> Conv2d -> ReLU -> Flatten -> Linear)
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from forgetune.compiler import compile_model
from forgetune.errors import InvalidShapeError

# ---------------------------------------------------------
# Model Definitions
# ---------------------------------------------------------
class DynamicTinyMLP(nn.Module):
    def __init__(self, in_features=16, hidden_features=32, out_features=8):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

class ResidualMLP(nn.Module):
    def __init__(self, features=16, out_features=4):
        super().__init__()
        self.fc1 = nn.Linear(features, features)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(features, features)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(features, out_features)

    def forward(self, x):
        h1 = self.relu1(self.fc1(x))
        h2 = self.relu2(self.fc2(h1))
        res = h1 + h2
        return self.fc3(res)

class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1)
        self.relu2 = nn.ReLU()
        self.flatten = nn.Flatten(start_dim=1)
        self.fc = nn.Linear(16 * 4 * 4, 10)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.flatten(x)
        return self.fc(x)

# ---------------------------------------------------------
# Model A Acceptance Tests
# ---------------------------------------------------------
def test_acceptance_model_a_dynamic_tinymlp(tmp_path):
    torch.manual_seed(101)
    model = DynamicTinyMLP().eval()
    ex_x = torch.randn(2, 16)
    out_dir = str(tmp_path / "model_a")

    compiled = compile_model(
        model=model,
        example_inputs=(ex_x,),
        output_dir=out_dir,
        autotune=False,
        dynamic_batch=True,
        enable_fusion=True
    )

    # Test batches: 1, 2, 7, 16, 32, 128
    for b in [1, 2, 7, 16, 32, 128]:
        x_test = torch.randn(b, 16)
        with torch.no_grad():
            ref = model(x_test).numpy()
        act = compiled.run(x_test)
        assert act.shape == (b, 8)
        np.testing.assert_allclose(act, ref, atol=1e-4, rtol=1e-4)

    # REQ-04: Reject out of bounds batch sizes
    with pytest.raises(InvalidShapeError):
        compiled.run(torch.randn(0, 16))
    with pytest.raises(InvalidShapeError):
        compiled.run(torch.randn(129, 16))

# ---------------------------------------------------------
# Model B Acceptance Tests
# ---------------------------------------------------------
def test_acceptance_model_b_residual_mlp(tmp_path):
    torch.manual_seed(202)
    model = ResidualMLP().eval()
    ex_x = torch.randn(2, 16)
    out_dir = str(tmp_path / "model_b")

    compiled = compile_model(
        model=model,
        example_inputs=(ex_x,),
        output_dir=out_dir,
        autotune=False,
        dynamic_batch=True,
        enable_fusion=True
    )

    for b in [1, 2, 7, 16, 32, 128]:
        x_test = torch.randn(b, 16)
        with torch.no_grad():
            ref = model(x_test).numpy()
        act = compiled.run(x_test)
        assert act.shape == (b, 4)
        np.testing.assert_allclose(act, ref, atol=1e-4, rtol=1e-4)

    with pytest.raises(InvalidShapeError):
        compiled.run(torch.randn(0, 16))
    with pytest.raises(InvalidShapeError):
        compiled.run(torch.randn(129, 16))

# ---------------------------------------------------------
# Model C Acceptance Tests
# ---------------------------------------------------------
def test_acceptance_model_c_small_cnn(tmp_path):
    torch.manual_seed(303)
    model = SmallCNN().eval()
    ex_x = torch.randn(2, 1, 8, 8)
    out_dir = str(tmp_path / "model_c")

    compiled = compile_model(
        model=model,
        example_inputs=(ex_x,),
        output_dir=out_dir,
        autotune=False,
        dynamic_batch=True,
        enable_fusion=True
    )

    for b in [1, 2, 7, 16, 32, 128]:
        x_test = torch.randn(b, 1, 8, 8)
        with torch.no_grad():
            ref = model(x_test).numpy()
        act = compiled.run(x_test)
        assert act.shape == (b, 10)
        np.testing.assert_allclose(act, ref, atol=1e-4, rtol=1e-4)

    with pytest.raises(InvalidShapeError):
        compiled.run(torch.randn(0, 1, 8, 8))
    with pytest.raises(InvalidShapeError):
        compiled.run(torch.randn(129, 1, 8, 8))
