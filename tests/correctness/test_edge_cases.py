"""
Category F: Edge and Randomized Tests.
Tests partial tile boundaries, prime dimension sizes, batch size 1, negative inputs, and zeros.
"""

import os
import subprocess
import torch
import torch.nn as nn
import numpy as np
import pytest
from forgetune import compile_model

class NonTiledModel(nn.Module):
    """Prime dimensions (7, 11, 13) that do not divide standard tile sizes (8, 16, 32)"""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(7, 11)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(11, 13)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

def test_prime_dimensions_and_partial_tiles():
    torch.manual_seed(101)
    model = NonTiledModel().eval()
    # Batch size 3 (prime), input width 7
    x = torch.randn(3, 7)

    compiled = compile_model(
        model=model,
        example_inputs=(x,),
        output_dir="build/artifacts/prime_dims",
        autotune=True
    )

    native_out = compiled.run(x)
    with torch.no_grad():
        expected = model(x).numpy()

    assert native_out.shape == (3, 13)
    assert np.allclose(native_out, expected, atol=1e-4)

def test_batch_size_one():
    torch.manual_seed(202)
    model = NonTiledModel().eval()
    # Batch size 1
    x = torch.randn(1, 7)

    compiled = compile_model(
        model=model,
        example_inputs=(x,),
        output_dir="build/artifacts/batch_one",
        autotune=True
    )

    native_out = compiled.run(x)
    with torch.no_grad():
        expected = model(x).numpy()

    assert native_out.shape == (1, 13)
    assert np.allclose(native_out, expected, atol=1e-4)

def test_all_zeros_and_all_negatives():
    torch.manual_seed(303)
    model = NonTiledModel().eval()
    x_zeros = torch.zeros(3, 7)
    x_negs = -torch.abs(torch.randn(3, 7))

    compiled = compile_model(
        model=model,
        example_inputs=(x_zeros,),
        output_dir="build/artifacts/zeros_and_negs",
        autotune=False
    )

    out_zeros = compiled.run(x_zeros)
    with torch.no_grad():
        expected_zeros = model(x_zeros).numpy()
    assert np.allclose(out_zeros, expected_zeros, atol=1e-4)

    out_negs = compiled.run(x_negs)
    with torch.no_grad():
        expected_negs = model(x_negs).numpy()
    assert np.allclose(out_negs, expected_negs, atol=1e-4)
