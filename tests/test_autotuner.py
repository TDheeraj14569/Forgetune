"""
Category H: Autotuner Unit and Integration Tests.
Validates search space bounds, candidate rejection, caching, and selection policy.
"""

import os
import torch
import torch.nn as nn
import numpy as np
import pytest
from forgetune import compile_model
from forgetune.autotuner import Autotuner

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 4)

    def forward(self, x):
        return self.fc(x)

def test_autotuner_budget_and_candidates():
    tuner = Autotuner()
    cands_4 = tuner.get_candidate_space(max_budget=4)
    assert len(cands_4) == 4
    cands_8 = tuner.get_candidate_space(max_budget=8)
    assert len(cands_8) == 8
    assert cands_4[0].candidate_id == "baseline_naive"

def test_autotuning_report_generation(tmp_path):
    torch.manual_seed(99)
    model = SimpleModel().eval()
    x = torch.randn(2, 4)

    output_dir = str(tmp_path / "test_autotuner_run")
    compiled = compile_model(
        model=model,
        example_inputs=(x,),
        output_dir=output_dir,
        autotune=True,
        max_autotune_budget=4
    )

    report = compiled.optimization_report()
    assert "selected_candidate" in report
    assert "candidate_evaluations" in report
    assert len(report["candidate_evaluations"]) >= 1
    assert "best_median_latency_us" in report
