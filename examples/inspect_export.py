"""
ForgeTune Example: Inspect PyTorch Export and ForgeTune IR.
"""

import torch
import torch.nn as nn
from forgetune import export_to_forgetune_ir

class TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(8, 3)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

def main():
    torch.manual_seed(42)
    model = TinyMLP().eval()
    example_input = torch.randn(2, 4)

    print("=================================================================")
    print("1. PyTorch torch.export.export inspection")
    print("=================================================================")
    exported = torch.export.export(model, (example_input,))
    print(exported.graph_module)
    print("\nGraph Signature:")
    print(exported.graph_signature)

    print("\n=================================================================")
    print("2. ForgeTune Normalized IR")
    print("=================================================================")
    ir_graph = export_to_forgetune_ir(model, (example_input,))
    print(ir_graph.to_json(indent=2))

if __name__ == "__main__":
    main()
