"""
ForgeTune Example: Compile and Run API Demonstration.
"""

import torch
import torch.nn as nn
import forgetune

class SingleLayerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 4)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.fc(x))

def main():
    torch.manual_seed(123)
    model = SingleLayerModel().eval()
    inp = torch.randn(2, 4)

    print("Compiling SingleLayerModel...")
    compiled = forgetune.compile_model(
        model=model,
        example_inputs=(inp,),
        output_dir="build/artifacts/single_layer",
        autotune=True
    )

    out = compiled.run(inp)
    print("Inference output:\n", out)
    print("Manifest:\n", compiled.manifest())

if __name__ == "__main__":
    main()
