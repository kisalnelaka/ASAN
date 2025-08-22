import torch
from src.models import BaselineMLP, MITInspired, ASAN

def test_models():
    x = torch.rand(10, 2)
    models = [BaselineMLP(), MITInspired(), ASAN()]
    for model in models:
        out = model(x)
        assert out.shape == (10, 2), f'{type(model).__name__} output shape incorrect'
    print('Tests passed.')

if __name__ == "__main__":
    test_models()