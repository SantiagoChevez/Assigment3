import torch.nn as nn
from typing import Sequence


class MLP(nn.Module):
    
    def __init__(self, input_size: int, hidden_sizes: Sequence[int] = (128, 64), dropout: float = 0.2, output_size: int = 7, activation=nn.ReLU):
        super().__init__()

        layers = []
        in_size = input_size
        for h in hidden_sizes:
            layers.append(nn.Linear(in_size, h))
            layers.append(activation())
            if dropout and dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_size = h

        # final linear layer to outputs (logits)
        layers.append(nn.Linear(in_size, output_size))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def build_default_model(input_size: int) -> MLP:
    """Convenience: build default MLP for this assignment.

    Default: two hidden layers [128, 64], ReLU, dropout=0.2, output_size=7
    """
    return MLP(input_size=input_size, hidden_sizes=(128, 64), dropout=0.2, output_size=7)


def build_cbow_model(input_size: int, hidden_sizes: Sequence[int] = (64,), dropout: float = 0.0, output_size: int = 7, activation=nn.ReLU) -> MLP:
    """Build an MLP suited for a CBOW-style input.

    CBOW typically averages context word embeddings before feeding them to a classifier.
    Because the input is already an aggregate (less noisy than raw sparse features), a
    smaller network and no dropout are reasonable defaults. Caller can override sizes.
    """
    return MLP(input_size=input_size, hidden_sizes=hidden_sizes, dropout=dropout, output_size=output_size, activation=activation)




