import torch
import torch.nn as nn
from typing import List


class Solution:

    def detect_dead_neurons(self, model: nn.Module, x: torch.Tensor) -> List[float]:
        # Forward pass through the model.
        # After each ReLU layer, compute the fraction of neurons that are dead.
        # A neuron is dead if it outputs 0 for ALL samples in the batch.
        # Return a list of dead fractions (one per ReLU layer), rounded to 4 decimals.
        dead_fractions = []
        with torch.no_grad():
            for m in model.children():
                x = m(x)
                if isinstance(m, nn.ReLU):
                    dead_fractions.append(
                        torch.mean(torch.all(x == 0, dim=0).float()).item()
                    )
        return dead_fractions

    def suggest_fix(self, dead_fractions: List[float]) -> str:
        # Given dead fractions per ReLU layer, suggest a fix.
        # Check in this order:
        # 1. 'use_leaky_relu' if any layer has dead fraction > 0.5
        # 2. 'reinitialize' if the first layer has dead fraction > 0.3
        # 3. 'reduce_learning_rate' if dead fraction strictly increases
        #    with depth AND the last layer's fraction > 0.1
        # 4. 'healthy' if max dead fraction < 0.1
        # 5. 'healthy' otherwise
        if any(e > 0.5 for e in dead_fractions):
            return "use_leaky_relu"
        elif dead_fractions[0] > 0.3:
            return "reinitialize"
        elif (
            all(
                dead_fractions[i] < dead_fractions[i + 1]
                for i in range(len(dead_fractions) - 1)
            ) and dead_fractions[-1] > 0.1
        ):
            return "reduce_learning_rate"
        elif max(dead_fractions) < 0.1:
            return "healthy"
        return "healthy"
