import torch
import torch.nn as nn
from typing import List, Dict


class Solution:

    def compute_activation_stats(self, model: nn.Module, x: torch.Tensor) -> List[Dict[str, float]]:
        # Forward pass through model layer by layer
        # After each nn.Linear, record: mean, std, dead_fraction
        # Run with torch.no_grad(). Round to 4 decimals.
        records = []
        with torch.no_grad():
            for m in model.children():
                x = m(x)
                if isinstance(m, nn.Linear):
                    record = {}
                    record["mean"] = round(torch.mean(x).item(), 4)
                    record["std"] = round(torch.std(x).item(), 4)
                    record["dead_fraction"] = round(
                        torch.mean(torch.all(x <= 0, dim=0).float()).item(),
                        4
                    )
                    records.append(record)
        return records

    def compute_gradient_stats(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> List[Dict[str, float]]:
        # Forward + backward pass with nn.MSELoss
        # For each nn.Linear layer's weight gradient, record: mean, std, norm
        # Call model.zero_grad() first. Round to 4 decimals.
        model.zero_grad()
        output = model(x)
        loss = nn.MSELoss()(output, y)
        loss.backward()

        records = []
        for m in model.children():
            if isinstance(m, nn.Linear):
                grad = m.weight.grad
                record = {}
                record["mean"] = round(grad.mean().item(), 4)
                record["std"] = round(grad.std().item(), 4)
                record["norm"] = round(torch.norm(grad).item(), 4)
                records.append(record)
        return records

    def diagnose(self, activation_stats: List[Dict[str, float]], gradient_stats: List[Dict[str, float]]) -> str:
        # Classify network health based on the stats
        # Return: 'dead_neurons', 'exploding_gradients', 'vanishing_gradients', or 'healthy'
        # Check in priority order (see problem description for thresholds)
        for d in activation_stats:
            if d['dead_fraction'] > 0.5:
                return 'dead_neurons'
        for d in gradient_stats:
            if d['norm'] > 1000:
                return 'exploding_gradients'
            elif d['norm'] < 1e-5:
                return 'vanishing_gradients'
        for d in activation_stats:
            if d['std'] < 0.1:
                return 'vanishing_gradients'
            elif d['std'] > 10.0:
                return 'exploding_gradients'
        return 'healthy'
