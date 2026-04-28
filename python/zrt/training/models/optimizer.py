"""Optimizer FLOPs and communication modeling.

Reference: Muon optimizer (Kumar et al. 2025), ZeRO paper.
"""
from __future__ import annotations


def ns_flops(m: int, n: int, K: int) -> int:
    """FLOPs for K-step Newton-Schulz orthogonalization on (m, n) matrix.

    Each NS iteration computes:
      X' = X @ (1.5I - 0.5X.T @ X)
    which decomposes into 2 GEMMs: X @ A and X.T @ X where A is (n, n).

    FLOPs per iteration:
      - X.T @ X:  2 × m × n²
      - X @ A:    2 × m × n²
      - Total:    4 × m × n²  (assuming m ≥ n)

    For general (m, n) where we take the smaller dimension for n:
      FLOPs = K × 4 × max(m, n) × min(m, n)²

    Args:
        m: First dimension of matrix (rows)
        n: Second dimension of matrix (columns)
        K: Number of Newton-Schulz iterations

    Returns:
        Total FLOPs for K iterations
    """
    max_dim = max(m, n)
    min_dim = min(m, n)
    return K * 4 * max_dim * min_dim * min_dim


def adam_step_flops(P: int) -> int:
    """FLOPs for Adam optimizer step on P parameters.

    Adam update per parameter:
      1. m = β₁ × m + (1 - β₁) × g           → 2 FLOPs
      2. v = β₂ × v + (1 - β₂) × g²          → 3 FLOPs
      3. m̂ = m / (1 - β₁ᵗ)                   → 1 FLOP
      4. v̂ = v / (1 - β₂ᵗ)                   → 1 FLOP
      5. update = lr × m̂ / (√v̂ + ε)         → 3 FLOPs
      6. w = w - update                       → 1 FLOP
      Total: ~11 FLOPs per parameter

    Including weight decay and gradient clipping:
      - Weight decay: w -= lr × wd × w       → 2 FLOPs
      - Gradient clipping (optional): norm calc + scale

    Conservative estimate: 16 FLOPs/param including overhead.

    Args:
        P: Number of parameters

    Returns:
        Total FLOPs for Adam optimizer step
    """
    return P * 16


def muon_step_flops(P: int, K: int, hidden: int) -> int:
    """FLOPs for Muon optimizer step on P parameters.

    Muon update per parameter group:
      1. Newton-Schulz orthogonalization (K iterations)
         - Each matrix is (output_dim, input_dim)
         - For typical weight matrix, output_dim × input_dim ≈ hidden × hidden
      2. Momentum update: m = β × m + g        → 2 FLOPs
      3. Final update: w = w - lr × orth(m)   → 2 FLOPs

    Approximation:
      - Assume weight matrices are roughly square or tall
      - NS operates on momentum matrix M of shape (m, n)
      - Total NS FLOPs: K × 4 × max(m, n) × min(m, n)²

    For simplicity, we estimate:
      - Per weight: NS contributes ≈ K × 4 × hidden FLOPs
      - Other ops: 4 FLOPs per param

    Args:
        P: Number of parameters
        K: Number of Newton-Schulz iterations
        hidden: Model hidden dimension (for NS matrix sizing)

    Returns:
        Total FLOPs for Muon optimizer step
    """
    ns_total = ns_flops(hidden, hidden, K)
    num_matrices = P // (hidden * hidden) if hidden > 0 else 1
    ns_flops_total = ns_total * num_matrices
    other_flops = P * 4
    return ns_flops_total + other_flops


def muon_optimizer_step_flops(
    P: int,
    K: int,
    hidden: int,
    muon_fraction: float = 0.85,
) -> int:
    """FLOPs for mixed Muon+Adam optimizer step.

    Muon is applied to a fraction of parameters (e.g., attention, FFN weights),
    while Adam is used for the remainder (embeddings, biases, router).

    Args:
        P: Total number of parameters
        K: Newton-Schulz iterations for Muon
        hidden: Model hidden dimension
        muon_fraction: Fraction of params using Muon (default 0.85)

    Returns:
        Total FLOPs for optimizer step
    """
    P_muon = int(P * muon_fraction)
    P_adam = P - P_muon

    muon_flops = muon_step_flops(P_muon, K, hidden)
    adam_flops = adam_step_flops(P_adam)

    return muon_flops + adam_flops