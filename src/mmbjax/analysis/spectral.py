"""
Spectral analysis module for weight matrices.

Analyzes eigenvalue distributions and conditioning of weight matrices.
"""

import numpy as np
from typing import Dict, Tuple, Optional, Any
from dataclasses import dataclass


@dataclass
class SpectralStats:
    """Spectral statistics for a weight matrix."""
    layer_name: str
    shape: Tuple[int, ...]
    rank: int
    condition_number: float
    spectral_norm: float  # Largest singular value
    stable_rank: float
    top_5_singular_values: np.ndarray
    eigenvalue_sum: float
    eigenvalue_product: float
    frobenius_norm: float


def compute_spectral_stats(matrix: np.ndarray, layer_name: str) -> SpectralStats:
    """
    Compute spectral properties of a matrix.

    For matrices that aren't 2D, reshapes appropriately for analysis.
    """
    original_shape = matrix.shape

    # Reshape if needed for SVD
    if len(matrix.shape) == 1:
        # Treat 1D as column vector
        matrix = matrix.reshape(-1, 1)
    elif len(matrix.shape) > 2:
        # Reshape to 2D, preserving first dimension if possible
        if len(matrix.shape) == 4:
            # Conv weights: (out_channels, in_channels, h, w)
            matrix = matrix.reshape(matrix.shape[0], -1)
        else:
            # General case: flatten to 2D
            matrix = matrix.reshape(matrix.shape[0], -1)

    # Compute SVD
    try:
        singular_values = np.linalg.svd(matrix, compute_uv=False)
    except:
        # If SVD fails, return basic stats
        return SpectralStats(
            layer_name=layer_name,
            shape=original_shape,
            rank=min(matrix.shape),
            condition_number=np.inf,
            spectral_norm=np.linalg.norm(matrix, 2),
            stable_rank=1.0,
            top_5_singular_values=np.zeros(5),
            eigenvalue_sum=0.0,
            eigenvalue_product=0.0,
            frobenius_norm=np.linalg.norm(matrix, 'fro')
        )

    # Compute various spectral metrics
    eps = 1e-10
    spectral_norm = singular_values[0] if len(singular_values) > 0 else 0
    min_sv = singular_values[-1] if len(singular_values) > 0 else eps

    condition_number = spectral_norm / (min_sv + eps)

    # Stable rank: ||A||_F^2 / ||A||_2^2
    frobenius_norm = np.linalg.norm(matrix, 'fro')
    stable_rank = (frobenius_norm ** 2) / (spectral_norm ** 2 + eps)

    # Effective rank (number of significant singular values)
    threshold = singular_values[0] * 1e-6 if len(singular_values) > 0 else 0
    rank = np.sum(singular_values > threshold)

    # Top singular values
    top_5 = singular_values[:5] if len(singular_values) >= 5 else singular_values

    # For square matrices, eigenvalue stats
    if matrix.shape[0] == matrix.shape[1]:
        try:
            eigenvalues = np.linalg.eigvals(matrix)
            eigenvalue_sum = np.sum(np.abs(eigenvalues))
            eigenvalue_product = np.prod(np.abs(eigenvalues))
        except:
            eigenvalue_sum = 0.0
            eigenvalue_product = 0.0
    else:
        eigenvalue_sum = np.sum(singular_values)
        eigenvalue_product = 0.0

    return SpectralStats(
        layer_name=layer_name,
        shape=original_shape,
        rank=rank,
        condition_number=condition_number,
        spectral_norm=spectral_norm,
        stable_rank=stable_rank,
        top_5_singular_values=top_5,
        eigenvalue_sum=eigenvalue_sum,
        eigenvalue_product=eigenvalue_product,
        frobenius_norm=frobenius_norm
    )


def analyze_spectral_properties(params: Dict[str, np.ndarray]) -> Dict[str, SpectralStats]:
    """
    Analyze spectral properties of all weight matrices in checkpoint.

    Args:
        params: Flattened parameter dictionary

    Returns:
        Dictionary mapping layer names to spectral statistics
    """
    results = {}

    for name, param in params.items():
        # Skip biases and 1D parameters
        if 'bias' in name or len(param.shape) == 1:
            continue

        # Skip very small parameters
        if param.size < 10:
            continue

        stats = compute_spectral_stats(param, name)
        results[name] = stats

    return results


def compare_spectral_properties(
    stats1: Dict[str, SpectralStats],
    stats2: Dict[str, SpectralStats]
) -> Dict[str, Dict[str, float]]:
    """
    Compare spectral properties between two checkpoints.

    Returns ratios and differences for key metrics.
    """
    comparison = {}

    for name in stats1.keys():
        if name not in stats2:
            continue

        s1, s2 = stats1[name], stats2[name]

        comparison[name] = {
            'condition_ratio': s2.condition_number / (s1.condition_number + 1e-10),
            'spectral_norm_ratio': s2.spectral_norm / (s1.spectral_norm + 1e-10),
            'stable_rank_diff': s2.stable_rank - s1.stable_rank,
            'rank_diff': s2.rank - s1.rank,
            'condition_change': s2.condition_number - s1.condition_number,
            'spectral_norm_change': s2.spectral_norm - s1.spectral_norm,
        }

    return comparison


def summarize_spectral_health(stats: Dict[str, SpectralStats]) -> Dict[str, Any]:
    """
    Summarize spectral health of a checkpoint.

    Identifies potential issues like ill-conditioning or rank collapse.
    """
    if not stats:
        return {}

    condition_numbers = [s.condition_number for s in stats.values()]
    spectral_norms = [s.spectral_norm for s in stats.values()]
    stable_ranks = [s.stable_rank for s in stats.values()]

    # Identify problematic layers
    ill_conditioned = [
        name for name, s in stats.items()
        if s.condition_number > 1e6
    ]

    low_rank = [
        name for name, s in stats.items()
        if s.stable_rank < min(s.shape) * 0.1  # Less than 10% of full rank
    ]

    exploded = [
        name for name, s in stats.items()
        if s.spectral_norm > 100  # Very large spectral norm
    ]

    return {
        'mean_condition': np.mean(condition_numbers),
        'max_condition': np.max(condition_numbers),
        'median_condition': np.median(condition_numbers),
        'mean_spectral_norm': np.mean(spectral_norms),
        'max_spectral_norm': np.max(spectral_norms),
        'mean_stable_rank': np.mean(stable_ranks),
        'ill_conditioned_layers': ill_conditioned,
        'low_rank_layers': low_rank,
        'exploded_layers': exploded,
        'num_layers_analyzed': len(stats),
        'health_score': compute_health_score(condition_numbers, spectral_norms)
    }


def compute_health_score(condition_numbers: list, spectral_norms: list) -> float:
    """
    Compute overall health score (0-100).

    Based on conditioning and spectral norm distribution.
    """
    # Penalize high condition numbers
    condition_score = 100 * np.exp(-np.mean(np.log10(condition_numbers)) / 5)

    # Penalize very high or very low spectral norms
    spectral_score = 100 * np.exp(-abs(np.log10(np.mean(spectral_norms))) / 2)

    # Combined score
    return (condition_score + spectral_score) / 2