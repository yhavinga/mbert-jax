"""
Layer-wise stability analysis module.

Analyzes parameter evolution and stability across checkpoints.
"""

import numpy as np
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass
from scipy import stats


@dataclass
class LayerStats:
    """Statistics for a single layer."""
    name: str
    shape: Tuple[int, ...]
    num_params: int
    l1_norm: float
    l2_norm: float
    linf_norm: float
    mean: float
    std: float
    skewness: float
    kurtosis: float
    sparsity: float  # Fraction of near-zero values


def flatten_params(params: Dict, prefix: str = "") -> Dict[str, np.ndarray]:
    """Flatten nested parameter dictionary."""
    flat = {}
    for key, value in params.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(flatten_params(value, full_key))
        else:
            flat[full_key] = value
    return flat


def compute_layer_stats(param: np.ndarray, name: str) -> LayerStats:
    """Compute comprehensive statistics for a layer."""
    flat_param = param.reshape(-1)

    # Compute norms
    l1_norm = np.linalg.norm(flat_param, 1)
    l2_norm = np.linalg.norm(flat_param, 2)
    linf_norm = np.linalg.norm(flat_param, np.inf)

    # Statistical moments
    mean = np.mean(flat_param)
    std = np.std(flat_param)
    skewness = stats.skew(flat_param)
    kurtosis_val = stats.kurtosis(flat_param)

    # Sparsity (fraction of values < 1e-6 in magnitude)
    sparsity = np.mean(np.abs(flat_param) < 1e-6)

    return LayerStats(
        name=name,
        shape=param.shape,
        num_params=flat_param.size,
        l1_norm=l1_norm,
        l2_norm=l2_norm,
        linf_norm=linf_norm,
        mean=mean,
        std=std,
        skewness=skewness,
        kurtosis=kurtosis_val,
        sparsity=sparsity
    )


def analyze_stability(params1: Dict, params2: Dict) -> Dict[str, Any]:
    """
    Analyze stability between two checkpoints.

    Returns detailed layer-wise comparison including:
    - Parameter norm changes
    - Update ratios
    - Cosine similarity
    - Statistical changes
    """
    flat1 = flatten_params(params1)
    flat2 = flatten_params(params2)

    results = {}

    for name in flat1.keys():
        if name not in flat2:
            continue

        p1, p2 = flat1[name], flat2[name]

        # Compute statistics for each checkpoint
        stats1 = compute_layer_stats(p1, name)
        stats2 = compute_layer_stats(p2, name)

        # Compute changes
        param_diff = p2 - p1

        # Relative changes
        l2_change = np.linalg.norm(param_diff.reshape(-1), 2)
        relative_l2_change = l2_change / (stats1.l2_norm + 1e-10)

        # Update ratio (training health indicator)
        update_ratio = l2_change / (stats1.l2_norm + 1e-10)

        # Maximum absolute change
        max_change = np.max(np.abs(param_diff))

        # Cosine similarity
        cosine_sim = np.dot(p1.reshape(-1), p2.reshape(-1)) / (
            stats1.l2_norm * stats2.l2_norm + 1e-10
        )

        # Angular distance in degrees
        angular_distance = np.arccos(np.clip(cosine_sim, -1, 1)) * 180 / np.pi

        results[name] = {
            'stats1': stats1,
            'stats2': stats2,
            'l2_change': l2_change,
            'relative_l2_change': relative_l2_change,
            'update_ratio': update_ratio,
            'max_change': max_change,
            'cosine_similarity': cosine_sim,
            'angular_distance': angular_distance,
            'l2_norm_ratio': stats2.l2_norm / (stats1.l2_norm + 1e-10),
            'std_ratio': stats2.std / (stats1.std + 1e-10),
            'sparsity_change': stats2.sparsity - stats1.sparsity
        }

    return results


def summarize_stability(results: Dict[str, Any]) -> Dict[str, Any]:
    """Create summary statistics from stability analysis."""
    if not results:
        return {}

    # Separate by layer type
    attention_layers = [k for k in results.keys() if 'attention' in k]
    mlp_layers = [k for k in results.keys() if 'mlp' in k or 'intermediate' in k]
    embedding_layers = [k for k in results.keys() if 'embedding' in k]
    norm_layers = [k for k in results.keys() if 'norm' in k or 'bias' in k]

    def get_stats(layers):
        if not layers:
            return {}
        update_ratios = [results[l]['update_ratio'] for l in layers]
        l2_ratios = [results[l]['l2_norm_ratio'] for l in layers]
        angular_distances = [results[l]['angular_distance'] for l in layers]

        return {
            'mean_update_ratio': np.mean(update_ratios),
            'max_update_ratio': np.max(update_ratios),
            'mean_l2_ratio': np.mean(l2_ratios),
            'max_l2_ratio': np.max(l2_ratios),
            'mean_angular_distance': np.mean(angular_distances),
            'num_layers': len(layers)
        }

    # Overall statistics
    all_update_ratios = [r['update_ratio'] for r in results.values()]
    all_l2_ratios = [r['l2_norm_ratio'] for r in results.values()]
    all_angular_distances = [r['angular_distance'] for r in results.values()]

    return {
        'overall': {
            'mean_update_ratio': np.mean(all_update_ratios),
            'max_update_ratio': np.max(all_update_ratios),
            'mean_l2_ratio': np.mean(all_l2_ratios),
            'max_l2_ratio': np.max(all_l2_ratios),
            'mean_angular_distance': np.mean(all_angular_distances),
            'total_layers': len(results)
        },
        'attention': get_stats(attention_layers),
        'mlp': get_stats(mlp_layers),
        'embedding': get_stats(embedding_layers),
        'normalization': get_stats(norm_layers),
        'exploded_layers': [
            name for name, r in results.items()
            if r['l2_norm_ratio'] > 1.5 or r['update_ratio'] > 0.5
        ]
    }