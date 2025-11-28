"""
Deep checkpoint comparison utilities.

Provides comprehensive comparison between checkpoints including
statistical analysis, distribution comparisons, and summary reports.
"""

import numpy as np
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass
from scipy import stats as scipy_stats


def compute_comprehensive_stats(params: Dict[str, np.ndarray]) -> Dict[str, Dict[str, Any]]:
    """
    Compute comprehensive statistics for all parameters.

    Returns detailed statistics for each layer including:
    - Basic stats (mean, std, min, max)
    - Distribution metrics (skewness, kurtosis)
    - Norms (L1, L2, Linf)
    - Percentiles
    """
    layer_stats = {}

    for name, param in params.items():
        flat_param = param.reshape(-1)

        # Basic statistics
        basic_stats = {
            'shape': param.shape,
            'num_params': flat_param.size,
            'mean': float(np.mean(flat_param)),
            'std': float(np.std(flat_param)),
            'min': float(np.min(flat_param)),
            'max': float(np.max(flat_param)),
            'abs_mean': float(np.mean(np.abs(flat_param))),
        }

        # Norms
        norms = {
            'l1_norm': float(np.linalg.norm(flat_param, 1)),
            'l2_norm': float(np.linalg.norm(flat_param, 2)),
            'linf_norm': float(np.linalg.norm(flat_param, np.inf)),
            'frobenius_norm': float(np.linalg.norm(param, 'fro')) if len(param.shape) > 1 else basic_stats['l2_norm'],
        }

        # Distribution metrics
        distribution = {
            'skewness': float(scipy_stats.skew(flat_param)),
            'kurtosis': float(scipy_stats.kurtosis(flat_param)),
            'median': float(np.median(flat_param)),
            'q25': float(np.percentile(flat_param, 25)),
            'q75': float(np.percentile(flat_param, 75)),
            'q95': float(np.percentile(flat_param, 95)),
            'q05': float(np.percentile(flat_param, 5)),
        }

        # Sparsity metrics
        sparsity = {
            'zero_fraction': float(np.mean(flat_param == 0)),
            'near_zero_fraction': float(np.mean(np.abs(flat_param) < 1e-6)),
            'small_fraction': float(np.mean(np.abs(flat_param) < 0.01)),
        }

        layer_stats[name] = {
            **basic_stats,
            **norms,
            **distribution,
            **sparsity
        }

    return layer_stats


def compare_checkpoints(
    stats1: Dict[str, Dict[str, Any]],
    stats2: Dict[str, Dict[str, Any]]
) -> Dict[str, Dict[str, Any]]:
    """
    Compare two checkpoint statistics.

    Returns ratios and differences for all metrics.
    """
    comparison = {}

    for name in stats1.keys():
        if name not in stats2:
            continue

        s1, s2 = stats1[name], stats2[name]

        # Compute ratios for positive values
        eps = 1e-10
        ratios = {}
        for key in ['l1_norm', 'l2_norm', 'linf_norm', 'std', 'abs_mean']:
            if key in s1 and key in s2:
                ratios[f'{key}_ratio'] = s2[key] / (s1[key] + eps)

        # Compute differences
        diffs = {}
        for key in ['mean', 'std', 'skewness', 'kurtosis', 'zero_fraction']:
            if key in s1 and key in s2:
                diffs[f'{key}_diff'] = s2[key] - s1[key]

        # Special comparisons
        special = {
            'magnitude_change': (s2['l2_norm'] - s1['l2_norm']) / (s1['l2_norm'] + eps),
            'distribution_shift': abs(s2['mean'] - s1['mean']) / (s1['std'] + eps),
            'spread_change': s2['std'] / (s1['std'] + eps) - 1,
        }

        comparison[name] = {**ratios, **diffs, **special}

    return comparison


def identify_significant_changes(
    comparison: Dict[str, Dict[str, Any]],
    threshold: float = 0.1
) -> Dict[str, List[str]]:
    """
    Identify layers with significant changes.

    Args:
        comparison: Output from compare_checkpoints
        threshold: Relative change threshold (default 10%)

    Returns:
        Dictionary categorizing layers by change type
    """
    categories = {
        'exploded': [],  # Large magnitude increase
        'collapsed': [],  # Large magnitude decrease
        'distribution_shifted': [],  # Mean shifted significantly
        'variance_changed': [],  # Std changed significantly
        'stable': [],  # Little change
    }

    for name, metrics in comparison.items():
        # Check for explosion (>50% increase in L2 norm)
        if metrics.get('l2_norm_ratio', 1) > 1.5:
            categories['exploded'].append(name)

        # Check for collapse (>50% decrease in L2 norm)
        elif metrics.get('l2_norm_ratio', 1) < 0.5:
            categories['collapsed'].append(name)

        # Check for distribution shift
        elif abs(metrics.get('distribution_shift', 0)) > 3:
            categories['distribution_shifted'].append(name)

        # Check for variance change
        elif abs(metrics.get('spread_change', 0)) > threshold:
            categories['variance_changed'].append(name)

        # Otherwise stable
        else:
            categories['stable'].append(name)

    return categories


def generate_comparison_summary(
    stats1: Dict[str, Dict[str, Any]],
    stats2: Dict[str, Dict[str, Any]],
    name1: str = "Checkpoint 1",
    name2: str = "Checkpoint 2"
) -> str:
    """
    Generate human-readable comparison summary.

    Returns formatted text summary of key findings.
    """
    comparison = compare_checkpoints(stats1, stats2)
    categories = identify_significant_changes(comparison)

    # Compute aggregate statistics
    all_l2_ratios = [c.get('l2_norm_ratio', 1) for c in comparison.values()]
    mean_l2_ratio = np.mean(all_l2_ratios)
    max_l2_ratio = np.max(all_l2_ratios)
    min_l2_ratio = np.min(all_l2_ratios)

    summary = []
    summary.append("=" * 70)
    summary.append(f"CHECKPOINT COMPARISON: {name1} vs {name2}")
    summary.append("=" * 70)
    summary.append("")

    # Overall statistics
    summary.append("OVERALL STATISTICS:")
    summary.append(f"  Total layers compared: {len(comparison)}")
    summary.append(f"  Mean L2 norm ratio: {mean_l2_ratio:.3f}")
    summary.append(f"  Max L2 norm ratio: {max_l2_ratio:.3f} (layer with most growth)")
    summary.append(f"  Min L2 norm ratio: {min_l2_ratio:.3f} (layer with most shrinkage)")
    summary.append("")

    # Layer categories
    summary.append("LAYER ANALYSIS:")
    summary.append(f"  Stable layers: {len(categories['stable'])}")
    summary.append(f"  Exploded layers: {len(categories['exploded'])}")
    if categories['exploded']:
        summary.append(f"    Examples: {', '.join(categories['exploded'][:3])}")
    summary.append(f"  Collapsed layers: {len(categories['collapsed'])}")
    summary.append(f"  Distribution shifted: {len(categories['distribution_shifted'])}")
    summary.append(f"  Variance changed: {len(categories['variance_changed'])}")
    summary.append("")

    # Key findings
    summary.append("KEY FINDINGS:")
    if mean_l2_ratio > 1.2:
        summary.append(f"  ⚠ Parameters grew by {(mean_l2_ratio-1)*100:.1f}% on average")
    elif mean_l2_ratio < 0.8:
        summary.append(f"  ⚠ Parameters shrank by {(1-mean_l2_ratio)*100:.1f}% on average")
    else:
        summary.append(f"  ✓ Parameter magnitudes remained relatively stable")

    if categories['exploded']:
        summary.append(f"  ⚠ {len(categories['exploded'])} layers show signs of explosion")

    if categories['collapsed']:
        summary.append(f"  ⚠ {len(categories['collapsed'])} layers show signs of collapse")

    # Find most changed layer
    max_change_layer = max(comparison.items(), key=lambda x: abs(x[1].get('l2_norm_ratio', 1) - 1))
    summary.append(f"  Most changed layer: {max_change_layer[0][:50]}")
    summary.append(f"    L2 norm ratio: {max_change_layer[1].get('l2_norm_ratio', 1):.3f}")

    return "\n".join(summary)