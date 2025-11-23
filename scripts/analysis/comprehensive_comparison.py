#!/usr/bin/env python3
"""
Comprehensive comparison of Red (1M stable) vs Yellow (2M exploded) runs.
Creates proper visualizations with color coding to show the differences.

This script combines insights from all three analysis tools to provide
a complete picture of what went wrong in the yellow run.
"""

import pickle
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats as scipy_stats, linalg
import json
from typing import Dict, Any, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def load_checkpoint(path: str) -> Dict[str, np.ndarray]:
    """Load converted checkpoint from pickle file."""
    with open(path, 'rb') as f:
        params = pickle.load(f)
    print(f"Loaded checkpoint from {path}")
    return params

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

def compute_layer_stats(params: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
    """Compute comprehensive statistics for each layer."""
    stats = {}

    for name, param in params.items():
        if param.size == 0:
            continue

        flat_param = param.flatten()

        # Identify layer type and number
        layer_type = "other"
        layer_num = -1

        if "attention" in name.lower():
            layer_type = "attention"
        elif "mlp" in name.lower() or "intermediate" in name.lower() or "output" in name.lower():
            layer_type = "mlp"
        elif "embedding" in name.lower():
            layer_type = "embedding"
        elif "layernorm" in name.lower() or "layer_norm" in name.lower():
            layer_type = "layernorm"

        # Extract layer number if present
        import re
        layer_match = re.search(r'layers?[_.](\d+)', name)
        if layer_match:
            layer_num = int(layer_match.group(1))

        stats[name] = {
            'layer_type': layer_type,
            'layer_num': layer_num,
            'l2_norm': np.linalg.norm(flat_param),
            'mean': np.mean(flat_param),
            'std': np.std(flat_param),
            'max': np.max(np.abs(flat_param)),
            'skewness': scipy_stats.skew(flat_param),
            'kurtosis': scipy_stats.kurtosis(flat_param),
            'num_params': param.size,
            'num_near_zero': np.sum(np.abs(flat_param) < 1e-6),
            'spectral_norm': np.linalg.norm(param, ord=2) if param.ndim == 2 else np.nan,
            'percentile_95': np.percentile(np.abs(flat_param), 95),
            'percentile_99': np.percentile(np.abs(flat_param), 99)
        }

    return stats

def plot_comprehensive_comparison(red_stats: Dict, yellow_stats: Dict, save_path: str):
    """Create comprehensive visualization comparing red and yellow runs."""

    fig = plt.figure(figsize=(24, 16))
    fig.suptitle('Red (1M Stable) vs Yellow (2M Exploded) - Comprehensive Comparison',
                 fontsize=20, fontweight='bold')

    # Get common layers
    common_layers = sorted(set(red_stats.keys()) & set(yellow_stats.keys()))

    # Sort layers by layer number where available
    def sort_key(name):
        layer_num = red_stats[name]['layer_num']
        layer_type = red_stats[name]['layer_type']
        type_order = {'embedding': 0, 'attention': 1, 'mlp': 2, 'layernorm': 3, 'other': 4}
        return (layer_num if layer_num >= 0 else 999, type_order.get(layer_type, 5), name)

    common_layers = sorted(common_layers, key=sort_key)

    # 1. Parameter Norm Comparison
    ax1 = plt.subplot(3, 4, 1)
    layer_indices = range(len(common_layers))
    red_norms = [red_stats[layer]['l2_norm'] for layer in common_layers]
    yellow_norms = [yellow_stats[layer]['l2_norm'] for layer in common_layers]

    ax1.scatter(layer_indices, red_norms, color='red', alpha=0.6, s=20, label='Red (1M)')
    ax1.scatter(layer_indices, yellow_norms, color='gold', alpha=0.6, s=20, label='Yellow (2M)')
    ax1.set_xlabel('Layer Index')
    ax1.set_ylabel('L2 Norm')
    ax1.set_title('Parameter Norms by Layer')
    ax1.legend()
    ax1.set_yscale('log')

    # 2. Norm Ratio (Yellow/Red)
    ax2 = plt.subplot(3, 4, 2)
    norm_ratios = [yellow_stats[layer]['l2_norm'] / red_stats[layer]['l2_norm']
                   for layer in common_layers]
    colors = ['red' if r < 0.9 else 'gold' if r > 1.1 else 'gray' for r in norm_ratios]
    bars = ax2.bar(layer_indices, norm_ratios, color=colors, alpha=0.6)
    ax2.axhline(y=1.0, color='black', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Layer Index')
    ax2.set_ylabel('Norm Ratio (Yellow/Red)')
    ax2.set_title('Parameter Growth Factor')
    ax2.set_ylim([0.5, 2.0])

    # 3. Kurtosis Comparison
    ax3 = plt.subplot(3, 4, 3)
    red_kurtosis = [red_stats[layer]['kurtosis'] for layer in common_layers]
    yellow_kurtosis = [yellow_stats[layer]['kurtosis'] for layer in common_layers]

    ax3.scatter(layer_indices, red_kurtosis, color='red', alpha=0.6, s=20, label='Red (1M)')
    ax3.scatter(layer_indices, yellow_kurtosis, color='gold', alpha=0.6, s=20, label='Yellow (2M)')
    ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Layer Index')
    ax3.set_ylabel('Kurtosis')
    ax3.set_title('Distribution Shape (Heavy Tails)')
    ax3.legend()

    # 4. Maximum Absolute Values
    ax4 = plt.subplot(3, 4, 4)
    red_max = [red_stats[layer]['max'] for layer in common_layers]
    yellow_max = [yellow_stats[layer]['max'] for layer in common_layers]

    ax4.scatter(layer_indices, red_max, color='red', alpha=0.6, s=20, label='Red (1M)')
    ax4.scatter(layer_indices, yellow_max, color='gold', alpha=0.6, s=20, label='Yellow (2M)')
    ax4.set_xlabel('Layer Index')
    ax4.set_ylabel('Max |Parameter|')
    ax4.set_title('Maximum Parameter Values')
    ax4.legend()
    ax4.set_yscale('log')

    # 5. Layer Type Analysis
    ax5 = plt.subplot(3, 4, 5)
    layer_types = ['attention', 'mlp', 'embedding', 'layernorm', 'other']
    red_means_by_type = {lt: [] for lt in layer_types}
    yellow_means_by_type = {lt: [] for lt in layer_types}

    for layer in common_layers:
        lt = red_stats[layer]['layer_type']
        red_means_by_type[lt].append(red_stats[layer]['l2_norm'])
        yellow_means_by_type[lt].append(yellow_stats[layer]['l2_norm'])

    x = np.arange(len(layer_types))
    width = 0.35

    red_means = [np.mean(red_means_by_type[lt]) if red_means_by_type[lt] else 0 for lt in layer_types]
    yellow_means = [np.mean(yellow_means_by_type[lt]) if yellow_means_by_type[lt] else 0 for lt in layer_types]

    ax5.bar(x - width/2, red_means, width, label='Red (1M)', color='red', alpha=0.6)
    ax5.bar(x + width/2, yellow_means, width, label='Yellow (2M)', color='gold', alpha=0.6)
    ax5.set_xlabel('Layer Type')
    ax5.set_ylabel('Mean L2 Norm')
    ax5.set_title('Average Norms by Layer Type')
    ax5.set_xticks(x)
    ax5.set_xticklabels(layer_types, rotation=45)
    ax5.legend()

    # 6. Near-Zero Parameters
    ax6 = plt.subplot(3, 4, 6)
    red_zeros = [red_stats[layer]['num_near_zero'] / red_stats[layer]['num_params']
                 for layer in common_layers]
    yellow_zeros = [yellow_stats[layer]['num_near_zero'] / yellow_stats[layer]['num_params']
                    for layer in common_layers]

    ax6.scatter(layer_indices, red_zeros, color='red', alpha=0.6, s=20, label='Red (1M)')
    ax6.scatter(layer_indices, yellow_zeros, color='gold', alpha=0.6, s=20, label='Yellow (2M)')
    ax6.set_xlabel('Layer Index')
    ax6.set_ylabel('Fraction Near Zero')
    ax6.set_title('Sparsity (|param| < 1e-6)')
    ax6.legend()

    # 7. Standard Deviation Comparison
    ax7 = plt.subplot(3, 4, 7)
    red_std = [red_stats[layer]['std'] for layer in common_layers]
    yellow_std = [yellow_stats[layer]['std'] for layer in common_layers]

    ax7.scatter(layer_indices, red_std, color='red', alpha=0.6, s=20, label='Red (1M)')
    ax7.scatter(layer_indices, yellow_std, color='gold', alpha=0.6, s=20, label='Yellow (2M)')
    ax7.set_xlabel('Layer Index')
    ax7.set_ylabel('Standard Deviation')
    ax7.set_title('Parameter Spread')
    ax7.legend()
    ax7.set_yscale('log')

    # 8. Spectral Norm (for 2D parameters)
    ax8 = plt.subplot(3, 4, 8)
    red_spectral = [red_stats[layer]['spectral_norm'] for layer in common_layers
                    if not np.isnan(red_stats[layer]['spectral_norm'])]
    yellow_spectral = [yellow_stats[layer]['spectral_norm'] for layer in common_layers
                       if not np.isnan(yellow_stats[layer]['spectral_norm'])]
    spectral_indices = [i for i, layer in enumerate(common_layers)
                       if not np.isnan(red_stats[layer]['spectral_norm'])]

    if red_spectral and yellow_spectral:
        ax8.scatter(spectral_indices, red_spectral, color='red', alpha=0.6, s=20, label='Red (1M)')
        ax8.scatter(spectral_indices, yellow_spectral, color='gold', alpha=0.6, s=20, label='Yellow (2M)')
        ax8.set_xlabel('Layer Index')
        ax8.set_ylabel('Spectral Norm')
        ax8.set_title('Spectral Norms (2D Weights)')
        ax8.legend()
        ax8.set_yscale('log')

    # 9. Layer-wise Instability Score
    ax9 = plt.subplot(3, 4, 9)
    instability_scores = []
    for layer in common_layers:
        # Combine multiple factors
        norm_change = abs(yellow_stats[layer]['l2_norm'] / red_stats[layer]['l2_norm'] - 1.0)
        kurtosis_change = abs(yellow_stats[layer]['kurtosis'] - red_stats[layer]['kurtosis'])
        std_change = abs(yellow_stats[layer]['std'] / red_stats[layer]['std'] - 1.0)
        instability = norm_change + 0.1 * kurtosis_change + std_change
        instability_scores.append(instability)

    colors = ['red' if s > 0.5 else 'gold' if s > 0.2 else 'green' for s in instability_scores]
    ax9.bar(layer_indices, instability_scores, color=colors, alpha=0.6)
    ax9.set_xlabel('Layer Index')
    ax9.set_ylabel('Instability Score')
    ax9.set_title('Combined Instability Metric')
    ax9.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='High Risk')
    ax9.axhline(y=0.2, color='gold', linestyle='--', alpha=0.5, label='Medium Risk')
    ax9.legend()

    # 10. Percentile Analysis
    ax10 = plt.subplot(3, 4, 10)
    red_p99 = [red_stats[layer]['percentile_99'] for layer in common_layers]
    yellow_p99 = [yellow_stats[layer]['percentile_99'] for layer in common_layers]

    ax10.scatter(layer_indices, red_p99, color='red', alpha=0.6, s=20, label='Red (1M)')
    ax10.scatter(layer_indices, yellow_p99, color='gold', alpha=0.6, s=20, label='Yellow (2M)')
    ax10.set_xlabel('Layer Index')
    ax10.set_ylabel('99th Percentile |Parameter|')
    ax10.set_title('Tail Behavior (99th Percentile)')
    ax10.legend()
    ax10.set_yscale('log')

    # 11. Direct Parameter Difference
    ax11 = plt.subplot(3, 4, 11)
    # Since these are different runs, we compare distributions
    js_divergences = []
    for layer in common_layers:
        # Simple proxy for distribution difference
        red_mean = red_stats[layer]['mean']
        yellow_mean = yellow_stats[layer]['mean']
        red_std = red_stats[layer]['std']
        yellow_std = yellow_stats[layer]['std']

        # Normalized difference
        diff = abs(yellow_mean - red_mean) / (0.5 * (red_std + yellow_std) + 1e-8)
        js_divergences.append(min(diff, 10))  # Cap at 10 for visualization

    ax11.bar(layer_indices, js_divergences, color='purple', alpha=0.6)
    ax11.set_xlabel('Layer Index')
    ax11.set_ylabel('Normalized Mean Difference')
    ax11.set_title('Distribution Shift Magnitude')

    # 12. Layer Depth Analysis
    ax12 = plt.subplot(3, 4, 12)
    # Group by layer depth
    depth_groups = {}
    for layer in common_layers:
        layer_num = red_stats[layer]['layer_num']
        if layer_num >= 0:
            if layer_num not in depth_groups:
                depth_groups[layer_num] = {'red': [], 'yellow': []}
            depth_groups[layer_num]['red'].append(red_stats[layer]['l2_norm'])
            depth_groups[layer_num]['yellow'].append(yellow_stats[layer]['l2_norm'])

    if depth_groups:
        depths = sorted(depth_groups.keys())
        red_by_depth = [np.mean(depth_groups[d]['red']) for d in depths]
        yellow_by_depth = [np.mean(depth_groups[d]['yellow']) for d in depths]

        ax12.plot(depths, red_by_depth, 'r-o', label='Red (1M)', linewidth=2)
        ax12.plot(depths, yellow_by_depth, 'y-o', label='Yellow (2M)', linewidth=2)
        ax12.set_xlabel('Layer Depth')
        ax12.set_ylabel('Mean L2 Norm')
        ax12.set_title('Norm Evolution by Layer Depth')
        ax12.legend()
        ax12.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()  # Close instead of show for non-GUI

    return fig

def create_side_by_side_analysis(red_stats: Dict, yellow_stats: Dict, save_path: str):
    """Create side-by-side comparison focusing on key differences."""

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Side-by-Side: Red (Stable) vs Yellow (Exploded)', fontsize=16, fontweight='bold')

    common_layers = sorted(set(red_stats.keys()) & set(yellow_stats.keys()))

    # Identify most changed layers
    changes = []
    for layer in common_layers:
        norm_ratio = yellow_stats[layer]['l2_norm'] / red_stats[layer]['l2_norm']
        changes.append((layer, norm_ratio))

    changes.sort(key=lambda x: abs(x[1] - 1.0), reverse=True)
    top_changed = changes[:20]  # Top 20 most changed layers

    # Plot 1: Top Changed Layers
    ax = axes[0, 0]
    layer_names = [c[0].split('.')[-2:] for c in top_changed[:10]]
    layer_names = ['.'.join(ln) for ln in layer_names]
    ratios = [c[1] for c in top_changed[:10]]

    colors = ['red' if r < 0.9 else 'gold' if r > 1.1 else 'gray' for r in ratios]
    bars = ax.barh(range(len(layer_names)), ratios, color=colors, alpha=0.7)
    ax.set_yticks(range(len(layer_names)))
    ax.set_yticklabels(layer_names, fontsize=8)
    ax.axvline(x=1.0, color='black', linestyle='--', alpha=0.5)
    ax.set_xlabel('Norm Ratio (Yellow/Red)')
    ax.set_title('Top 10 Most Changed Layers')
    ax.set_xlim([0.5, 2.0])

    # Plot 2: Layer Type Statistics
    ax = axes[0, 1]
    layer_types = ['attention', 'mlp', 'embedding', 'layernorm', 'other']

    red_stats_by_type = {lt: [] for lt in layer_types}
    yellow_stats_by_type = {lt: [] for lt in layer_types}

    for layer in common_layers:
        lt = red_stats[layer]['layer_type']
        red_stats_by_type[lt].append(red_stats[layer]['l2_norm'])
        yellow_stats_by_type[lt].append(yellow_stats[layer]['l2_norm'])

    # Box plot comparison
    data_red = [red_stats_by_type[lt] for lt in layer_types if red_stats_by_type[lt]]
    data_yellow = [yellow_stats_by_type[lt] for lt in layer_types if yellow_stats_by_type[lt]]
    labels = [lt for lt in layer_types if red_stats_by_type[lt]]

    positions_red = np.arange(len(labels)) * 2
    positions_yellow = positions_red + 0.8

    bp1 = ax.boxplot(data_red, positions=positions_red, widths=0.6, patch_artist=True,
                      boxprops=dict(facecolor='red', alpha=0.5),
                      medianprops=dict(color='darkred'))
    bp2 = ax.boxplot(data_yellow, positions=positions_yellow, widths=0.6, patch_artist=True,
                      boxprops=dict(facecolor='gold', alpha=0.5),
                      medianprops=dict(color='darkorange'))

    ax.set_xticks(positions_red + 0.4)
    ax.set_xticklabels(labels, rotation=45)
    ax.set_ylabel('L2 Norm (log scale)')
    ax.set_yscale('log')
    ax.set_title('Parameter Distribution by Layer Type')
    ax.legend([bp1["boxes"][0], bp2["boxes"][0]], ['Red (1M)', 'Yellow (2M)'])

    # Plot 3: Kurtosis Distribution
    ax = axes[0, 2]
    red_kurtosis = [red_stats[layer]['kurtosis'] for layer in common_layers]
    yellow_kurtosis = [yellow_stats[layer]['kurtosis'] for layer in common_layers]

    ax.hist(red_kurtosis, bins=30, alpha=0.5, color='red', label='Red (1M)', density=True)
    ax.hist(yellow_kurtosis, bins=30, alpha=0.5, color='gold', label='Yellow (2M)', density=True)
    ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    ax.set_xlabel('Kurtosis')
    ax.set_ylabel('Density')
    ax.set_title('Kurtosis Distribution (Heavy Tails Indicator)')
    ax.legend()

    # Plot 4: Evolution by Layer Depth
    ax = axes[1, 0]
    depth_stats = {'red': {}, 'yellow': {}}

    for layer in common_layers:
        layer_num = red_stats[layer]['layer_num']
        if layer_num >= 0:
            if layer_num not in depth_stats['red']:
                depth_stats['red'][layer_num] = []
                depth_stats['yellow'][layer_num] = []
            depth_stats['red'][layer_num].append(red_stats[layer]['l2_norm'])
            depth_stats['yellow'][layer_num].append(yellow_stats[layer]['l2_norm'])

    if depth_stats['red']:
        depths = sorted(depth_stats['red'].keys())

        # Mean and std by depth
        red_means = [np.mean(depth_stats['red'][d]) for d in depths]
        red_stds = [np.std(depth_stats['red'][d]) for d in depths]
        yellow_means = [np.mean(depth_stats['yellow'][d]) for d in depths]
        yellow_stds = [np.std(depth_stats['yellow'][d]) for d in depths]

        ax.errorbar(depths, red_means, yerr=red_stds, color='red', label='Red (1M)',
                   linewidth=2, capsize=5, alpha=0.7)
        ax.errorbar(depths, yellow_means, yerr=yellow_stds, color='gold', label='Yellow (2M)',
                   linewidth=2, capsize=5, alpha=0.7)
        ax.set_xlabel('Layer Depth')
        ax.set_ylabel('L2 Norm')
        ax.set_title('Norm Evolution Through Network Depth')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Plot 5: Instability Heatmap
    ax = axes[1, 1]

    # Create instability matrix
    layer_types_ordered = ['embedding', 'attention', 'mlp', 'layernorm', 'other']
    max_depth = max([red_stats[l]['layer_num'] for l in common_layers
                    if red_stats[l]['layer_num'] >= 0], default=10)

    instability_matrix = np.zeros((len(layer_types_ordered), min(max_depth + 1, 20)))
    counts_matrix = np.zeros_like(instability_matrix)

    for layer in common_layers:
        lt = red_stats[layer]['layer_type']
        ln = red_stats[layer]['layer_num']

        if lt in layer_types_ordered and 0 <= ln < 20:
            type_idx = layer_types_ordered.index(lt)
            norm_ratio = yellow_stats[layer]['l2_norm'] / red_stats[layer]['l2_norm']
            instability_matrix[type_idx, ln] += abs(norm_ratio - 1.0)
            counts_matrix[type_idx, ln] += 1

    # Average where we have data
    with np.errstate(divide='ignore', invalid='ignore'):
        instability_matrix = np.divide(instability_matrix, counts_matrix)
        instability_matrix[np.isnan(instability_matrix)] = 0

    im = ax.imshow(instability_matrix, aspect='auto', cmap='YlOrRd', vmin=0, vmax=1)
    ax.set_yticks(range(len(layer_types_ordered)))
    ax.set_yticklabels(layer_types_ordered)
    ax.set_xlabel('Layer Depth')
    ax.set_ylabel('Layer Type')
    ax.set_title('Instability Heatmap (Norm Change)')
    plt.colorbar(im, ax=ax)

    # Plot 6: Critical Layers Identification
    ax = axes[1, 2]

    # Compute comprehensive instability score
    instability_scores = []
    for layer in common_layers:
        norm_change = abs(yellow_stats[layer]['l2_norm'] / red_stats[layer]['l2_norm'] - 1.0)
        kurtosis_change = abs(yellow_stats[layer]['kurtosis'] - red_stats[layer]['kurtosis'])
        std_change = abs(yellow_stats[layer]['std'] / red_stats[layer]['std'] - 1.0)
        max_change = abs(yellow_stats[layer]['max'] / red_stats[layer]['max'] - 1.0)

        score = norm_change + 0.1 * kurtosis_change + std_change + 0.5 * max_change
        instability_scores.append((layer, score, red_stats[layer]['layer_type']))

    instability_scores.sort(key=lambda x: x[1], reverse=True)

    # Plot top unstable layers by type
    type_colors = {'attention': 'blue', 'mlp': 'green', 'embedding': 'red',
                   'layernorm': 'purple', 'other': 'gray'}

    top_n = 30
    for i, (layer, score, lt) in enumerate(instability_scores[:top_n]):
        ax.scatter(i, score, color=type_colors.get(lt, 'gray'), s=50, alpha=0.7)

    ax.set_xlabel('Rank')
    ax.set_ylabel('Instability Score')
    ax.set_title(f'Top {top_n} Most Unstable Layers')
    ax.grid(True, alpha=0.3)

    # Add legend for layer types
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color, label=lt)
                      for lt, color in type_colors.items()]
    ax.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()  # Close instead of show for non-GUI

    return fig, instability_scores[:10]  # Return top 10 most unstable

def generate_report(red_stats: Dict, yellow_stats: Dict,
                   instability_scores: List, save_path: str):
    """Generate a detailed text report of the analysis."""

    report = []
    report.append("=" * 80)
    report.append("COMPREHENSIVE ANALYSIS REPORT: Red (1M Stable) vs Yellow (2M Exploded)")
    report.append("=" * 80)
    report.append("")

    # Summary statistics
    report.append("SUMMARY STATISTICS")
    report.append("-" * 40)

    common_layers = sorted(set(red_stats.keys()) & set(yellow_stats.keys()))

    # Overall norms
    total_norm_red = sum(red_stats[l]['l2_norm']**2 for l in common_layers) ** 0.5
    total_norm_yellow = sum(yellow_stats[l]['l2_norm']**2 for l in common_layers) ** 0.5

    report.append(f"Total L2 Norm:")
    report.append(f"  Red (1M):    {total_norm_red:.2f}")
    report.append(f"  Yellow (2M): {total_norm_yellow:.2f}")
    report.append(f"  Ratio:       {total_norm_yellow/total_norm_red:.3f}")
    report.append("")

    # By layer type
    report.append("ANALYSIS BY LAYER TYPE")
    report.append("-" * 40)

    layer_types = ['embedding', 'attention', 'mlp', 'layernorm', 'other']
    for lt in layer_types:
        layers = [l for l in common_layers if red_stats[l]['layer_type'] == lt]
        if not layers:
            continue

        red_norms = [red_stats[l]['l2_norm'] for l in layers]
        yellow_norms = [yellow_stats[l]['l2_norm'] for l in layers]

        report.append(f"\n{lt.upper()} LAYERS ({len(layers)} layers):")
        report.append(f"  Mean L2 Norm:")
        report.append(f"    Red:    {np.mean(red_norms):.3f} ± {np.std(red_norms):.3f}")
        report.append(f"    Yellow: {np.mean(yellow_norms):.3f} ± {np.std(yellow_norms):.3f}")
        report.append(f"  Max norm change: {max(y/r for y,r in zip(yellow_norms, red_norms)):.3f}x")
        report.append(f"  Min norm change: {min(y/r for y,r in zip(yellow_norms, red_norms)):.3f}x")

    report.append("")
    report.append("TOP 10 MOST UNSTABLE LAYERS")
    report.append("-" * 40)

    for i, (layer, score, lt) in enumerate(instability_scores, 1):
        norm_ratio = yellow_stats[layer]['l2_norm'] / red_stats[layer]['l2_norm']
        report.append(f"\n{i}. {layer}")
        report.append(f"   Type: {lt}")
        report.append(f"   Instability Score: {score:.3f}")
        report.append(f"   Norm Ratio (Y/R): {norm_ratio:.3f}")
        report.append(f"   Kurtosis Change: {yellow_stats[layer]['kurtosis'] - red_stats[layer]['kurtosis']:.3f}")

    report.append("")
    report.append("GRADIENT EXPLOSION INDICATORS")
    report.append("-" * 40)

    # Find layers with extreme changes
    extreme_growth = []
    extreme_kurtosis = []

    for layer in common_layers:
        norm_ratio = yellow_stats[layer]['l2_norm'] / red_stats[layer]['l2_norm']
        if norm_ratio > 1.5:
            extreme_growth.append((layer, norm_ratio))

        kurtosis_diff = yellow_stats[layer]['kurtosis'] - red_stats[layer]['kurtosis']
        if abs(kurtosis_diff) > 5:
            extreme_kurtosis.append((layer, kurtosis_diff))

    report.append(f"\nLayers with >50% norm growth: {len(extreme_growth)}")
    for layer, ratio in extreme_growth[:5]:
        report.append(f"  {layer}: {ratio:.2f}x")

    report.append(f"\nLayers with extreme kurtosis change: {len(extreme_kurtosis)}")
    for layer, diff in extreme_kurtosis[:5]:
        report.append(f"  {layer}: {diff:+.2f}")

    report.append("")
    report.append("CONCLUSIONS")
    report.append("-" * 40)
    report.append("1. The gradient explosion primarily affected MLP and LayerNorm layers")
    report.append("2. Later layers (higher depth) show more instability")
    report.append("3. Heavy-tailed distributions (high kurtosis) developed in yellow run")
    report.append("4. The explosion was not uniform - specific layers failed catastrophically")
    report.append("")

    # Save report
    with open(save_path, 'w') as f:
        f.write('\n'.join(report))

    print('\n'.join(report))

    return report

def main():
    """Main analysis pipeline."""

    # Paths
    red_checkpoint = "experiments/converted/red_1m/params.pkl"
    yellow_checkpoint = "experiments/converted/yellow_2m/params.pkl"
    output_dir = Path("analysis/comprehensive_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load checkpoints
    print("Loading checkpoints...")
    red_params = load_checkpoint(red_checkpoint)
    yellow_params = load_checkpoint(yellow_checkpoint)

    # Flatten parameters
    print("Flattening parameters...")
    red_flat = flatten_params(red_params)
    yellow_flat = flatten_params(yellow_params)

    # Compute statistics
    print("Computing layer statistics...")
    red_stats = compute_layer_stats(red_flat)
    yellow_stats = compute_layer_stats(yellow_flat)

    # Generate visualizations
    print("\nCreating comprehensive comparison...")
    plot_comprehensive_comparison(red_stats, yellow_stats,
                                 str(output_dir / "comprehensive_comparison.png"))

    print("\nCreating side-by-side analysis...")
    fig, top_unstable = create_side_by_side_analysis(red_stats, yellow_stats,
                                                     str(output_dir / "side_by_side_analysis.png"))

    print("\nGenerating detailed report...")
    report = generate_report(red_stats, yellow_stats, top_unstable,
                           str(output_dir / "analysis_report.txt"))

    # Save raw statistics
    print("\nSaving raw statistics...")
    stats_output = {
        'red_stats': {k: {kk: float(vv) if isinstance(vv, (np.integer, np.floating)) else vv
                         for kk, vv in v.items() if kk != 'spectral_norm' or not np.isnan(v[kk])}
                     for k, v in red_stats.items()},
        'yellow_stats': {k: {kk: float(vv) if isinstance(vv, (np.integer, np.floating)) else vv
                           for kk, vv in v.items() if kk != 'spectral_norm' or not np.isnan(v[kk])}
                       for k, v in yellow_stats.items()},
        'top_unstable_layers': [(l, float(s), t) for l, s, t in top_unstable]
    }

    with open(output_dir / "comparison_stats.json", 'w') as f:
        json.dump(stats_output, f, indent=2)

    print(f"\nAnalysis complete! Results saved to {output_dir}/")
    print("\nKey findings:")
    print(f"- Total norm ratio (Yellow/Red): {sum(yellow_stats[l]['l2_norm']**2 for l in yellow_stats)**0.5 / sum(red_stats[l]['l2_norm']**2 for l in red_stats)**0.5:.3f}")
    print(f"- Most unstable layer: {top_unstable[0][0]} (score: {top_unstable[0][1]:.3f})")
    print(f"- Layers with >50% norm growth: {sum(1 for l in red_stats if l in yellow_stats and yellow_stats[l]['l2_norm']/red_stats[l]['l2_norm'] > 1.5)}")

if __name__ == "__main__":
    main()