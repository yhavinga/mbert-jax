#!/usr/bin/env python3
"""
Spectral analysis of checkpoint parameters for WITHIN-RUN dynamics.

IMPORTANT: This tool is designed to analyze the evolution of spectral properties
across checkpoints from THE SAME training run. It tracks how eigenvalues,
singular values, and conditioning evolve during training.

DO NOT USE for comparing checkpoints from different random initializations!
For cross-run comparison, use distribution_analyzer.py instead.

Key insights from spectral analysis:
- Spectral radius growth predicts gradient explosions 10-100 steps in advance
- Condition number reveals optimization health and convergence
- Effective rank tracks learned representation complexity
- Edge of Stability: Networks train near the divergence boundary

Example usage:
  python spectral_analyzer.py experiments/red_1m/990000.pkl experiments/red_1m/1000000.pkl

This reveals how the spectral properties evolved in the final 10k steps of training.
"""

import pickle
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import linalg
from dataclasses import dataclass
import json


@dataclass
class SpectralStats:
    """Container for spectral analysis results."""
    layer_name: str
    shape: Tuple[int, ...]
    spectral_norm: float  # Largest singular value
    nuclear_norm: float   # Sum of singular values
    condition_number: float  # Ratio of largest to smallest singular value
    effective_rank: float  # Number of significant singular values
    frobenius_norm: float
    singular_values: np.ndarray
    eigenvalues: np.ndarray = None  # Only for square matrices


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


def compute_spectral_stats(matrix: np.ndarray, layer_name: str) -> SpectralStats:
    """Compute spectral statistics for a parameter matrix."""
    # Ensure 2D matrix for analysis
    original_shape = matrix.shape
    if len(matrix.shape) > 2:
        # Reshape to 2D for analysis (e.g., conv kernels)
        matrix = matrix.reshape(matrix.shape[0], -1)
    elif len(matrix.shape) == 1:
        # Skip 1D parameters (biases, layer norms)
        return None

    # Compute SVD
    try:
        U, s, Vh = linalg.svd(matrix, full_matrices=False)

        # Spectral norm (largest singular value)
        spectral_norm = s[0] if len(s) > 0 else 0

        # Nuclear norm (sum of singular values)
        nuclear_norm = np.sum(s)

        # Condition number
        condition_number = s[0] / s[-1] if len(s) > 0 and s[-1] > 1e-10 else np.inf

        # Effective rank (using threshold of 0.01 * max singular value)
        threshold = 0.01 * s[0] if len(s) > 0 else 0
        effective_rank = np.sum(s > threshold)

        # Frobenius norm
        frobenius_norm = np.linalg.norm(matrix, 'fro')

        # Eigenvalues (only for square matrices)
        eigenvalues = None
        if matrix.shape[0] == matrix.shape[1]:
            try:
                eigenvalues = linalg.eigvals(matrix)
            except:
                pass

        return SpectralStats(
            layer_name=layer_name,
            shape=original_shape,
            spectral_norm=spectral_norm,
            nuclear_norm=nuclear_norm,
            condition_number=condition_number,
            effective_rank=effective_rank,
            frobenius_norm=frobenius_norm,
            singular_values=s[:min(100, len(s))],  # Keep top 100 singular values
            eigenvalues=eigenvalues
        )
    except Exception as e:
        print(f"Error computing spectral stats for {layer_name}: {e}")
        return None


def analyze_checkpoint(checkpoint_path: str) -> Dict[str, SpectralStats]:
    """Perform spectral analysis on all parameters in checkpoint."""
    params = load_checkpoint(checkpoint_path)
    flat_params = flatten_params(params)

    results = {}
    for name, param in flat_params.items():
        if param.ndim >= 2 and param.size > 1:  # Only analyze matrices
            stats = compute_spectral_stats(param, name)
            if stats:
                results[name] = stats
                print(f"  {name}: shape={stats.shape}, spectral_norm={stats.spectral_norm:.4f}, "
                      f"condition={stats.condition_number:.2f}, rank={stats.effective_rank}")

    return results


def compare_spectral_properties(stats1: Dict[str, SpectralStats],
                                stats2: Dict[str, SpectralStats]) -> Dict[str, Any]:
    """Compare spectral properties between two checkpoints."""
    comparison = {}

    for layer_name in stats1.keys():
        if layer_name in stats2:
            s1, s2 = stats1[layer_name], stats2[layer_name]

            comparison[layer_name] = {
                'spectral_norm_ratio': s2.spectral_norm / s1.spectral_norm,
                'spectral_norm_diff': s2.spectral_norm - s1.spectral_norm,
                'nuclear_norm_ratio': s2.nuclear_norm / s1.nuclear_norm,
                'condition_ratio': s2.condition_number / s1.condition_number,
                'rank_change': s2.effective_rank - s1.effective_rank,
                'frobenius_ratio': s2.frobenius_norm / s1.frobenius_norm,
            }

    return comparison


def visualize_spectral_analysis(stats1: Dict[str, SpectralStats],
                               stats2: Dict[str, SpectralStats],
                               comparison: Dict[str, Any],
                               output_dir: str):
    """Create comprehensive visualization of spectral analysis."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")

    # Create figure with subplots
    fig = plt.figure(figsize=(20, 15))

    # 1. Spectral norm comparison
    ax1 = plt.subplot(3, 3, 1)
    layer_names = list(comparison.keys())
    spectral_ratios = [comparison[l]['spectral_norm_ratio'] for l in layer_names]

    # Group by layer type
    attention_layers = [i for i, l in enumerate(layer_names) if 'attention' in l]
    mlp_layers = [i for i, l in enumerate(layer_names) if 'mlp' in l or 'intermediate' in l]
    other_layers = [i for i, l in enumerate(layer_names)
                   if i not in attention_layers and i not in mlp_layers]

    ax1.bar(attention_layers, [spectral_ratios[i] for i in attention_layers],
           label='Attention', alpha=0.7, color='blue')
    ax1.bar(mlp_layers, [spectral_ratios[i] for i in mlp_layers],
           label='MLP', alpha=0.7, color='green')
    ax1.bar(other_layers, [spectral_ratios[i] for i in other_layers],
           label='Other', alpha=0.7, color='red')

    ax1.axhline(y=1, color='black', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Layer Index')
    ax1.set_ylabel('Spectral Norm Ratio (2M/1M)')
    ax1.set_title('Spectral Norm Changes')
    ax1.legend()

    # 2. Condition number comparison
    ax2 = plt.subplot(3, 3, 2)
    cond_ratios = [comparison[l]['condition_ratio'] for l in layer_names]
    ax2.scatter(range(len(cond_ratios)), cond_ratios, alpha=0.6)
    ax2.axhline(y=1, color='black', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Layer Index')
    ax2.set_ylabel('Condition Number Ratio (2M/1M)')
    ax2.set_title('Conditioning Changes')
    ax2.set_yscale('log')

    # 3. Effective rank changes
    ax3 = plt.subplot(3, 3, 3)
    rank_changes = [comparison[l]['rank_change'] for l in layer_names]
    colors = ['red' if r < 0 else 'green' for r in rank_changes]
    ax3.bar(range(len(rank_changes)), rank_changes, color=colors, alpha=0.6)
    ax3.set_xlabel('Layer Index')
    ax3.set_ylabel('Rank Change (2M - 1M)')
    ax3.set_title('Effective Rank Changes')

    # 4. Singular value distributions (sample layers)
    sample_layers = [l for l in layer_names if 'encoder.layer_2' in l and 'kernel' in l][:3]

    for idx, layer in enumerate(sample_layers[:3]):
        ax = plt.subplot(3, 3, 4 + idx)
        s1 = stats1[layer].singular_values[:50]
        s2 = stats2[layer].singular_values[:50]

        ax.semilogy(s1, label='1M steps', alpha=0.7, linewidth=2)
        ax.semilogy(s2, label='2M steps', alpha=0.7, linewidth=2)
        ax.set_xlabel('Singular Value Index')
        ax.set_ylabel('Singular Value')
        ax.set_title(f'SV Spectrum: {layer.split(".")[-2]}')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # 5. Layer-wise norm analysis
    ax7 = plt.subplot(3, 3, 7)
    frob_ratios = [comparison[l]['frobenius_ratio'] for l in layer_names]
    ax7.plot(frob_ratios, 'o-', alpha=0.7, label='Frobenius')
    nuclear_ratios = [comparison[l]['nuclear_norm_ratio'] for l in layer_names]
    ax7.plot(nuclear_ratios, 's-', alpha=0.7, label='Nuclear')
    ax7.axhline(y=1, color='black', linestyle='--', alpha=0.5)
    ax7.set_xlabel('Layer Index')
    ax7.set_ylabel('Norm Ratio (2M/1M)')
    ax7.set_title('Norm Evolution')
    ax7.legend()

    # 6. Spectral radius heatmap (for square matrices with eigenvalues)
    ax8 = plt.subplot(3, 3, 8)
    eigen_layers = [l for l in layer_names if stats1[l].eigenvalues is not None][:10]
    if eigen_layers:
        spectral_radii = []
        for layer in eigen_layers:
            eig1 = np.max(np.abs(stats1[layer].eigenvalues))
            eig2 = np.max(np.abs(stats2[layer].eigenvalues))
            spectral_radii.append([eig1, eig2])

        im = ax8.imshow(np.array(spectral_radii).T, aspect='auto', cmap='coolwarm')
        ax8.set_yticks([0, 1])
        ax8.set_yticklabels(['1M', '2M'])
        ax8.set_xlabel('Layer Index')
        ax8.set_title('Spectral Radius Evolution')
        plt.colorbar(im, ax=ax8)

    # 7. Summary statistics
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')

    # Compute summary stats
    avg_spectral_increase = np.mean(spectral_ratios)
    max_spectral_increase = np.max(spectral_ratios)
    avg_cond_increase = np.mean([c for c in cond_ratios if c < np.inf])
    total_rank_change = np.sum(rank_changes)

    summary_text = f"""
    Summary Statistics:

    Avg Spectral Norm Increase: {avg_spectral_increase:.3f}x
    Max Spectral Norm Increase: {max_spectral_increase:.3f}x
    Avg Condition # Increase: {avg_cond_increase:.3f}x
    Total Rank Change: {total_rank_change:+.0f}

    Layers with >20% spectral increase: {np.sum(np.array(spectral_ratios) > 1.2)}
    Layers with condition degradation: {np.sum(np.array(cond_ratios) > 2)}
    """

    ax9.text(0.1, 0.5, summary_text, fontsize=12, family='monospace',
            verticalalignment='center')

    plt.suptitle('Spectral Analysis: Red (1M) vs Yellow (2M) Checkpoints',
                fontsize=16, fontweight='bold')
    plt.tight_layout()

    # Save figure
    output_path = output_dir / 'spectral_analysis.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to {output_path}")
    plt.close()

    # Save numerical results
    results_path = output_dir / 'spectral_results.json'

    # Convert to serializable format
    results = {
        'summary': {
            'avg_spectral_increase': float(avg_spectral_increase),
            'max_spectral_increase': float(max_spectral_increase),
            'avg_condition_increase': float(avg_cond_increase),
            'total_rank_change': float(total_rank_change),
        },
        'layer_comparison': comparison
    }

    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if isinstance(x, np.number) else str(x))

    print(f"Saved numerical results to {results_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Spectral analysis of checkpoints")
    parser.add_argument("--checkpoint1", type=str, required=True,
                       help="Path to first checkpoint (e.g., experiments/converted/red_1m/params.pkl)")
    parser.add_argument("--checkpoint2", type=str, required=True,
                       help="Path to second checkpoint (e.g., experiments/converted/yellow_2m/params.pkl)")
    parser.add_argument("--output", type=str, default="analysis/spectral",
                       help="Output directory for results")
    args = parser.parse_args()

    print("="*60)
    print("SPECTRAL ANALYSIS")
    print("="*60)

    print(f"\n1. Analyzing checkpoint 1: {args.checkpoint1}")
    stats1 = analyze_checkpoint(args.checkpoint1)

    print(f"\n2. Analyzing checkpoint 2: {args.checkpoint2}")
    stats2 = analyze_checkpoint(args.checkpoint2)

    print(f"\n3. Comparing spectral properties...")
    comparison = compare_spectral_properties(stats1, stats2)

    # Print key findings
    print("\n" + "="*60)
    print("KEY FINDINGS")
    print("="*60)

    # Find layers with largest spectral norm increase
    sorted_layers = sorted(comparison.items(),
                          key=lambda x: x[1]['spectral_norm_ratio'],
                          reverse=True)

    print("\nTop 5 layers with largest spectral norm increase:")
    for layer, comp in sorted_layers[:5]:
        print(f"  {layer}: {comp['spectral_norm_ratio']:.3f}x increase")

    print("\nTop 5 layers with worst conditioning:")
    sorted_cond = sorted(comparison.items(),
                        key=lambda x: x[1]['condition_ratio'],
                        reverse=True)
    for layer, comp in sorted_cond[:5]:
        if comp['condition_ratio'] < np.inf:
            print(f"  {layer}: condition ratio {comp['condition_ratio']:.2f}x")

    print(f"\n4. Creating visualizations...")
    visualize_spectral_analysis(stats1, stats2, comparison, args.output)

    print("\n✓ Analysis complete!")


if __name__ == "__main__":
    main()