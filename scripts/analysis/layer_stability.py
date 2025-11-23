#!/usr/bin/env python3
"""
Layer-wise stability analysis for tracking parameter evolution.

PURPOSE: Analyze parameter stability and update dynamics either within a single
training run OR between different runs. This tool is versatile and provides
complementary metrics to both spectral and distribution analysis.

USE THIS TOOL for:
- Tracking parameter norm growth (detecting instability)
- Computing update ratios ||Δθ||/||θ|| (training health indicator)
- Identifying which layers are still actively learning
- Detecting gradient explosion warning signs
- Comparing layer-wise statistics across any checkpoints

Key metrics per layer:
- Parameter norms (L1, L2, L∞)
- Update magnitude and direction
- Cosine similarity (alignment of parameters)
- Statistical moments (mean, std, skewness, kurtosis)

Example usage:
  # Within same run (tracks evolution)
  python layer_stability.py experiments/red_1m/990000.pkl experiments/red_1m/1000000.pkl

  # Across different runs (compares final states)
  python layer_stability.py experiments/red_1m/1000000.pkl experiments/yellow_2m/2000000.pkl

Healthy training typically shows:
- Decreasing update ratios over time
- Stable or slowly growing parameter norms
- Layer norms that don't explode exponentially
"""

import pickle
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
import json
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


def load_checkpoint(path: str) -> Dict[str, np.ndarray]:
    """Load converted checkpoint from pickle file."""
    with open(path, 'rb') as f:
        params = pickle.load(f)
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
    """Analyze stability between two checkpoints."""
    flat1 = flatten_params(params1)
    flat2 = flatten_params(params2)

    results = {}

    for name in flat1.keys():
        if name not in flat2:
            print(f"Warning: {name} not in second checkpoint")
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

        # Update ratio (like in training)
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


def visualize_stability(results: Dict[str, Any], output_dir: str):
    """Create comprehensive stability visualizations."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')

    # Extract layer names and organize by type
    layer_names = list(results.keys())
    attention_layers = [l for l in layer_names if 'attention' in l]
    mlp_layers = [l for l in layer_names if 'mlp' in l or 'intermediate' in l]
    embedding_layers = [l for l in layer_names if 'embedding' in l]
    norm_layers = [l for l in layer_names if 'norm' in l or 'bias' in l]

    # Create figure
    fig = plt.figure(figsize=(24, 16))

    # 1. Layer-wise L2 norm changes
    ax1 = plt.subplot(3, 4, 1)
    l2_ratios = [results[l]['l2_norm_ratio'] for l in layer_names]

    # Color by layer type
    colors = []
    for l in layer_names:
        if l in attention_layers:
            colors.append('blue')
        elif l in mlp_layers:
            colors.append('green')
        elif l in embedding_layers:
            colors.append('red')
        else:
            colors.append('gray')

    ax1.scatter(range(len(layer_names)), l2_ratios, c=colors, alpha=0.6)
    ax1.axhline(y=1, color='black', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Layer Index')
    ax1.set_ylabel('L2 Norm Ratio (2M/1M)')
    ax1.set_title('Parameter Norm Evolution')

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='blue', label='Attention'),
                       Patch(facecolor='green', label='MLP'),
                       Patch(facecolor='red', label='Embedding'),
                       Patch(facecolor='gray', label='Other')]
    ax1.legend(handles=legend_elements, loc='upper right')

    # 2. Update ratios (gradient-like metric)
    ax2 = plt.subplot(3, 4, 2)
    update_ratios = [results[l]['update_ratio'] for l in layer_names]
    ax2.bar(range(len(layer_names)), update_ratios, color=colors, alpha=0.6)
    ax2.set_xlabel('Layer Index')
    ax2.set_ylabel('Update Ratio ||Δθ||/||θ||')
    ax2.set_title('Parameter Update Magnitudes')
    ax2.axhline(y=0.1, color='red', linestyle='--', alpha=0.5, label='Typical threshold')
    ax2.legend()

    # 3. Angular distance histogram
    ax3 = plt.subplot(3, 4, 3)
    angular_distances = [results[l]['angular_distance'] for l in layer_names]
    ax3.hist(angular_distances, bins=30, alpha=0.7, color='purple')
    ax3.set_xlabel('Angular Distance (degrees)')
    ax3.set_ylabel('Number of Layers')
    ax3.set_title('Parameter Direction Changes')
    ax3.axvline(x=np.mean(angular_distances), color='red', linestyle='--',
                label=f'Mean: {np.mean(angular_distances):.1f}°')
    ax3.legend()

    # 4. Cosine similarity by layer type
    ax4 = plt.subplot(3, 4, 4)

    def get_cosine_sims(layers):
        return [results[l]['cosine_similarity'] for l in layers if l in results]

    att_cos = get_cosine_sims(attention_layers)
    mlp_cos = get_cosine_sims(mlp_layers)
    emb_cos = get_cosine_sims(embedding_layers)

    box_data = []
    labels = []
    if att_cos:
        box_data.append(att_cos)
        labels.append('Attention')
    if mlp_cos:
        box_data.append(mlp_cos)
        labels.append('MLP')
    if emb_cos:
        box_data.append(emb_cos)
        labels.append('Embedding')

    if box_data:
        bp = ax4.boxplot(box_data, labels=labels)
        ax4.set_ylabel('Cosine Similarity')
        ax4.set_title('Parameter Alignment by Layer Type')
        ax4.axhline(y=1, color='black', linestyle='--', alpha=0.3)

    # 5. Layer depth analysis (for encoder layers)
    ax5 = plt.subplot(3, 4, 5)

    # Extract encoder layer numbers
    encoder_updates = {}
    for name, res in results.items():
        if 'encoder.layers_' in name:
            # Extract layer number
            parts = name.split('.')
            for part in parts:
                if part.startswith('layers_'):
                    layer_num = int(part.split('_')[1])
                    if layer_num not in encoder_updates:
                        encoder_updates[layer_num] = []
                    encoder_updates[layer_num].append(res['update_ratio'])
                    break

    if encoder_updates:
        layer_nums = sorted(encoder_updates.keys())
        mean_updates = [np.mean(encoder_updates[i]) for i in layer_nums]
        std_updates = [np.std(encoder_updates[i]) for i in layer_nums]

        ax5.errorbar(layer_nums, mean_updates, yerr=std_updates,
                    marker='o', capsize=5, capthick=2)
        ax5.set_xlabel('Encoder Layer')
        ax5.set_ylabel('Mean Update Ratio')
        ax5.set_title('Update Magnitude by Layer Depth')
        ax5.grid(True, alpha=0.3)

    # 6. Standard deviation changes
    ax6 = plt.subplot(3, 4, 6)
    std_ratios = [results[l]['std_ratio'] for l in layer_names]
    ax6.scatter(range(len(layer_names)), std_ratios, c=colors, alpha=0.6)
    ax6.axhline(y=1, color='black', linestyle='--', alpha=0.5)
    ax6.set_xlabel('Layer Index')
    ax6.set_ylabel('Std Ratio (2M/1M)')
    ax6.set_title('Parameter Variance Evolution')

    # 7. Maximum absolute changes
    ax7 = plt.subplot(3, 4, 7)
    max_changes = [results[l]['max_change'] for l in layer_names]
    ax7.bar(range(len(layer_names)), max_changes, color=colors, alpha=0.6)
    ax7.set_xlabel('Layer Index')
    ax7.set_ylabel('Max Absolute Change')
    ax7.set_title('Largest Parameter Changes')
    ax7.set_yscale('log')

    # 8. Sparsity changes
    ax8 = plt.subplot(3, 4, 8)
    sparsity_changes = [results[l]['sparsity_change'] for l in layer_names]
    positive_sparse = [s if s > 0 else 0 for s in sparsity_changes]
    negative_sparse = [s if s < 0 else 0 for s in sparsity_changes]

    ax8.bar(range(len(layer_names)), positive_sparse, color='red', alpha=0.6, label='Increased')
    ax8.bar(range(len(layer_names)), negative_sparse, color='blue', alpha=0.6, label='Decreased')
    ax8.set_xlabel('Layer Index')
    ax8.set_ylabel('Sparsity Change')
    ax8.set_title('Evolution of Near-Zero Values')
    ax8.legend()

    # 9. Kurtosis analysis (distribution shape)
    ax9 = plt.subplot(3, 4, 9)
    kurt1 = [results[l]['stats1'].kurtosis for l in layer_names]
    kurt2 = [results[l]['stats2'].kurtosis for l in layer_names]

    ax9.scatter(kurt1, kurt2, c=colors, alpha=0.6)
    ax9.plot([min(kurt1), max(kurt1)], [min(kurt1), max(kurt1)],
             'k--', alpha=0.3, label='No change')
    ax9.set_xlabel('Kurtosis at 1M')
    ax9.set_ylabel('Kurtosis at 2M')
    ax9.set_title('Distribution Shape Evolution')
    ax9.legend()

    # 10. Layer-wise stability score
    ax10 = plt.subplot(3, 4, 10)

    # Compute stability score (lower is more stable)
    stability_scores = []
    for l in layer_names:
        score = (
            results[l]['update_ratio'] * 100 +  # Weight update magnitude
            abs(1 - results[l]['cosine_similarity']) * 50 +  # Direction change
            abs(1 - results[l]['l2_norm_ratio']) * 20  # Norm change
        )
        stability_scores.append(score)

    # Find most and least stable layers
    sorted_idx = np.argsort(stability_scores)
    top_stable = sorted_idx[:5]
    top_unstable = sorted_idx[-5:]

    ax10.bar(range(len(layer_names)), stability_scores, color=colors, alpha=0.6)
    ax10.set_xlabel('Layer Index')
    ax10.set_ylabel('Instability Score')
    ax10.set_title('Composite Stability Metric')

    # Mark extremes
    for idx in top_unstable:
        ax10.axvline(x=idx, color='red', alpha=0.3, linestyle=':')

    # 11-12. Summary statistics
    ax11 = plt.subplot(3, 4, 11)
    ax11.axis('off')

    # Compute summary statistics
    mean_update = np.mean(update_ratios)
    max_update = np.max(update_ratios)
    mean_angular = np.mean(angular_distances)
    mean_cosine = np.mean([results[l]['cosine_similarity'] for l in layer_names])

    summary1 = f"""Global Statistics:

Mean Update Ratio: {mean_update:.4f}
Max Update Ratio: {max_update:.4f}
Mean Angular Distance: {mean_angular:.1f}°
Mean Cosine Similarity: {mean_cosine:.4f}

Layers with >10% updates: {np.sum(np.array(update_ratios) > 0.1)}
Layers with >45° rotation: {np.sum(np.array(angular_distances) > 45)}
"""

    ax11.text(0.1, 0.5, summary1, fontsize=11, family='monospace',
             verticalalignment='center')

    ax12 = plt.subplot(3, 4, 12)
    ax12.axis('off')

    # Most unstable layers
    unstable_text = "Most Unstable Layers:\n\n"
    for i, idx in enumerate(top_unstable[-5:]):
        name = layer_names[idx]
        short_name = name.split('.')[-2] if '.' in name else name
        unstable_text += f"{i+1}. {short_name}\n"
        unstable_text += f"   Update: {update_ratios[idx]:.3f}\n"

    ax12.text(0.1, 0.7, unstable_text, fontsize=10, family='monospace',
             verticalalignment='top')

    plt.suptitle('Layer Stability Analysis: Red (1M) vs Yellow (2M)',
                fontsize=16, fontweight='bold')
    plt.tight_layout()

    # Save figure
    output_path = output_dir / 'layer_stability.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to {output_path}")
    plt.close()

    # Save numerical results
    results_path = output_dir / 'stability_results.json'

    # Convert to serializable format
    json_results = {}
    for name, res in results.items():
        json_results[name] = {
            'shape': res['stats1'].shape,
            'num_params': res['stats1'].num_params,
            'l2_norm_1m': float(res['stats1'].l2_norm),
            'l2_norm_2m': float(res['stats2'].l2_norm),
            'update_ratio': float(res['update_ratio']),
            'cosine_similarity': float(res['cosine_similarity']),
            'angular_distance': float(res['angular_distance']),
            'l2_norm_ratio': float(res['l2_norm_ratio']),
            'max_change': float(res['max_change'])
        }

    with open(results_path, 'w') as f:
        json.dump(json_results, f, indent=2)

    print(f"Saved numerical results to {results_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Layer stability analysis")
    parser.add_argument("--checkpoint1", type=str, required=True)
    parser.add_argument("--checkpoint2", type=str, required=True)
    parser.add_argument("--output", type=str, default="analysis/stability")
    args = parser.parse_args()

    print("="*60)
    print("LAYER STABILITY ANALYSIS")
    print("="*60)

    print(f"\n1. Loading checkpoints...")
    params1 = load_checkpoint(args.checkpoint1)
    params2 = load_checkpoint(args.checkpoint2)

    print(f"2. Analyzing stability...")
    results = analyze_stability(params1, params2)

    # Print key findings
    print("\n" + "="*60)
    print("KEY FINDINGS")
    print("="*60)

    # Find most changed layers
    sorted_layers = sorted(results.items(),
                          key=lambda x: x[1]['update_ratio'],
                          reverse=True)

    print("\nTop 5 most changed layers (by update ratio):")
    for name, res in sorted_layers[:5]:
        print(f"  {name}")
        print(f"    Update ratio: {res['update_ratio']:.4f}")
        print(f"    Angular distance: {res['angular_distance']:.1f}°")
        print(f"    L2 norm ratio: {res['l2_norm_ratio']:.3f}")

    print("\nLayers with significant direction changes (>45°):")
    rotated = [(n, r) for n, r in results.items() if r['angular_distance'] > 45]
    for name, res in rotated[:5]:
        print(f"  {name}: {res['angular_distance']:.1f}°")

    print(f"\n3. Creating visualizations...")
    visualize_stability(results, args.output)

    print("\n✓ Analysis complete!")


if __name__ == "__main__":
    main()