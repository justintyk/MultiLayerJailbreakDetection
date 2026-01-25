"""
Phase 3: Dimensionality Reduction and Cluster-Based Analysis

This script implements Phase 3 from docs/Impl.pdf:
- Manifold Mapping: Apply PCA and K-Means/DBSCAN to {δfi} to identify distinct clusters
- Semantic Mapping: Assign each cluster the dominant label (from rubrics or AO)
- Outputs cluster analysis with labels and statistics

Usage:
    python src/cluster_interventions.py --all
    python src/cluster_interventions.py --n-clusters 5
    python src/cluster_interventions.py --method dbscan
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from collections import Counter

import numpy as np
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from rubrics import RubricDefinition


def load_intervention_logs(log_path: Path) -> List[Dict]:
    """Load intervention logs from JSONL file."""
    records = []
    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def extract_delta_f_matrix(records: List[Dict]) -> Tuple[np.ndarray, List[Dict]]:
    """
    Extract δf vectors from records into a matrix.
    
    Returns:
        delta_f_matrix: [N, D] matrix of intervention vectors
        valid_records: Records that had valid delta_f
    """
    delta_fs = []
    valid_records = []
    
    for record in records:
        if 'delta_f' in record:
            delta_f = np.array(record['delta_f'], dtype=np.float32)
            delta_fs.append(delta_f)
            valid_records.append(record)
    
    if not delta_fs:
        raise ValueError("No delta_f vectors found in records")
    
    return np.stack(delta_fs), valid_records


def apply_pca(delta_f_matrix: np.ndarray, n_components: int = 50) -> Tuple[np.ndarray, object]:
    """
    Apply PCA for dimensionality reduction.
    
    Args:
        delta_f_matrix: [N, D] matrix
        n_components: Number of PCA components (or fraction of variance)
        
    Returns:
        reduced: [N, n_components] reduced matrix
        pca: Fitted PCA object
    """
    from sklearn.decomposition import PCA
    
    # Adjust n_components if larger than available
    n_samples, n_features = delta_f_matrix.shape
    max_components = min(n_samples, n_features)
    n_components = min(n_components, max_components)
    
    pca = PCA(n_components=n_components)
    reduced = pca.fit_transform(delta_f_matrix)
    
    explained_var = sum(pca.explained_variance_ratio_) * 100
    print(f"  PCA: {n_features}D → {n_components}D (explains {explained_var:.1f}% variance)")
    
    return reduced, pca


def cluster_kmeans(reduced_matrix: np.ndarray, n_clusters: int = 5) -> np.ndarray:
    """
    Apply K-Means clustering.
    
    Args:
        reduced_matrix: [N, D] reduced matrix
        n_clusters: Number of clusters
        
    Returns:
        labels: Cluster labels for each sample
    """
    from sklearn.cluster import KMeans
    
    # Adjust n_clusters if larger than n_samples
    n_samples = reduced_matrix.shape[0]
    n_clusters = min(n_clusters, n_samples)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(reduced_matrix)
    
    return labels


def cluster_dbscan(reduced_matrix: np.ndarray, eps: float = 0.5, min_samples: int = 2) -> np.ndarray:
    """
    Apply DBSCAN clustering.
    
    Args:
        reduced_matrix: [N, D] reduced matrix
        eps: Maximum distance between samples
        min_samples: Minimum samples in neighborhood
        
    Returns:
        labels: Cluster labels (-1 for noise)
    """
    from sklearn.cluster import DBSCAN
    from sklearn.preprocessing import StandardScaler
    
    # Normalize for DBSCAN
    scaler = StandardScaler()
    normalized = scaler.fit_transform(reduced_matrix)
    
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(normalized)
    
    return labels


def assign_cluster_labels(
    cluster_labels: np.ndarray,
    records: List[Dict]
) -> Dict[int, Dict]:
    """
    Assign semantic labels to clusters based on dominant rubric in each cluster.
    
    This implements the "Semantic Mapping" step from Phase 3:
    Assign each cluster Ck the dominant label.
    
    Args:
        cluster_labels: Array of cluster assignments
        records: Original records with rubric information
        
    Returns:
        cluster_info: Dict mapping cluster_id -> {label, count, avg_score, rubrics, ...}
    """
    cluster_info = {}
    unique_clusters = set(cluster_labels)
    
    for cluster_id in unique_clusters:
        # Get indices of samples in this cluster
        mask = cluster_labels == cluster_id
        indices = np.where(mask)[0]
        
        # Collect rubric texts and scores
        rubric_texts = []
        scores = []
        prompts = []
        
        for idx in indices:
            record = records[idx]
            rubric_texts.append(record.get('rubric_text', 'Unknown'))
            scores.append(record.get('score', 0))
            if 'prompt' in record:
                prompts.append(record['prompt'])
        
        # Find dominant rubric (most common)
        rubric_counts = Counter(rubric_texts)
        dominant_rubric, dominant_count = rubric_counts.most_common(1)[0]
        
        # Generate a short label from the dominant rubric
        label = generate_short_label(dominant_rubric)
        
        cluster_info[cluster_id] = {
            'label': label,
            'size': len(indices),
            'avg_score': np.mean(scores),
            'std_score': np.std(scores) if len(scores) > 1 else 0,
            'dominant_rubric': dominant_rubric,
            'rubric_distribution': dict(rubric_counts),
            'sample_prompts': prompts[:3],  # Keep up to 3 examples
            'indices': indices.tolist()
        }
    
    return cluster_info


def generate_short_label(rubric_text: str) -> str:
    """
    Generate a short semantic label from rubric text.
    
    Maps common rubric patterns to labels like those in PDF Table 1:
    - Ignore-System-Prompt
    - Latent-Harm-Injection
    - Roleplay-Jailbreak
    """
    rubric_lower = rubric_text.lower()
    
    # Pattern matching for common jailbreak mechanisms
    if 'system' in rubric_lower and ('ignore' in rubric_lower or 'bypass' in rubric_lower):
        return "Ignore-System-Prompt"
    elif 'roleplay' in rubric_lower or 'persona' in rubric_lower or 'character' in rubric_lower:
        return "Roleplay-Jailbreak"
    elif 'harm' in rubric_lower and ('inject' in rubric_lower or 'embed' in rubric_lower):
        return "Latent-Harm-Injection"
    elif 'obfuscat' in rubric_lower or 'encod' in rubric_lower or 'translat' in rubric_lower:
        return "Obfuscation-Attack"
    elif 'exam' in rubric_lower or 'cheat' in rubric_lower or 'test' in rubric_lower:
        return "Exam-Cheating"
    elif 'password' in rubric_lower or 'hash' in rubric_lower or 'security' in rubric_lower:
        return "Security-Bypass"
    elif 'plagiar' in rubric_lower:
        return "Plagiarism-Evasion"
    elif 'review' in rubric_lower or 'fake' in rubric_lower:
        return "Fake-Content-Generation"
    elif 'stalk' in rubric_lower or 'harass' in rubric_lower:
        return "Harassment-Content"
    elif 'context' in rubric_lower and 'manipulat' in rubric_lower:
        return "Context-Manipulation"
    else:
        # Generate label from first few words
        words = rubric_text.split()[:4]
        return "-".join(w.capitalize() for w in words if len(w) > 2)[:30]


def print_cluster_summary(cluster_info: Dict[int, Dict], method: str) -> None:
    """
    Print cluster summary in the format of PDF Table 1.
    """
    print("\n" + "="*80)
    print(f"CLUSTER ANALYSIS RESULTS ({method.upper()})")
    print("="*80)
    
    # Sort by cluster size (descending)
    sorted_clusters = sorted(
        cluster_info.items(), 
        key=lambda x: x[1]['size'], 
        reverse=True
    )
    
    # Print table header (matching PDF Table 1 format)
    print("\n" + "-"*80)
    print(f"{'Cluster':<8} {'Label (Semantic)':<30} {'Size (n)':<10} {'Avg Score':<12} {'Std':<8}")
    print("-"*80)
    
    total_samples = sum(info['size'] for _, info in sorted_clusters)
    
    for cluster_id, info in sorted_clusters:
        label = info['label']
        if cluster_id == -1:
            label = "NOISE (unclustered)"
        print(f"{cluster_id:<8} {label:<30} {info['size']:<10} {info['avg_score']:<12.3f} {info['std_score']:<8.3f}")
    
    print("-"*80)
    print(f"{'TOTAL':<8} {'':<30} {total_samples:<10}")
    
    # Print detailed info per cluster
    print("\n" + "="*80)
    print("CLUSTER DETAILS")
    print("="*80)
    
    for cluster_id, info in sorted_clusters:
        if cluster_id == -1:
            continue  # Skip noise cluster details
            
        print(f"\n[Cluster {cluster_id}: {info['label']}]")
        print(f"  Size: {info['size']} samples")
        print(f"  Avg Score: {info['avg_score']:.3f} ± {info['std_score']:.3f}")
        print(f"  Dominant Rubric: {info['dominant_rubric'][:80]}...")
        
        if len(info['rubric_distribution']) > 1:
            print(f"  Rubric Distribution:")
            for rubric, count in sorted(info['rubric_distribution'].items(), key=lambda x: -x[1])[:3]:
                print(f"    - {rubric[:60]}...: {count}")
        
        if info['sample_prompts']:
            print(f"  Sample Prompts:")
            for prompt in info['sample_prompts'][:2]:
                print(f"    - {prompt[:70]}...")


def save_cluster_results(
    cluster_info: Dict[int, Dict],
    records: List[Dict],
    cluster_labels: np.ndarray,
    output_path: Path,
    pca_components: Optional[np.ndarray] = None
) -> None:
    """Save cluster results to JSON."""
    
    # Add cluster assignment to each record
    labeled_records = []
    for i, record in enumerate(records):
        labeled_record = {
            k: v for k, v in record.items() if k != 'delta_f'  # Exclude large delta_f
        }
        labeled_record['cluster_id'] = int(cluster_labels[i])
        labeled_record['cluster_label'] = cluster_info[cluster_labels[i]]['label']
        labeled_records.append(labeled_record)
    
    # Create summary
    summary = {
        'timestamp': datetime.utcnow().isoformat(),
        'n_samples': len(records),
        'n_clusters': len([c for c in cluster_info.keys() if c != -1]),
        'clusters': {
            str(k): {
                'label': v['label'],
                'size': v['size'],
                'avg_score': v['avg_score'],
                'dominant_rubric': v['dominant_rubric']
            }
            for k, v in cluster_info.items()
        }
    }
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save summary
    summary_path = output_path.with_suffix('.summary.json')
    with summary_path.open('w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    
    # Save labeled records
    with output_path.open('w', encoding='utf-8') as f:
        for rec in labeled_records:
            f.write(json.dumps(rec) + '\n')
    
    print(f"\n✓ Saved cluster summary to: {summary_path}")
    print(f"✓ Saved labeled records to: {output_path}")


def visualize_clusters(
    reduced_matrix: np.ndarray,
    cluster_labels: np.ndarray,
    cluster_info: Dict[int, Dict],
    output_path: Optional[Path] = None
) -> None:
    """
    Visualize clusters using 2D projection.
    """
    try:
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA
    except ImportError:
        print("\n  Note: Install matplotlib for visualization: pip install matplotlib")
        return
    
    # Reduce to 2D for visualization
    if reduced_matrix.shape[1] > 2:
        pca_2d = PCA(n_components=2)
        coords_2d = pca_2d.fit_transform(reduced_matrix)
    else:
        coords_2d = reduced_matrix
    
    # Create plot
    plt.figure(figsize=(10, 8))
    
    unique_labels = set(cluster_labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    
    for cluster_id, color in zip(sorted(unique_labels), colors):
        mask = cluster_labels == cluster_id
        label = cluster_info[cluster_id]['label'] if cluster_id != -1 else "Noise"
        plt.scatter(
            coords_2d[mask, 0], 
            coords_2d[mask, 1],
            c=[color],
            label=f"C{cluster_id}: {label}",
            alpha=0.7,
            s=50
        )
    
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Intervention Clusters (PCA 2D Projection)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    if output_path:
        plot_path = output_path.with_suffix('.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved visualization to: {plot_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Phase 3: Cluster interventions and assign semantic labels"
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Path to specific intervention log file (JSONL)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all intervention logs in logs/interventions/"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="logs/clustered",
        help="Output directory for clustered results"
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["kmeans", "dbscan"],
        default="kmeans",
        help="Clustering method (default: kmeans)"
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=5,
        help="Number of clusters for K-Means (default: 5)"
    )
    parser.add_argument(
        "--pca-components",
        type=int,
        default=50,
        help="Number of PCA components (default: 50)"
    )
    parser.add_argument(
        "--dbscan-eps",
        type=float,
        default=0.5,
        help="DBSCAN eps parameter (default: 0.5)"
    )
    parser.add_argument(
        "--dbscan-min-samples",
        type=int,
        default=2,
        help="DBSCAN min_samples parameter (default: 2)"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate cluster visualization"
    )
    
    args = parser.parse_args()
    
    # Determine log files to process
    if args.log_file:
        log_files = [Path(args.log_file)]
    elif args.all:
        log_dir = Path(__file__).parent.parent / "logs" / "interventions"
        log_files = list(log_dir.glob("*.jsonl"))
    else:
        # Default: process both success logs
        log_dir = Path(__file__).parent.parent / "logs" / "interventions"
        log_files = [
            log_dir / "exploration_success.jsonl",
            log_dir / "policy_success.jsonl"
        ]
        log_files = [f for f in log_files if f.exists()]
    
    if not log_files:
        print("No log files found to process!")
        print("Run 'python src/train.py' first to generate intervention logs.")
        return
    
    print("="*60)
    print("PHASE 3: Dimensionality Reduction and Cluster Analysis")
    print("="*60)
    print(f"Method: {args.method.upper()}")
    if args.method == "kmeans":
        print(f"N clusters: {args.n_clusters}")
    else:
        print(f"DBSCAN eps: {args.dbscan_eps}, min_samples: {args.dbscan_min_samples}")
    print(f"PCA components: {args.pca_components}")
    
    # Load all intervention records
    print(f"\nLoading intervention logs...")
    all_records = []
    for log_path in log_files:
        if log_path.exists():
            records = load_intervention_logs(log_path)
            all_records.extend(records)
            print(f"  - {log_path.name}: {len(records)} records")
    
    if not all_records:
        print("No records found in log files!")
        return
    
    print(f"\nTotal records: {len(all_records)}")
    
    # Extract δf matrix
    print("\nExtracting δf vectors...")
    delta_f_matrix, valid_records = extract_delta_f_matrix(all_records)
    print(f"  Matrix shape: {delta_f_matrix.shape}")
    
    # Apply PCA
    print("\nApplying PCA...")
    reduced_matrix, pca = apply_pca(delta_f_matrix, n_components=args.pca_components)
    
    # Apply clustering
    print(f"\nApplying {args.method.upper()} clustering...")
    if args.method == "kmeans":
        cluster_labels = cluster_kmeans(reduced_matrix, n_clusters=args.n_clusters)
    else:
        cluster_labels = cluster_dbscan(
            reduced_matrix, 
            eps=args.dbscan_eps, 
            min_samples=args.dbscan_min_samples
        )
    
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    print(f"  Found {n_clusters} clusters")
    
    # Assign semantic labels to clusters
    print("\nAssigning semantic labels...")
    cluster_info = assign_cluster_labels(cluster_labels, valid_records)
    
    # Print summary
    print_cluster_summary(cluster_info, args.method)
    
    # Save results
    output_dir = Path(args.output_dir)
    output_path = output_dir / f"clusters_{args.method}.jsonl"
    save_cluster_results(
        cluster_info=cluster_info,
        records=valid_records,
        cluster_labels=cluster_labels,
        output_path=output_path
    )
    
    # Visualize if requested
    if args.visualize:
        print("\nGenerating visualization...")
        visualize_clusters(
            reduced_matrix=reduced_matrix,
            cluster_labels=cluster_labels,
            cluster_info=cluster_info,
            output_path=output_path
        )
    
    print("\n" + "="*60)
    print("PHASE 3 COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
