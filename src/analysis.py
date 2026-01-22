"""
Module: Clustering & Analysis - Discover multiple jailbreak mechanisms.

This module implements dimensionality reduction, clustering, and semantic mapping
to identify distinct jailbreak mechanisms in activation space.

Key components:
- ClusterAnalyzer: Main clustering pipeline (PCA/UMAP + K-means/DBSCAN)
- ClusterResults: Dataclass for storing cluster analysis results
- Visualization functions: Plot intervention space and cluster heatmaps
"""

from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("Warning: UMAP not available. Install with: pip install umap-learn")


@dataclass
class ClusterResults:
    """
    Results from clustering analysis.
    
    Attributes:
        cluster_assignments: Array of cluster IDs for each intervention
        cluster_centers: Mean intervention vector for each cluster
        cluster_labels: Dominant semantic label per cluster (from AO)
        cluster_confidences: Average AO confidence per cluster
        silhouette_score: Cluster quality metric [-1, 1]
        n_clusters: Number of clusters found
        reduced_embeddings: 2D/3D embeddings for visualization
        intervention_scores: Verifier scores for each intervention
    """
    cluster_assignments: np.ndarray
    cluster_centers: np.ndarray
    cluster_labels: List[str]
    cluster_confidences: List[float]
    silhouette_score: float
    n_clusters: int
    reduced_embeddings: np.ndarray
    intervention_scores: np.ndarray
    metadata: Dict


class ClusterAnalyzer:
    """
    Main clustering pipeline for intervention analysis.
    
    Pipeline:
    1. Dimensionality reduction (PCA or UMAP)
    2. Clustering (K-means or DBSCAN)
    3. Semantic labeling (via AO)
    4. Cluster statistics and visualization
    """
    
    def __init__(
        self,
        reduction_method: str = "pca",
        clustering_method: str = "kmeans",
        n_components: int = 50,
        random_state: int = 42
    ):
        """
        Initialize cluster analyzer.
        
        Args:
            reduction_method: "pca" or "umap"
            clustering_method: "kmeans" or "dbscan"
            n_components: Number of dimensions for reduction
            random_state: Random seed for reproducibility
        """
        self.reduction_method = reduction_method
        self.clustering_method = clustering_method
        self.n_components = n_components
        self.random_state = random_state
        
        # Initialize reducer
        if reduction_method == "pca":
            self.reducer = PCA(n_components=n_components, random_state=random_state)
        elif reduction_method == "umap":
            if not UMAP_AVAILABLE:
                raise ImportError("UMAP not available. Install with: pip install umap-learn")
            self.reducer = umap.UMAP(
                n_components=n_components,
                random_state=random_state,
                n_neighbors=15,
                min_dist=0.1
            )
        else:
            raise ValueError(f"Unknown reduction method: {reduction_method}")
    
    def analyze_interventions(
        self,
        interventions: List[torch.Tensor],
        ao_labels: List[str],
        ao_confidences: List[float],
        verifier_scores: List[float],
        n_clusters: Optional[int] = None,
        eps: float = 0.5,
        min_samples: int = 5
    ) -> ClusterResults:
        """
        Perform full clustering analysis on interventions.
        
        Args:
            interventions: List of intervention vectors δf
            ao_labels: Semantic labels from AO
            ao_confidences: Confidence scores from AO
            verifier_scores: Verifier scores for each intervention
            n_clusters: Number of clusters (for K-means, auto-detect if None)
            eps: DBSCAN epsilon parameter
            min_samples: DBSCAN min_samples parameter
            
        Returns:
            ClusterResults with full analysis
        """
        print(f"\n{'='*60}")
        print("CLUSTERING ANALYSIS")
        print(f"{'='*60}")
        print(f"Interventions: {len(interventions)}")
        print(f"Reduction: {self.reduction_method}")
        print(f"Clustering: {self.clustering_method}")
        
        # 1. Convert interventions to numpy array
        X = self._prepare_data(interventions)
        print(f"Data shape: {X.shape}")
        
        # 2. Dimensionality reduction
        print(f"\nReducing to {self.n_components} dimensions...")
        X_reduced = self.reducer.fit_transform(X)
        print(f"Reduced shape: {X_reduced.shape}")
        
        # 3. Clustering
        if self.clustering_method == "kmeans":
            if n_clusters is None:
                # Auto-detect optimal k using elbow method
                n_clusters = self._find_optimal_k(X_reduced, max_k=10)
            
            print(f"\nClustering with K-means (k={n_clusters})...")
            clusterer = KMeans(
                n_clusters=n_clusters,
                random_state=self.random_state,
                n_init=10
            )
            cluster_assignments = clusterer.fit_predict(X_reduced)
            
        elif self.clustering_method == "dbscan":
            print(f"\nClustering with DBSCAN (eps={eps}, min_samples={min_samples})...")
            clusterer = DBSCAN(eps=eps, min_samples=min_samples)
            cluster_assignments = clusterer.fit_predict(X_reduced)
            n_clusters = len(set(cluster_assignments)) - (1 if -1 in cluster_assignments else 0)
            print(f"Found {n_clusters} clusters")
        
        else:
            raise ValueError(f"Unknown clustering method: {self.clustering_method}")
        
        # 4. Compute cluster statistics
        print("\nComputing cluster statistics...")
        cluster_centers = self._compute_cluster_centers(X, cluster_assignments, n_clusters)
        cluster_labels, cluster_confidences = self._assign_cluster_labels(
            cluster_assignments,
            ao_labels,
            ao_confidences,
            n_clusters
        )
        
        # 5. Compute silhouette score
        if len(set(cluster_assignments)) > 1:
            sil_score = silhouette_score(X_reduced, cluster_assignments)
        else:
            sil_score = 0.0
        
        print(f"\nSilhouette score: {sil_score:.3f}")
        
        # 6. Create 2D embeddings for visualization
        print("\nCreating 2D embeddings for visualization...")
        if self.reduction_method == "pca":
            viz_reducer = PCA(n_components=2, random_state=self.random_state)
        else:
            viz_reducer = umap.UMAP(n_components=2, random_state=self.random_state)
        
        embeddings_2d = viz_reducer.fit_transform(X)
        
        # 7. Print cluster summary
        self._print_cluster_summary(
            cluster_assignments,
            cluster_labels,
            cluster_confidences,
            verifier_scores,
            n_clusters
        )
        
        return ClusterResults(
            cluster_assignments=cluster_assignments,
            cluster_centers=cluster_centers,
            cluster_labels=cluster_labels,
            cluster_confidences=cluster_confidences,
            silhouette_score=sil_score,
            n_clusters=n_clusters,
            reduced_embeddings=embeddings_2d,
            intervention_scores=np.array(verifier_scores),
            metadata={
                'reduction_method': self.reduction_method,
                'clustering_method': self.clustering_method,
                'n_components': self.n_components,
            }
        )
    
    def _prepare_data(self, interventions: List[torch.Tensor]) -> np.ndarray:
        """Convert intervention tensors to numpy array."""
        # Convert to numpy and flatten if needed
        arrays = []
        for interv in interventions:
            if isinstance(interv, torch.Tensor):
                arr = interv.cpu().numpy()
            else:
                arr = np.array(interv)
            
            # Flatten if multi-dimensional
            if arr.ndim > 1:
                arr = arr.flatten()
            
            arrays.append(arr)
        
        return np.vstack(arrays)
    
    def _find_optimal_k(self, X: np.ndarray, max_k: int = 10) -> int:
        """Find optimal number of clusters using elbow method."""
        inertias = []
        K_range = range(2, min(max_k + 1, len(X)))
        
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            kmeans.fit(X)
            inertias.append(kmeans.inertia_)
        
        # Simple elbow detection: find point of maximum curvature
        if len(inertias) < 2:
            return 2
        
        # Compute second derivative
        diffs = np.diff(inertias)
        second_diffs = np.diff(diffs)
        
        # Find elbow (maximum second derivative)
        elbow_idx = np.argmax(second_diffs) + 2  # +2 because of double diff
        optimal_k = list(K_range)[min(elbow_idx, len(K_range) - 1)]
        
        print(f"  Auto-detected optimal k: {optimal_k}")
        return optimal_k
    
    def _compute_cluster_centers(
        self,
        X: np.ndarray,
        assignments: np.ndarray,
        n_clusters: int
    ) -> np.ndarray:
        """Compute mean intervention for each cluster."""
        centers = []
        for i in range(n_clusters):
            cluster_mask = assignments == i
            if cluster_mask.sum() > 0:
                center = X[cluster_mask].mean(axis=0)
                centers.append(center)
            else:
                # Empty cluster
                centers.append(np.zeros(X.shape[1]))
        
        return np.vstack(centers)
    
    def _assign_cluster_labels(
        self,
        assignments: np.ndarray,
        ao_labels: List[str],
        ao_confidences: List[float],
        n_clusters: int
    ) -> Tuple[List[str], List[float]]:
        """Assign dominant AO label to each cluster."""
        cluster_labels = []
        cluster_confidences = []
        
        for i in range(n_clusters):
            cluster_mask = assignments == i
            
            if cluster_mask.sum() == 0:
                cluster_labels.append("Empty")
                cluster_confidences.append(0.0)
                continue
            
            # Get labels and confidences for this cluster
            cluster_ao_labels = [ao_labels[j] for j in range(len(ao_labels)) if cluster_mask[j]]
            cluster_ao_confs = [ao_confidences[j] for j in range(len(ao_confidences)) if cluster_mask[j]]
            
            # Find most common label
            from collections import Counter
            label_counts = Counter(cluster_ao_labels)
            dominant_label = label_counts.most_common(1)[0][0]
            
            # Average confidence for dominant label
            dominant_confs = [
                conf for label, conf in zip(cluster_ao_labels, cluster_ao_confs)
                if label == dominant_label
            ]
            avg_conf = np.mean(dominant_confs) if dominant_confs else 0.0
            
            cluster_labels.append(dominant_label)
            cluster_confidences.append(avg_conf)
        
        return cluster_labels, cluster_confidences
    
    def _print_cluster_summary(
        self,
        assignments: np.ndarray,
        labels: List[str],
        confidences: List[float],
        scores: List[float],
        n_clusters: int
    ):
        """Print summary statistics for each cluster."""
        print(f"\n{'='*60}")
        print("CLUSTER SUMMARY")
        print(f"{'='*60}")
        
        scores_arr = np.array(scores)
        
        for i in range(n_clusters):
            cluster_mask = assignments == i
            n_members = cluster_mask.sum()
            
            if n_members == 0:
                continue
            
            cluster_scores = scores_arr[cluster_mask]
            avg_score = cluster_scores.mean()
            
            print(f"\nCluster {i}: {labels[i]}")
            print(f"  Members: {n_members}")
            print(f"  AO Confidence: {confidences[i]:.2f}")
            print(f"  Avg Verifier Score: {avg_score:.2f}")


def plot_intervention_space(
    results: ClusterResults,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8)
):
    """
    Plot 2D visualization of intervention space with cluster colors.
    
    Args:
        results: ClusterResults from analyze_interventions()
        save_path: Optional path to save figure
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    
    # Create scatter plot
    scatter = plt.scatter(
        results.reduced_embeddings[:, 0],
        results.reduced_embeddings[:, 1],
        c=results.cluster_assignments,
        cmap='tab10',
        s=100,
        alpha=0.6,
        edgecolors='black',
        linewidth=0.5
    )
    
    # Add cluster labels
    for i in range(results.n_clusters):
        cluster_mask = results.cluster_assignments == i
        if cluster_mask.sum() == 0:
            continue
        
        # Compute cluster center in 2D space
        center_x = results.reduced_embeddings[cluster_mask, 0].mean()
        center_y = results.reduced_embeddings[cluster_mask, 1].mean()
        
        # Add label
        plt.annotate(
            results.cluster_labels[i],
            (center_x, center_y),
            fontsize=12,
            fontweight='bold',
            ha='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.7)
        )
    
    plt.colorbar(scatter, label='Cluster ID')
    plt.xlabel('Dimension 1', fontsize=12)
    plt.ylabel('Dimension 2', fontsize=12)
    plt.title(
        f'Intervention Space Clustering\n'
        f'Method: {results.metadata["clustering_method"]}, '
        f'Silhouette: {results.silhouette_score:.3f}',
        fontsize=14
    )
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    plt.show()


def plot_cluster_heatmap(
    results: ClusterResults,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 8)
):
    """
    Plot heatmap of intervention magnitudes across clusters.
    
    Args:
        results: ClusterResults from analyze_interventions()
        save_path: Optional path to save figure
        figsize: Figure size
    """
    # Take first 100 dimensions for visualization
    max_dims = min(100, results.cluster_centers.shape[1])
    centers_subset = results.cluster_centers[:, :max_dims]
    
    plt.figure(figsize=figsize)
    
    # Create heatmap
    sns.heatmap(
        centers_subset,
        cmap='RdBu_r',
        center=0,
        yticklabels=[f"{i}: {label}" for i, label in enumerate(results.cluster_labels)],
        cbar_kws={'label': 'Intervention Magnitude'},
        linewidths=0.5
    )
    
    plt.xlabel('Activation Dimension', fontsize=12)
    plt.ylabel('Cluster', fontsize=12)
    plt.title(
        f'Cluster Intervention Patterns\n'
        f'(First {max_dims} dimensions)',
        fontsize=14
    )
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved heatmap to {save_path}")
    
    plt.show()


if __name__ == "__main__":
    print("Clustering & Analysis Module")
    print("\nThis module provides clustering analysis for intervention discovery.")
    print("\nUsage:")
    print("  1. Collect successful interventions from policy training")
    print("  2. Label interventions with AO")
    print("  3. Run ClusterAnalyzer.analyze_interventions()")
    print("  4. Visualize with plot_intervention_space() and plot_cluster_heatmap()")
