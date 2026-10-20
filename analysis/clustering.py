"""
===============================================================================
EPL Insight Engine — Clustering Engine
===============================================================================
Implements K-Means clustering on TF-IDF vectors, selecting K by
Silhouette Score and recording inertia for the Elbow diagnostic.

Algorithm: K-Means Clustering
    - Objective: Partition N documents into K clusters where each
      document belongs to the cluster with the nearest centroid.
    - Optimization: Minimizes within-cluster sum of squares (inertia).

Cluster Quality Metrics:
    - Elbow Method: Plot inertia vs. K, look for the "elbow" point.
    - Silhouette Score: Measures how similar a document is to its own
      cluster vs. other clusters. Range: [-1, 1], higher is better.

Author  : EPL Insight Engine Team
===============================================================================
"""

import os
import logging

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

from config import CLUSTER_CONFIG, ANALYSIS_RESULTS_DIR

logger = logging.getLogger(__name__)


class ClusteringEngine:
    """
    K-Means clustering engine for EPL article analysis.

    This class:
        1. Selects the number of clusters (K) by Silhouette Score and records
           inertia values for the Elbow diagnostic.
        2. Runs K-Means clustering on TF-IDF vectors.
        3. Extracts top keywords per cluster for labeling.
        4. Reduces dimensionality via PCA for 2D visualization.

    Attributes:
        config (dict): Clustering hyperparameters from config.py.
        kmeans (KMeans): Fitted K-Means model.
        labels (np.ndarray): Cluster labels for each document.
        optimal_k (int): Optimal number of clusters.
        k_values (list): Cluster counts evaluated during optimal K search.
        inertias (list): Inertia values for each K (for elbow plot).
        silhouette_scores (list): Silhouette scores for each K.
        cluster_keywords (dict): Top keywords per cluster.
        pca_coordinates (np.ndarray): 2D PCA projection of TF-IDF.

    Example:
        >>> engine = ClusteringEngine()
        >>> optimal_k = engine.find_optimal_k(tfidf_matrix)
        >>> labels = engine.fit(tfidf_matrix, k=optimal_k)
        >>> keywords = engine.get_cluster_keywords(tfidf_vectorizer)
    """

    def __init__(self):
        """Initialize the clustering engine with configuration."""
        self.config = CLUSTER_CONFIG
        self.kmeans = None
        self.labels = None
        self.optimal_k = None
        self.k_values = []
        self.inertias = []
        self.silhouette_scores = []
        self.cluster_keywords = {}
        self.pca_coordinates = None

        logger.info(
            f"ClusteringEngine initialized. "
            f"K range: [{self.config['min_k']}, {self.config['max_k']}]"
        )

    # =========================================================================
    # 1. FIND OPTIMAL K (Silhouette Score)
    # =========================================================================
    def find_optimal_k(self, tfidf_matrix):
        """
        Find the optimal number of clusters using the Silhouette Score.
        The Elbow Method metrics are recorded for visualization.

        Elbow Method:
            Plots inertia (within-cluster sum of squares) against K.
            The "elbow" point where inertia starts decreasing more
            slowly suggests the optimal K.

        Silhouette Score:
            For each K, computes the average silhouette coefficient.
            Higher silhouette = better-defined clusters.
            The K with the highest silhouette score is selected.

        Args:
            tfidf_matrix (sparse matrix): TF-IDF document-term matrix.

        Returns:
            int: Optimal number of clusters.
        """
        logger.info("Finding optimal K using silhouette scores...")

        n_documents = tfidf_matrix.shape[0]
        if n_documents < 2:
            raise ValueError("At least 2 documents are required for clustering.")

        max_k = min(
            self.config["max_k"],
            n_documents if n_documents == 2 else n_documents - 1,
        )
        min_k = min(self.config["min_k"], max_k)
        self.k_values = list(range(min_k, max_k + 1))
        self.inertias = []
        self.silhouette_scores = []
        valid_scores = []

        for k in self.k_values:
            logger.info(f"  Testing K={k}...")

            # Run K-Means
            km = KMeans(
                n_clusters=k,
                random_state=self.config["random_state"],
                max_iter=self.config["max_iterations"],
                n_init=self.config["n_init"],
            )
            km.fit(tfidf_matrix)

            # Record inertia
            self.inertias.append(km.inertia_)

            # Silhouette scoring requires between 2 and n_documents - 1 labels.
            n_labels = np.unique(km.labels_).size
            if 2 <= n_labels < n_documents:
                sil_score = silhouette_score(
                    tfidf_matrix,
                    km.labels_,
                    sample_size=min(1000, n_documents),
                    random_state=self.config["random_state"],
                )
                self.silhouette_scores.append(sil_score)
                valid_scores.append((k, sil_score))
                logger.info(
                    f"    Inertia: {km.inertia_:.2f} | "
                    f"Silhouette: {sil_score:.4f}"
                )
            else:
                self.silhouette_scores.append(np.nan)
                logger.info(
                    f"    Inertia: {km.inertia_:.2f} | "
                    "Silhouette: unavailable for this cluster assignment"
                )

        if valid_scores:
            self.optimal_k, best_score = max(valid_scores, key=lambda item: item[1])
            logger.info(
                f"\n  ✅ Optimal K = {self.optimal_k} "
                f"(Silhouette: {best_score:.4f})"
            )
        else:
            self.optimal_k = min(
                max(self.config["default_k"], min_k),
                max_k,
            )
            logger.warning(
                "No valid silhouette score; using fallback "
                f"K={self.optimal_k}."
            )

        return self.optimal_k

    # =========================================================================
    # 2. FIT K-MEANS CLUSTERING
    # =========================================================================
    def fit(self, tfidf_matrix, k=None):
        """
        Fit K-Means clustering to the TF-IDF matrix.

        Args:
            tfidf_matrix (sparse matrix): TF-IDF document-term matrix.
            k (int, optional): Number of clusters. If None, uses
                               optimal_k or default_k.

        Returns:
            np.ndarray: Cluster label for each document.
        """
        if k is None:
            k = self.optimal_k or self.config["default_k"]

        n_documents = tfidf_matrix.shape[0]
        if not 1 <= k <= n_documents:
            raise ValueError(
                f"Number of clusters must be between 1 and {n_documents}; got {k}."
            )

        logger.info(f"Fitting K-Means with K={k}...")

        self.kmeans = KMeans(
            n_clusters=k,
            random_state=self.config["random_state"],
            max_iter=self.config["max_iterations"],
            n_init=self.config["n_init"],
        )
        self.labels = self.kmeans.fit_predict(tfidf_matrix)

        # Log cluster sizes
        unique, counts = np.unique(self.labels, return_counts=True)
        logger.info("Cluster sizes:")
        for cluster_id, count in zip(unique, counts):
            logger.info(f"  Cluster {cluster_id}: {count} documents")

        return self.labels

    # =========================================================================
    # 3. EXTRACT CLUSTER KEYWORDS
    # =========================================================================
    def get_cluster_keywords(self, tfidf_vectorizer, n_terms=None):
        """
        Extract the top keywords for each cluster by examining
        the cluster centroids in TF-IDF space.

        The centroid of a cluster in TF-IDF space has a high value
        for terms that are characteristic of that cluster's documents.

        Args:
            tfidf_vectorizer (TfidfVectorizer): Fitted TF-IDF vectorizer
                                                 (needed for feature names).
            n_terms (int, optional): Number of top terms per cluster.

        Returns:
            dict: {cluster_id: [(term, weight), ...], ...}
        """
        if self.kmeans is None:
            logger.error("K-Means not fitted yet. Call fit() first.")
            return {}

        n_terms = n_terms or self.config["top_terms_per_cluster"]
        feature_names = tfidf_vectorizer.get_feature_names_out()
        centroids = self.kmeans.cluster_centers_

        self.cluster_keywords = {}

        logger.info(f"\nCluster Keywords (top {n_terms} per cluster):")
        for cluster_id in range(len(centroids)):
            # Get indices of top terms by centroid weight
            top_indices = centroids[cluster_id].argsort()[-n_terms:][::-1]
            top_terms = [
                (feature_names[i], round(centroids[cluster_id][i], 4))
                for i in top_indices
            ]

            self.cluster_keywords[cluster_id] = top_terms

            # Log the keywords
            terms_str = ", ".join([t[0] for t in top_terms[:5]])
            logger.info(f"  Cluster {cluster_id}: {terms_str}...")

        return self.cluster_keywords

    # =========================================================================
    # 4. PCA DIMENSIONALITY REDUCTION
    # =========================================================================
    def reduce_dimensions(self, tfidf_matrix, n_components=2):
        """
        Reduce TF-IDF matrix to 2D using PCA for visualization.

        PCA (Principal Component Analysis) projects high-dimensional
        TF-IDF vectors into 2D space while preserving as much
        variance as possible.

        Args:
            tfidf_matrix (sparse matrix): TF-IDF document-term matrix.
            n_components (int): Target dimensionality (2 for scatter plot).

        Returns:
            np.ndarray: 2D coordinates (shape: n_documents × 2).
        """
        logger.info(f"Reducing dimensions to {n_components}D via PCA...")

        n_documents, n_features = tfidf_matrix.shape
        if n_documents < 2 or n_features < 1:
            raise ValueError(
                "PCA requires at least 2 documents and 1 feature."
            )

        fitted_components = min(n_components, n_documents, n_features)
        pca = PCA(
            n_components=fitted_components,
            random_state=self.config["random_state"],
        )
        coordinates = pca.fit_transform(tfidf_matrix.toarray())
        if fitted_components < n_components:
            coordinates = np.pad(
                coordinates,
                ((0, 0), (0, n_components - fitted_components)),
            )
        self.pca_coordinates = coordinates

        # Log explained variance
        explained_var = pca.explained_variance_ratio_
        variance_summary = ", ".join(
            f"PC{index + 1}={variance:.3f}"
            for index, variance in enumerate(explained_var)
        )
        logger.info(f"  PCA explained variance: {variance_summary}")
        logger.info(f"  Total explained variance: {sum(explained_var):.3f}")

        return self.pca_coordinates

    # =========================================================================
    # 5. SAVE CLUSTERING RESULTS
    # =========================================================================
    def save_results(self, df):
        """
        Save clustering results to CSV.

        Args:
            df (pd.DataFrame): DataFrame with cluster labels added.

        Returns:
            str: Path to saved file.
        """
        os.makedirs(ANALYSIS_RESULTS_DIR, exist_ok=True)

        # Save cluster assignments
        cluster_path = os.path.join(ANALYSIS_RESULTS_DIR, "cluster_assignments.csv")
        cluster_cols = ["title", "source", "cluster"]
        existing_cols = [c for c in cluster_cols if c in df.columns]
        df[existing_cols].to_csv(cluster_path, index=False, encoding="utf-8")
        logger.info(f"Saved cluster assignments to {cluster_path}")

        # Save cluster keywords
        if self.cluster_keywords:
            keywords_path = os.path.join(ANALYSIS_RESULTS_DIR, "cluster_keywords.csv")
            rows = []
            for cluster_id, keywords in self.cluster_keywords.items():
                for term, weight in keywords:
                    rows.append({
                        "cluster": cluster_id,
                        "keyword": term,
                        "weight": weight,
                    })
            kw_df = pd.DataFrame(rows)
            kw_df.to_csv(keywords_path, index=False)
            logger.info(f"Saved cluster keywords to {keywords_path}")

        return cluster_path

    def print_summary(self):
        """Print a formatted summary of clustering results."""
        print("\n" + "=" * 70)
        print("🔬  CLUSTERING SUMMARY")
        print("=" * 70)

        if self.optimal_k:
            print(f"\n✅ Optimal K: {self.optimal_k}")

        if self.labels is not None:
            unique, counts = np.unique(self.labels, return_counts=True)
            print(f"\n📊 Cluster Distribution:")
            for cluster_id, count in zip(unique, counts):
                bar = "█" * min(count, 30)
                print(f"   Cluster {cluster_id}: {count:4d} articles {bar}")

        if self.cluster_keywords:
            print(f"\n🔤 Cluster Keywords:")
            for cluster_id, keywords in self.cluster_keywords.items():
                terms = ", ".join([f"{t}" for t, _ in keywords[:7]])
                print(f"   Cluster {cluster_id}: {terms}")

        print("\n" + "=" * 70)
