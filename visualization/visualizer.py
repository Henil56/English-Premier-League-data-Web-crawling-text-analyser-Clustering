"""
===============================================================================
EPL Insight Engine — Visualizer
===============================================================================
Generates all visualizations for the analysis pipeline:

    1. Word Cloud (overall + per cluster)
    2. Cluster Scatter Plot (PCA-reduced 2D)
    3. Elbow Curve (K vs. Inertia)
    4. Silhouette Score Plot (K vs. Silhouette)
    5. Sentiment Distribution (bar chart)
    6. Source Comparison (article counts, sentiment by source)
    7. Top TF-IDF Terms (horizontal bar chart)
    8. Word Frequency Bar Chart

Libraries:
    - matplotlib: Core plotting
    - seaborn: Statistical visualizations
    - wordcloud: Word cloud generation

Author  : EPL Insight Engine Team
===============================================================================
"""

import os
import logging

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

from config import VIZ_CONFIG, VISUALIZATIONS_DIR

logger = logging.getLogger(__name__)

# Apply global style
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette(VIZ_CONFIG["color_palette"])


class Visualizer:
    """
    Visualization engine for EPL Insight Engine.

    Generates publication-quality charts and saves them as PNG files.
    All plots use a consistent style with the configured color palette.

    Attributes:
        output_dir (str): Directory to save generated plots.

    Example:
        >>> viz = Visualizer()
        >>> viz.plot_cluster_scatter(pca_coords, labels)
        >>> viz.plot_wordcloud(cleaned_texts)
    """

    def __init__(self):
        """Initialize the visualizer and create output directory."""
        self.output_dir = VISUALIZATIONS_DIR
        os.makedirs(self.output_dir, exist_ok=True)
        logger.info(f"Visualizer initialized. Output: {self.output_dir}")

    def _save_figure(self, fig, filename):
        """
        Save a matplotlib figure to disk.

        Args:
            fig (matplotlib.figure.Figure): Figure to save.
            filename (str): Output filename (without path).

        Returns:
            str: Full path to saved file.
        """
        filepath = os.path.join(self.output_dir, filename)
        fig.savefig(
            filepath,
            dpi=VIZ_CONFIG["figure_dpi"],
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
        )
        plt.close(fig)
        logger.info(f"  📊 Saved: {filepath}")
        return filepath

    # =========================================================================
    # 1. WORD CLOUD
    # =========================================================================
    def plot_wordcloud(self, cleaned_texts, title="EPL Articles — Word Cloud",
                       filename="wordcloud_overall.png"):
        """
        Generate a word cloud from preprocessed text.

        Word clouds display words sized proportionally to their
        frequency in the corpus. Larger words = more frequent.

        Args:
            cleaned_texts (list): List of preprocessed text strings.
            title (str): Plot title.
            filename (str): Output filename.

        Returns:
            str: Path to saved figure.
        """
        logger.info("Generating word cloud...")

        # Combine all texts
        all_text = " ".join(cleaned_texts)

        if not all_text.strip():
            logger.warning("No text for word cloud.")
            return None

        # Generate word cloud
        wc = WordCloud(
            width=VIZ_CONFIG["wordcloud_width"],
            height=VIZ_CONFIG["wordcloud_height"],
            max_words=VIZ_CONFIG["wordcloud_max_words"],
            background_color="white",
            colormap="viridis",
            collocations=False,
            contour_width=2,
            contour_color="steelblue",
        ).generate(all_text)

        # Plot
        fig, ax = plt.subplots(1, 1, figsize=(16, 8))
        ax.imshow(wc, interpolation="bilinear")
        ax.set_title(title, fontsize=18, fontweight="bold", pad=15)
        ax.axis("off")

        return self._save_figure(fig, filename)

    def plot_cluster_wordclouds(self, cleaned_texts, labels, n_clusters):
        """
        Generate a word cloud for each cluster.

        Args:
            cleaned_texts (list): Preprocessed texts.
            labels (np.ndarray): Cluster labels.
            n_clusters (int): Number of clusters.

        Returns:
            str: Path to saved figure.
        """
        logger.info("Generating per-cluster word clouds...")

        # Determine grid layout
        cols = min(3, n_clusters)
        rows = (n_clusters + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(8 * cols, 6 * rows))

        if n_clusters == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        colors = ["viridis", "plasma", "inferno", "magma", "cividis",
                   "twilight", "coolwarm", "RdYlGn", "Spectral", "Set3"]

        for cluster_id in range(n_clusters):
            # Get texts for this cluster
            cluster_texts = [
                cleaned_texts[i]
                for i in range(len(labels))
                if labels[i] == cluster_id
            ]
            combined = " ".join(cluster_texts)

            if not combined.strip():
                axes[cluster_id].text(
                    0.5, 0.5, "No data", ha="center", va="center", fontsize=14
                )
                axes[cluster_id].set_title(f"Cluster {cluster_id}")
                axes[cluster_id].axis("off")
                continue

            wc = WordCloud(
                width=800, height=400,
                max_words=60,
                background_color="white",
                colormap=colors[cluster_id % len(colors)],
                collocations=False,
            ).generate(combined)

            axes[cluster_id].imshow(wc, interpolation="bilinear")
            axes[cluster_id].set_title(
                f"Cluster {cluster_id} ({len(cluster_texts)} articles)",
                fontsize=14, fontweight="bold",
            )
            axes[cluster_id].axis("off")

        # Hide unused subplots
        for i in range(n_clusters, len(axes)):
            axes[i].axis("off")

        fig.suptitle(
            "Word Clouds by Cluster", fontsize=20, fontweight="bold", y=1.02
        )
        plt.tight_layout()

        return self._save_figure(fig, "wordclouds_by_cluster.png")

    # =========================================================================
    # 2. CLUSTER SCATTER PLOT (PCA)
    # =========================================================================
    def plot_cluster_scatter(self, pca_coordinates, labels, title="K-Means Clusters (PCA 2D Projection)"):
        """
        Plot clusters in 2D PCA-reduced space.

        Each document is represented as a point, colored by its
        cluster assignment. Cluster centroids are marked with X.

        Args:
            pca_coordinates (np.ndarray): 2D coordinates (N × 2).
            labels (np.ndarray): Cluster labels.
            title (str): Plot title.

        Returns:
            str: Path to saved figure.
        """
        logger.info("Generating cluster scatter plot...")

        fig, ax = plt.subplots(1, 1, figsize=(12, 8))

        n_clusters = len(set(labels))
        palette = sns.color_palette(VIZ_CONFIG["color_palette"], n_clusters)

        # Plot each cluster
        for cluster_id in range(n_clusters):
            mask = labels == cluster_id
            ax.scatter(
                pca_coordinates[mask, 0],
                pca_coordinates[mask, 1],
                c=[palette[cluster_id]],
                label=f"Cluster {cluster_id} (n={mask.sum()})",
                alpha=0.7,
                s=80,
                edgecolors="white",
                linewidth=0.5,
            )

            # Plot centroid
            centroid_x = pca_coordinates[mask, 0].mean()
            centroid_y = pca_coordinates[mask, 1].mean()
            ax.scatter(
                centroid_x, centroid_y,
                c=[palette[cluster_id]],
                marker="X", s=200,
                edgecolors="black", linewidth=2,
                zorder=5,
            )

        ax.set_xlabel("Principal Component 1", fontsize=12)
        ax.set_ylabel("Principal Component 2", fontsize=12)
        ax.set_title(title, fontsize=16, fontweight="bold", pad=15)
        ax.legend(fontsize=10, loc="best", framealpha=0.9)
        ax.grid(True, alpha=0.3)

        return self._save_figure(fig, "cluster_scatter_pca.png")

    # =========================================================================
    # 3. ELBOW CURVE
    # =========================================================================
    def plot_elbow_curve(self, k_range, inertias, optimal_k=None):
        """
        Plot the Elbow Curve: Inertia (WCSS) vs. Number of Clusters (K).

        The "elbow" point — where the rate of decrease sharply
        changes — suggests the optimal K.

        Args:
            k_range (range): Range of K values tested.
            inertias (list): Inertia (WCSS) for each K.
            optimal_k (int, optional): Highlight the chosen K.

        Returns:
            str: Path to saved figure.
        """
        logger.info("Generating elbow curve...")

        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        ax.plot(
            list(k_range), inertias,
            marker="o", linewidth=2, markersize=8,
            color="#2196F3", markerfacecolor="#1565C0",
        )

        if optimal_k:
            idx = list(k_range).index(optimal_k)
            ax.axvline(
                x=optimal_k, color="#FF5722", linestyle="--",
                linewidth=2, alpha=0.7, label=f"Optimal K = {optimal_k}",
            )
            ax.scatter(
                [optimal_k], [inertias[idx]],
                c="#FF5722", s=150, zorder=5,
                marker="*", edgecolors="black",
            )

        ax.set_xlabel("Number of Clusters (K)", fontsize=12)
        ax.set_ylabel("Inertia (Within-Cluster Sum of Squares)", fontsize=12)
        ax.set_title("Elbow Method — Optimal K Selection", fontsize=16, fontweight="bold")
        ax.set_xticks(list(k_range))
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        return self._save_figure(fig, "elbow_curve.png")

    # =========================================================================
    # 4. SILHOUETTE SCORE PLOT
    # =========================================================================
    def plot_silhouette_scores(self, k_range, silhouette_scores, optimal_k=None):
        """
        Plot Silhouette Scores vs. K.

        Higher silhouette score = better cluster separation.

        Args:
            k_range (range): Range of K values.
            silhouette_scores (list): Silhouette scores.
            optimal_k (int, optional): Highlight optimal K.

        Returns:
            str: Path to saved figure.
        """
        logger.info("Generating silhouette score plot...")

        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        ax.plot(
            list(k_range), silhouette_scores,
            marker="s", linewidth=2, markersize=8,
            color="#4CAF50", markerfacecolor="#2E7D32",
        )

        if optimal_k and optimal_k in list(k_range):
            idx = list(k_range).index(optimal_k)
            ax.axvline(
                x=optimal_k, color="#FF9800", linestyle="--",
                linewidth=2, alpha=0.7, label=f"Best K = {optimal_k}",
            )
            ax.scatter(
                [optimal_k], [silhouette_scores[idx]],
                c="#FF9800", s=150, zorder=5,
                marker="*", edgecolors="black",
            )

        ax.set_xlabel("Number of Clusters (K)", fontsize=12)
        ax.set_ylabel("Silhouette Score", fontsize=12)
        ax.set_title(
            "Silhouette Score — Cluster Quality", fontsize=16, fontweight="bold"
        )
        ax.set_xticks(list(k_range))
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        return self._save_figure(fig, "silhouette_scores.png")

    # =========================================================================
    # 5. SENTIMENT DISTRIBUTION
    # =========================================================================
    def plot_sentiment_distribution(self, df):
        """
        Plot sentiment distribution as a bar chart and histogram.

        Creates a two-panel figure:
            Left:  Bar chart of sentiment label counts
            Right: Histogram of compound sentiment scores

        Args:
            df (pd.DataFrame): DataFrame with sentiment columns.

        Returns:
            str: Path to saved figure.
        """
        logger.info("Generating sentiment distribution plots...")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # --- Left: Sentiment Label Counts ---
        colors_map = {"Positive": "#4CAF50", "Neutral": "#FFC107", "Negative": "#F44336"}
        label_counts = df["sentiment_label"].value_counts()

        bars = ax1.bar(
            label_counts.index, label_counts.values,
            color=[colors_map.get(l, "#999") for l in label_counts.index],
            edgecolor="white", linewidth=1.5,
        )

        # Add value labels on bars
        for bar, val in zip(bars, label_counts.values):
            ax1.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                str(val), ha="center", va="bottom", fontweight="bold", fontsize=12,
            )

        ax1.set_title("Sentiment Distribution", fontsize=14, fontweight="bold")
        ax1.set_ylabel("Number of Articles", fontsize=11)
        ax1.set_xlabel("Sentiment", fontsize=11)

        # --- Right: Compound Score Histogram ---
        ax2.hist(
            df["sentiment_compound"], bins=20,
            color="#42A5F5", edgecolor="white", alpha=0.8,
        )
        ax2.axvline(x=0, color="red", linestyle="--", linewidth=1.5, alpha=0.6)
        ax2.axvline(
            x=df["sentiment_compound"].mean(),
            color="#FF9800", linestyle="-", linewidth=2,
            label=f"Mean: {df['sentiment_compound'].mean():.3f}",
        )
        ax2.set_title("Compound Sentiment Score Distribution", fontsize=14, fontweight="bold")
        ax2.set_xlabel("Compound Score", fontsize=11)
        ax2.set_ylabel("Frequency", fontsize=11)
        ax2.legend(fontsize=10)

        plt.tight_layout()
        return self._save_figure(fig, "sentiment_distribution.png")

    # =========================================================================
    # 6. SOURCE COMPARISON
    # =========================================================================
    def plot_source_comparison(self, df):
        """
        Compare data across sources (BBC, Sky, EPL).

        Creates a two-panel figure:
            Left:  Article count by source
            Right: Average sentiment by source

        Args:
            df (pd.DataFrame): DataFrame with 'source' and sentiment columns.

        Returns:
            str: Path to saved figure.
        """
        logger.info("Generating source comparison plots...")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        palette = sns.color_palette(VIZ_CONFIG["color_palette"])

        # --- Left: Article Count by Source ---
        source_counts = df["source"].value_counts()
        bars = ax1.bar(
            source_counts.index, source_counts.values,
            color=palette[:len(source_counts)],
            edgecolor="white", linewidth=1.5,
        )
        for bar, val in zip(bars, source_counts.values):
            ax1.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                str(val), ha="center", va="bottom", fontweight="bold", fontsize=12,
            )
        ax1.set_title("Articles by Source", fontsize=14, fontweight="bold")
        ax1.set_ylabel("Count", fontsize=11)
        ax1.tick_params(axis="x", rotation=15)

        # --- Right: Avg Sentiment by Source ---
        if "sentiment_compound" in df.columns:
            avg_sentiment = df.groupby("source")["sentiment_compound"].mean()
            bar_colors = ["#4CAF50" if v >= 0 else "#F44336" for v in avg_sentiment.values]
            bars2 = ax2.bar(
                avg_sentiment.index, avg_sentiment.values,
                color=bar_colors, edgecolor="white", linewidth=1.5,
            )
            for bar, val in zip(bars2, avg_sentiment.values):
                ax2.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005 if val >= 0 else bar.get_height() - 0.02,
                    f"{val:.3f}", ha="center", fontweight="bold", fontsize=11,
                )
            ax2.axhline(y=0, color="gray", linestyle="-", linewidth=0.8)
            ax2.set_title("Avg Sentiment by Source", fontsize=14, fontweight="bold")
            ax2.set_ylabel("Avg Compound Score", fontsize=11)
            ax2.tick_params(axis="x", rotation=15)

        plt.tight_layout()
        return self._save_figure(fig, "source_comparison.png")

    # =========================================================================
    # 7. TOP TF-IDF TERMS
    # =========================================================================
    def plot_top_tfidf_terms(self, top_terms, n=20):
        """
        Horizontal bar chart of top TF-IDF terms.

        Args:
            top_terms (list): List of (term, score) tuples.
            n (int): Number of terms to display.

        Returns:
            str: Path to saved figure.
        """
        logger.info("Generating TF-IDF terms chart...")

        terms = [t[0] for t in top_terms[:n]][::-1]
        scores = [t[1] for t in top_terms[:n]][::-1]

        fig, ax = plt.subplots(1, 1, figsize=(10, 8))

        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(terms)))
        ax.barh(terms, scores, color=colors, edgecolor="white", height=0.7)

        ax.set_xlabel("Average TF-IDF Score", fontsize=12)
        ax.set_title("Top TF-IDF Terms Across Corpus", fontsize=16, fontweight="bold")
        ax.grid(axis="x", alpha=0.3)

        plt.tight_layout()
        return self._save_figure(fig, "top_tfidf_terms.png")

    # =========================================================================
    # 8. WORD FREQUENCY BAR CHART
    # =========================================================================
    def plot_word_frequencies(self, word_freqs, n=20, title="Top Word Frequencies"):
        """
        Bar chart of most frequent words.

        Args:
            word_freqs (list): List of (word, count) tuples.
            n (int): Number of words to display.
            title (str): Plot title.

        Returns:
            str: Path to saved figure.
        """
        logger.info("Generating word frequency chart...")

        words = [w[0] for w in word_freqs[:n]][::-1]
        counts = [w[1] for w in word_freqs[:n]][::-1]

        fig, ax = plt.subplots(1, 1, figsize=(10, 8))

        colors = plt.cm.plasma(np.linspace(0.3, 0.9, len(words)))
        ax.barh(words, counts, color=colors, edgecolor="white", height=0.7)

        ax.set_xlabel("Frequency", fontsize=12)
        ax.set_title(title, fontsize=16, fontweight="bold")
        ax.grid(axis="x", alpha=0.3)

        plt.tight_layout()
        return self._save_figure(fig, "word_frequencies.png")

    # =========================================================================
    # 9. GENERATE ALL VISUALIZATIONS
    # =========================================================================
    def generate_all(self, df, cleaned_texts, clustering_engine, text_analyzer):
        """
        Generate all visualizations in one call.

        Args:
            df (pd.DataFrame): Full analyzed DataFrame.
            cleaned_texts (list): Preprocessed texts.
            clustering_engine (ClusteringEngine): Fitted clustering engine.
            text_analyzer (TextAnalyzer): Completed text analyzer.

        Returns:
            dict: Mapping of plot name → file path.
        """
        logger.info("\n📊 Generating all visualizations...")
        saved = {}

        # 1. Overall word cloud
        path = self.plot_wordcloud(cleaned_texts)
        if path:
            saved["wordcloud"] = path

        # 2. Cluster word clouds
        if clustering_engine.labels is not None:
            path = self.plot_cluster_wordclouds(
                cleaned_texts, clustering_engine.labels,
                len(set(clustering_engine.labels)),
            )
            if path:
                saved["cluster_wordclouds"] = path

        # 3. Cluster scatter plot
        if clustering_engine.pca_coordinates is not None:
            path = self.plot_cluster_scatter(
                clustering_engine.pca_coordinates, clustering_engine.labels,
            )
            saved["cluster_scatter"] = path

        # 4. Elbow curve
        if clustering_engine.inertias:
            path = self.plot_elbow_curve(
                clustering_engine.k_values,
                clustering_engine.inertias,
                clustering_engine.optimal_k,
            )
            saved["elbow_curve"] = path

        # 5. Silhouette scores
        if clustering_engine.silhouette_scores:
            path = self.plot_silhouette_scores(
                clustering_engine.k_values,
                clustering_engine.silhouette_scores,
                clustering_engine.optimal_k,
            )
            saved["silhouette_scores"] = path

        # 6. Sentiment distribution
        if "sentiment_label" in df.columns:
            path = self.plot_sentiment_distribution(df)
            saved["sentiment_dist"] = path

        # 7. Source comparison
        path = self.plot_source_comparison(df)
        saved["source_comparison"] = path

        # 8. Top TF-IDF terms
        if "top_tfidf_terms" in text_analyzer.results:
            path = self.plot_top_tfidf_terms(text_analyzer.results["top_tfidf_terms"])
            saved["tfidf_terms"] = path

        # 9. Word frequencies
        if "word_frequencies" in text_analyzer.results:
            path = self.plot_word_frequencies(
                text_analyzer.results["word_frequencies"]["overall"]
            )
            saved["word_freq"] = path

        logger.info(f"\n✅ Generated {len(saved)} visualizations in {self.output_dir}")
        return saved
