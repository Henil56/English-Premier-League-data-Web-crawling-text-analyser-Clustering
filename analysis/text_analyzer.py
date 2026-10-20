"""
===============================================================================
EPL Insight Engine — Text Analyzer
===============================================================================
Performs TF-IDF vectorization, word frequency analysis, and
sentiment analysis on preprocessed EPL articles.

Key Analyses:
    1. TF-IDF Vectorization  — Converts text to numerical feature vectors
    2. Word Frequency         — Top terms per source and overall
    3. Sentiment Analysis     — VADER sentiment scores per article

Libraries:
    - scikit-learn: TF-IDF vectorizer
    - NLTK / vaderSentiment: Sentiment analysis
    - pandas: Data structuring

Author  : EPL Insight Engine Team
===============================================================================
"""

import os
import logging
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from config import NLP_CONFIG, ANALYSIS_RESULTS_DIR

logger = logging.getLogger(__name__)


class TextAnalyzer:
    """
    Text analysis engine for EPL articles.

    Performs three core analyses:
        1. TF-IDF Vectorization — transforms cleaned text into
           numerical feature vectors for downstream clustering.
        2. Word Frequency Analysis — identifies the most frequent
           terms across the corpus and per source.
        3. Sentiment Analysis — scores each article on a positive/
           negative/neutral scale using VADER.

    Attributes:
        tfidf_vectorizer (TfidfVectorizer): Fitted TF-IDF model.
        tfidf_matrix (sparse matrix): Document-term TF-IDF matrix.
        feature_names (list): Vocabulary terms from TF-IDF.
        sentiment_analyzer (SentimentIntensityAnalyzer): VADER instance.
        results (dict): Stored analysis results.

    Example:
        >>> analyzer = TextAnalyzer()
        >>> tfidf_matrix = analyzer.compute_tfidf(cleaned_texts)
        >>> sentiments = analyzer.analyze_sentiment(articles_df)
    """

    def __init__(self):
        """Initialize TF-IDF vectorizer and VADER sentiment analyzer."""
        self.tfidf_vectorizer = TfidfVectorizer(
            max_df=NLP_CONFIG["max_document_frequency"],
            min_df=NLP_CONFIG["min_document_frequency"],
            max_features=NLP_CONFIG["max_features"],
            ngram_range=NLP_CONFIG["ngram_range"],
            sublinear_tf=True,  # Apply log normalization to TF
        )
        self.tfidf_matrix = None
        self.feature_names = None

        # VADER sentiment analyzer
        self.sentiment_analyzer = SentimentIntensityAnalyzer()

        self.results = {}
        logger.info("TextAnalyzer initialized.")

    # =========================================================================
    # 1. TF-IDF VECTORIZATION
    # =========================================================================
    def compute_tfidf(self, cleaned_texts):
        """
        Compute TF-IDF (Term Frequency - Inverse Document Frequency) matrix.

        TF-IDF measures how important a word is to a document relative
        to the entire corpus. Words that appear frequently in one document
        but rarely across all documents get higher scores.

        Formula:
            TF-IDF(t, d, D) = TF(t, d) × IDF(t, D)
            where TF = term frequency in document d
                  IDF = log(N / df(t)) = inverse document frequency

        Args:
            cleaned_texts (list): List of preprocessed text strings.

        Returns:
            scipy.sparse.csr_matrix: TF-IDF document-term matrix
                                      (shape: n_documents × n_features).
        """
        logger.info(f"Computing TF-IDF for {len(cleaned_texts)} documents...")

        # Filter out empty documents
        valid_texts = [t if t.strip() else "empty_document" for t in cleaned_texts]

        # Fit and transform
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(valid_texts)
        self.feature_names = self.tfidf_vectorizer.get_feature_names_out()

        logger.info(
            f"TF-IDF matrix shape: {self.tfidf_matrix.shape} "
            f"(documents × features)"
        )
        logger.info(f"Vocabulary size: {len(self.feature_names)} terms")

        self.results["tfidf_shape"] = self.tfidf_matrix.shape
        self.results["vocabulary_size"] = len(self.feature_names)

        return self.tfidf_matrix

    def get_top_tfidf_terms(self, n=20):
        """
        Get the top N terms by average TF-IDF score across all documents.

        This reveals which terms are most distinctive/important in
        the entire corpus.

        Args:
            n (int): Number of top terms to return.

        Returns:
            list: List of (term, avg_score) tuples, sorted descending.
        """
        if self.tfidf_matrix is None:
            logger.error("TF-IDF not computed yet. Call compute_tfidf() first.")
            return []

        # Compute mean TF-IDF per term across all documents
        mean_tfidf = np.asarray(self.tfidf_matrix.mean(axis=0)).flatten()

        # Sort by score
        top_indices = mean_tfidf.argsort()[-n:][::-1]
        top_terms = [
            (self.feature_names[i], round(mean_tfidf[i], 4))
            for i in top_indices
        ]

        self.results["top_tfidf_terms"] = top_terms
        return top_terms

    # =========================================================================
    # 2. WORD FREQUENCY ANALYSIS
    # =========================================================================
    def compute_word_frequencies(self, cleaned_texts, n=30):
        """
        Compute word frequency distribution across the corpus.

        Args:
            cleaned_texts (list): List of preprocessed text strings.
            n (int): Number of top words to return.

        Returns:
            dict: {
                "overall": [(word, count), ...],
                "total_words": int,
                "unique_words": int
            }
        """
        logger.info("Computing word frequencies...")

        # Flatten all tokens
        all_tokens = []
        for text in cleaned_texts:
            all_tokens.extend(text.split())

        # Count frequencies
        freq_counter = Counter(all_tokens)
        top_words = freq_counter.most_common(n)

        result = {
            "overall": top_words,
            "total_words": len(all_tokens),
            "unique_words": len(freq_counter),
        }

        self.results["word_frequencies"] = result

        logger.info(
            f"Word frequency analysis complete. "
            f"Total: {len(all_tokens)} | Unique: {len(freq_counter)}"
        )

        return result

    def compute_word_frequencies_by_source(self, df, text_column="cleaned_content", n=15):
        """
        Compute word frequencies grouped by source (BBC, Sky, EPL).

        Args:
            df (pd.DataFrame): DataFrame with 'source' and text column.
            text_column (str): Name of the column with cleaned text.
            n (int): Top N words per source.

        Returns:
            dict: {source_name: [(word, count), ...], ...}
        """
        logger.info("Computing word frequencies by source...")

        source_freqs = {}
        for source in df["source"].unique():
            source_texts = df[df["source"] == source][text_column].tolist()
            all_tokens = []
            for text in source_texts:
                if isinstance(text, str):
                    all_tokens.extend(text.split())

            freq_counter = Counter(all_tokens)
            source_freqs[source] = freq_counter.most_common(n)

            logger.info(
                f"  {source}: {len(all_tokens)} total words, "
                f"{len(freq_counter)} unique"
            )

        self.results["source_word_frequencies"] = source_freqs
        return source_freqs

    # =========================================================================
    # 3. SENTIMENT ANALYSIS (VADER)
    # =========================================================================
    def analyze_sentiment(self, df, text_column="content"):
        """
        Perform VADER sentiment analysis on each article.

        VADER (Valence Aware Dictionary and sEntiment Reasoner) is a
        lexicon-based sentiment analyzer specifically tuned for social
        media and news text. It produces four scores:
            - neg: Negative sentiment proportion
            - neu: Neutral sentiment proportion
            - pos: Positive sentiment proportion
            - compound: Normalized composite score (-1 to +1)

        Compound Score Interpretation:
            - compound >= 0.05  → Positive
            - compound <= -0.05 → Negative
            - else              → Neutral

        Args:
            df (pd.DataFrame): DataFrame with article text.
            text_column (str): Column name containing article text.

        Returns:
            pd.DataFrame: Input DataFrame with added sentiment columns:
                          sentiment_neg, sentiment_neu, sentiment_pos,
                          sentiment_compound, sentiment_label
        """
        logger.info(f"Running VADER sentiment analysis on {len(df)} articles...")

        sentiments = []
        for _, row in df.iterrows():
            text = row[text_column]
            if not isinstance(text, str) or not text.strip():
                sentiments.append({
                    "neg": 0.0, "neu": 1.0, "pos": 0.0, "compound": 0.0
                })
                continue

            # VADER analysis
            scores = self.sentiment_analyzer.polarity_scores(text)
            sentiments.append(scores)

        # Add sentiment columns to DataFrame
        sentiment_df = pd.DataFrame(sentiments)
        df["sentiment_neg"] = sentiment_df["neg"].values
        df["sentiment_neu"] = sentiment_df["neu"].values
        df["sentiment_pos"] = sentiment_df["pos"].values
        df["sentiment_compound"] = sentiment_df["compound"].values

        # Classify into labels
        df["sentiment_label"] = df["sentiment_compound"].apply(
            lambda x: "Positive" if x >= 0.05 else ("Negative" if x <= -0.05 else "Neutral")
        )

        # Log summary
        label_counts = df["sentiment_label"].value_counts()
        logger.info("Sentiment Analysis Results:")
        for label, count in label_counts.items():
            logger.info(f"  {label}: {count} articles ({count/len(df)*100:.1f}%)")

        avg_compound = df["sentiment_compound"].mean()
        logger.info(f"  Average compound score: {avg_compound:.3f}")

        self.results["sentiment_summary"] = {
            "label_counts": label_counts.to_dict(),
            "avg_compound": round(avg_compound, 3),
        }

        return df

    # =========================================================================
    # 4. SAVE RESULTS
    # =========================================================================
    def save_results(self, df):
        """
        Save analysis results to CSV files.

        Args:
            df (pd.DataFrame): Analyzed DataFrame.

        Returns:
            dict: Paths to saved files.
        """
        os.makedirs(ANALYSIS_RESULTS_DIR, exist_ok=True)
        saved_files = {}

        # Save full analyzed dataset
        full_path = os.path.join(ANALYSIS_RESULTS_DIR, "analyzed_articles.csv")
        df.to_csv(full_path, index=False, encoding="utf-8")
        saved_files["full_analysis"] = full_path
        logger.info(f"Saved full analysis to {full_path}")

        # Save sentiment summary
        sentiment_path = os.path.join(ANALYSIS_RESULTS_DIR, "sentiment_summary.csv")
        sentiment_cols = ["title", "source", "sentiment_compound", "sentiment_label"]
        existing_cols = [c for c in sentiment_cols if c in df.columns]
        df[existing_cols].to_csv(sentiment_path, index=False, encoding="utf-8")
        saved_files["sentiment"] = sentiment_path

        # Save top TF-IDF terms
        if "top_tfidf_terms" in self.results:
            tfidf_path = os.path.join(ANALYSIS_RESULTS_DIR, "top_tfidf_terms.csv")
            tfidf_df = pd.DataFrame(
                self.results["top_tfidf_terms"],
                columns=["term", "avg_tfidf_score"],
            )
            tfidf_df.to_csv(tfidf_path, index=False)
            saved_files["tfidf_terms"] = tfidf_path

        # Save word frequencies
        if "word_frequencies" in self.results:
            freq_path = os.path.join(ANALYSIS_RESULTS_DIR, "word_frequencies.csv")
            freq_df = pd.DataFrame(
                self.results["word_frequencies"]["overall"],
                columns=["word", "frequency"],
            )
            freq_df.to_csv(freq_path, index=False)
            saved_files["word_freq"] = freq_path

        logger.info(f"All analysis results saved to {ANALYSIS_RESULTS_DIR}")
        return saved_files

    def print_summary(self):
        """Print a formatted summary of all analysis results to console."""
        print("\n" + "=" * 70)
        print("📊  TEXT ANALYSIS SUMMARY")
        print("=" * 70)

        if "tfidf_shape" in self.results:
            print(f"\n📄 TF-IDF Matrix: {self.results['tfidf_shape'][0]} docs × "
                  f"{self.results['tfidf_shape'][1]} features")
            print(f"   Vocabulary size: {self.results['vocabulary_size']} terms")

        if "top_tfidf_terms" in self.results:
            print(f"\n🔤 Top TF-IDF Terms:")
            for term, score in self.results["top_tfidf_terms"][:10]:
                bar = "█" * int(score * 200)
                print(f"   {term:25s} {score:.4f} {bar}")

        if "word_frequencies" in self.results:
            wf = self.results["word_frequencies"]
            print(f"\n📈 Word Frequency Stats:")
            print(f"   Total words: {wf['total_words']:,}")
            print(f"   Unique words: {wf['unique_words']:,}")
            print(f"\n   Top 10 Words:")
            for word, count in wf["overall"][:10]:
                bar = "█" * min(int(count / 2), 30)
                print(f"   {word:20s} {count:5d} {bar}")

        if "sentiment_summary" in self.results:
            ss = self.results["sentiment_summary"]
            print(f"\n😊 Sentiment Analysis:")
            print(f"   Average compound: {ss['avg_compound']:.3f}")
            for label, count in ss["label_counts"].items():
                print(f"   {label:10s}: {count} articles")

        print("\n" + "=" * 70)
