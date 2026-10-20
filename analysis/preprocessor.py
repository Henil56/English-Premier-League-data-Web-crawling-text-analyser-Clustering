"""
===============================================================================
EPL Insight Engine — Text Preprocessor
===============================================================================
Handles all text cleaning and preprocessing for NLP analysis.

Pipeline:
    Raw Text → Lowercase → Remove HTML → Remove URLs → Remove Punctuation
    → Tokenize → Remove Stopwords → Lemmatize → Filter Short Words
    → Cleaned Tokens / Cleaned Text

Key Libraries:
    - NLTK: Tokenization, stopword removal, lemmatization
    - re: Regular expressions for pattern removal

Author  : EPL Insight Engine Team
===============================================================================
"""

import re
import string
import logging

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from config import NLP_CONFIG

# Download required NLTK data (first run only)
for resource in ["punkt", "punkt_tab", "stopwords", "wordnet", "averaged_perceptron_tagger", "omw-1.4"]:
    try:
        nltk.data.find(f"tokenizers/{resource}" if "punkt" in resource else f"corpora/{resource}")
    except LookupError:
        nltk.download(resource, quiet=True)

logger = logging.getLogger(__name__)


class TextPreprocessor:
    """
    Text preprocessing pipeline for EPL article analysis.

    This class implements a complete NLP preprocessing pipeline:
        1. HTML tag removal
        2. URL removal
        3. Lowercasing
        4. Punctuation and special character removal
        5. Tokenization (NLTK word_tokenize)
        6. Stopword removal (English + custom EPL stopwords)
        7. Lemmatization (WordNet lemmatizer)
        8. Short word filtering

    Attributes:
        lemmatizer (WordNetLemmatizer): NLTK lemmatizer instance.
        stop_words (set): Combined English + custom stopwords.
        min_word_length (int): Minimum word length to keep.

    Example:
        >>> preprocessor = TextPreprocessor()
        >>> clean = preprocessor.preprocess("Arsenal signed a new striker!")
        >>> print(clean)
        'arsenal signed striker'
    """

    def __init__(self):
        """
        Initialize the preprocessor with NLTK resources and
        custom EPL-specific stopwords.
        """
        self.lemmatizer = WordNetLemmatizer()
        self.min_word_length = NLP_CONFIG["min_word_length"]

        # Combine NLTK English stopwords with custom EPL stopwords
        self.stop_words = set(stopwords.words("english"))
        self.stop_words.update(NLP_CONFIG["custom_stopwords"])

        logger.info(
            f"TextPreprocessor initialized. "
            f"Stopwords: {len(self.stop_words)} | "
            f"Min word length: {self.min_word_length}"
        )

    def remove_html_tags(self, text):
        """
        Remove HTML tags from text using regex.

        Args:
            text (str): Input text possibly containing HTML tags.

        Returns:
            str: Text with HTML tags removed.
        """
        return re.sub(r"<[^>]+>", " ", text)

    def remove_urls(self, text):
        """
        Remove URLs (http, https, www) from text.

        Args:
            text (str): Input text possibly containing URLs.

        Returns:
            str: Text with URLs removed.
        """
        return re.sub(r"https?://\S+|www\.\S+", " ", text)

    def remove_special_characters(self, text):
        """
        Remove punctuation, numbers, and special characters.
        Keeps only alphabetic characters and spaces.

        Args:
            text (str): Input text.

        Returns:
            str: Text with only letters and spaces.
        """
        # Remove numbers
        text = re.sub(r"\d+", " ", text)
        # Remove punctuation
        text = text.translate(str.maketrans("", "", string.punctuation))
        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def tokenize(self, text):
        """
        Tokenize text into individual words using NLTK.

        Args:
            text (str): Input text.

        Returns:
            list: List of word tokens.
        """
        return word_tokenize(text)

    def remove_stopwords(self, tokens):
        """
        Remove stopwords from a list of tokens.

        Args:
            tokens (list): List of word tokens.

        Returns:
            list: Tokens with stopwords removed.
        """
        return [token for token in tokens if token.lower() not in self.stop_words]

    def lemmatize(self, tokens):
        """
        Lemmatize tokens to their base/root forms.

        Uses WordNet lemmatizer which maps words to their dictionary
        form (e.g., 'running' → 'run', 'better' → 'good').

        Args:
            tokens (list): List of word tokens.

        Returns:
            list: Lemmatized tokens.
        """
        return [self.lemmatizer.lemmatize(token) for token in tokens]

    def filter_short_words(self, tokens):
        """
        Remove tokens shorter than the minimum word length.

        Args:
            tokens (list): List of word tokens.

        Returns:
            list: Filtered tokens.
        """
        return [token for token in tokens if len(token) >= self.min_word_length]

    def preprocess(self, text):
        """
        Execute the full preprocessing pipeline on a single text.

        Pipeline Steps:
            1. Remove HTML tags
            2. Remove URLs
            3. Convert to lowercase
            4. Remove special characters
            5. Tokenize
            6. Remove stopwords
            7. Lemmatize
            8. Filter short words
            9. Rejoin into cleaned string

        Args:
            text (str): Raw input text.

        Returns:
            str: Fully preprocessed text.
        """
        if not text or not isinstance(text, str):
            return ""

        # Step 1: Remove HTML
        text = self.remove_html_tags(text)

        # Step 2: Remove URLs
        text = self.remove_urls(text)

        # Step 3: Lowercase
        text = text.lower()

        # Step 4: Remove special characters
        text = self.remove_special_characters(text)

        # Step 5: Tokenize
        tokens = self.tokenize(text)

        # Step 6: Remove stopwords
        tokens = self.remove_stopwords(tokens)

        # Step 7: Lemmatize
        tokens = self.lemmatize(tokens)

        # Step 8: Filter short words
        tokens = self.filter_short_words(tokens)

        # Rejoin tokens into a cleaned string
        return " ".join(tokens)

    def preprocess_batch(self, texts):
        """
        Preprocess a batch of texts.

        Args:
            texts (list): List of raw text strings.

        Returns:
            list: List of preprocessed text strings.
        """
        logger.info(f"Preprocessing batch of {len(texts)} documents...")
        processed = []
        for i, text in enumerate(texts):
            cleaned = self.preprocess(text)
            processed.append(cleaned)
            if (i + 1) % 10 == 0:
                logger.info(f"  Preprocessed {i + 1}/{len(texts)} documents")

        # Report statistics
        non_empty = sum(1 for t in processed if t.strip())
        logger.info(
            f"Preprocessing complete. "
            f"Non-empty documents: {non_empty}/{len(texts)}"
        )
        return processed

    def get_tokens(self, text):
        """
        Preprocess text and return as a list of tokens (not joined string).

        Useful for word frequency analysis and other token-level operations.

        Args:
            text (str): Raw input text.

        Returns:
            list: List of cleaned tokens.
        """
        cleaned = self.preprocess(text)
        return cleaned.split() if cleaned else []
