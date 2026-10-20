"""
===============================================================================
EPL Insight Engine — Configuration
===============================================================================
Central configuration file for the EPL Insight Engine project.
Contains all URLs, file paths, crawling parameters, NLP settings,
and clustering hyperparameters.

Author  : EPL Insight Engine Team
Course  : Web Crawling + Text Analysis + Clustering
Domain  : English Premier League
===============================================================================
"""

import os

# =============================================================================
# 1. PROJECT PATHS
# =============================================================================
# Base directory (root of the project)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Output directories — auto-created at runtime
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
CRAWLED_DATA_DIR = os.path.join(OUTPUT_DIR, "crawled_data")
ANALYSIS_RESULTS_DIR = os.path.join(OUTPUT_DIR, "analysis_results")
VISUALIZATIONS_DIR = os.path.join(OUTPUT_DIR, "visualizations")

# =============================================================================
# 2. TARGET URLs FOR WEB CRAWLING
# =============================================================================
# BBC Sport — Premier League section
BBC_URLS = [
    "https://www.bbc.co.uk/sport/football/premier-league",
    "https://www.bbc.co.uk/sport/football/premier-league/scores-fixtures",
]

# Sky Sports — Premier League news
SKY_URLS = [
    "https://www.skysports.com/premier-league-news",
    "https://www.skysports.com/premier-league-results",
]

# Official Premier League website
EPL_URLS = [
    "https://www.premierleague.com/news",
    "https://www.premierleague.com/results",
    "https://www.premierleague.com/stats",
]

# =============================================================================
# 3. WEB CRAWLING PARAMETERS
# =============================================================================
CRAWL_CONFIG = {
    "max_articles_per_source": 30,       # Max articles to crawl per source
    "request_timeout": 15,               # HTTP timeout in seconds
    "delay_between_requests": 2,         # Politeness delay (seconds)
    "max_retries": 3,                    # Retry count on failure
    "user_agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
}

# =============================================================================
# 4. NLP & TEXT ANALYSIS PARAMETERS
# =============================================================================
NLP_CONFIG = {
    # Preprocessing
    "min_word_length": 3,                # Discard words shorter than this
    "max_document_frequency": 0.85,      # TF-IDF: ignore terms in >85% of docs
    "min_document_frequency": 2,         # TF-IDF: ignore terms in <2 docs
    "max_features": 5000,                # Max vocabulary size for TF-IDF
    "ngram_range": (1, 2),               # Unigrams + bigrams

    # Custom stopwords specific to EPL / football domain
    "custom_stopwords": [
        "said", "says", "would", "could", "also", "one", "two",
        "get", "got", "make", "made", "like", "just", "know",
        "bbc", "sky", "sports", "sport", "football", "premier",
        "league", "club", "team", "player", "game", "match",
        "season", "year", "time", "first", "last", "new",
    ],
}

# =============================================================================
# 5. CLUSTERING PARAMETERS
# =============================================================================
CLUSTER_CONFIG = {
    "min_k": 2,                          # Min clusters to try
    "max_k": 10,                         # Max clusters to try
    "default_k": 5,                      # Default if auto-detect fails
    "random_state": 42,                  # Reproducibility seed
    "max_iterations": 300,               # K-Means max iterations
    "n_init": 10,                        # Number of K-Means initializations
    "top_terms_per_cluster": 10,         # Top keywords per cluster
}

# =============================================================================
# 6. VISUALIZATION PARAMETERS
# =============================================================================
VIZ_CONFIG = {
    "figure_dpi": 150,                   # Resolution for saved figures
    "color_palette": "Set2",             # Seaborn color palette
    "wordcloud_max_words": 100,          # Max words in word cloud
    "wordcloud_width": 1600,             # Word cloud image width
    "wordcloud_height": 800,             # Word cloud image height
    "font_family": "sans-serif",         # Font for plots
}

# =============================================================================
# 7. DEMO / SAMPLE DATA FLAG
# =============================================================================
# When True, uses built-in sample data instead of live crawling.
# Useful for demonstration when live sites are unavailable.
USE_DEMO_DATA = True
