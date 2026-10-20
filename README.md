# 🏟️ EPL Insight Engine

> **English Premier League: Web Crawling + Text Analysis + Clustering**

A comprehensive Python project that crawls EPL news from **BBC Sport**, **Sky Sports**, and **premierleague.com**, performs NLP text analysis (TF-IDF, sentiment analysis), and applies **K-Means clustering** to discover thematic groups in football journalism.

---

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Pipeline Stages](#pipeline-stages)
6. [Project Structure](#project-structure)
7. [Output](#output)
8. [Methodology](#methodology)
9. [Libraries Used](#libraries-used)

---

## 🏗️ Project Overview

| Aspect | Details |
|--------|---------|
| **Domain** | English Premier League (EPL) |
| **Data Sources** | BBC Sport, Sky Sports, PremierLeague.com |
| **Techniques** | Web Crawling, TF-IDF, Sentiment Analysis, K-Means Clustering |
| **Language** | Python 3.8+ |
| **Output** | CSV exports, PNG visualizations, console reports |

---

## 🏛️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    EPL Insight Engine Pipeline                   │
├─────────────┬──────────────┬──────────────┬─────────────────────┤
│  Stage 1    │   Stage 2    │   Stage 3    │    Stage 4          │
│  Crawling   │   Analysis   │  Clustering  │   Visualization     │
│             │              │              │                     │
│ BBC Sport ──┤ Preprocess ──┤ Optimal K ───┤ Word Clouds         │
│ Sky Sports ─┤ TF-IDF ──────┤ K-Means ─────┤ Cluster Scatter     │
│ EPL.com ────┤ Sentiment ───┤ PCA ─────────┤ Elbow Curve         │
│             │ Word Freq ───┤ Keywords ────┤ Sentiment Charts    │
└─────────────┴──────────────┴──────────────┴─────────────────────┘
```

---

## 🔧 Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)

### Steps

```bash
# 1. Navigate to the project directory
cd epl-insight-engine

# 2. (Optional) Create a virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt
```

### Run Tests

```bash
python -m unittest discover -s tests
```

---

## 🚀 Usage

### Run with Demo Data (Default — Recommended for First Run)
```bash
python main.py
```

### Run with Live Web Crawling
```bash
python main.py --live
```

### Force Demo Mode
```bash
python main.py --demo
```

> **Note:** Demo mode uses 60 pre-built EPL articles spanning 5 categories
> (Transfers, Match Reports, Injuries, Managerial, Tactics). This ensures
> the pipeline produces meaningful results regardless of website availability.
> `dashboard.html` is a static snapshot of the demo run; run `python main.py --demo`
> to refresh its chart images. Its summary numbers are not live-mode metrics.

---

## 📊 Pipeline Stages

### Stage 1: Data Collection
- **Live Mode**: Crawls BBC Sport, Sky Sports, and PremierLeague.com
- **Demo Mode**: Uses 60 realistic sample articles
- Crawlers use retry logic, rate limiting, URL deduplication, a source-wide article limit, and multi-strategy extraction

### Stage 2: Text Preprocessing
- HTML tag and URL removal
- Lowercasing and punctuation removal
- NLTK tokenization
- Stopword removal (English + custom EPL stopwords)
- WordNet lemmatization
- Short word filtering

### Stage 3: Text Analysis
- **TF-IDF Vectorization**: Converts text to numerical feature vectors
- **Sentiment Analysis**: VADER scoring (positive/negative/neutral)
- **Word Frequency**: Top terms overall and per source

### Stage 4: K-Means Clustering
- **Optimal K Detection**: Highest valid Silhouette Score (bounded by corpus size)
- **Elbow Curve**: Recorded as a diagnostic visualization
- **K-Means Fitting**: On TF-IDF vectors
- **Keyword Extraction**: Top terms per cluster from centroids
- **PCA Reduction**: 2D projection for scatter plot

### Stage 5: Visualization (9 plots generated)
| # | Visualization | Description |
|---|---------------|-------------|
| 1 | Word Cloud (Overall) | Most frequent terms across all articles |
| 2 | Cluster Word Clouds | Per-cluster word clouds |
| 3 | Cluster Scatter Plot | 2D PCA projection of clusters |
| 4 | Elbow Curve | Inertia vs. K as a diagnostic |
| 5 | Silhouette Scores | Cluster quality metric vs. K |
| 6 | Sentiment Distribution | Positive/Neutral/Negative counts |
| 7 | Source Comparison | Article counts & sentiment by source |
| 8 | Top TF-IDF Terms | Horizontal bar chart of key terms |
| 9 | Word Frequencies | Most frequent words bar chart |

### Stage 6: Export
- CSV files with all analysis results
- JSON files with raw crawled data
- PNG visualizations

---

## 📁 Project Structure

```
epl-insight-engine/
├── requirements.txt              # Python dependencies
├── config.py                     # Configuration constants
├── main.py                       # Pipeline orchestrator (entry point)
├── README.md                     # This file
│
├── crawlers/                     # Web crawling modules
│   ├── __init__.py
│   ├── base_crawler.py           # Abstract base crawler (retry, headers)
│   ├── bbc_crawler.py            # BBC Sport EPL crawler
│   ├── sky_crawler.py            # Sky Sports EPL crawler
│   └── epl_crawler.py            # PremierLeague.com crawler
│
├── analysis/                     # Text analysis modules
│   ├── __init__.py
│   ├── preprocessor.py           # NLTK preprocessing pipeline
│   ├── text_analyzer.py          # TF-IDF, sentiment, word frequency
│   └── clustering.py             # K-Means clustering engine
│
├── visualization/                # Chart generation
│   ├── __init__.py
│   └── visualizer.py             # All plot functions
│
└── output/                       # Generated results (auto-created)
    ├── crawled_data/             # Raw articles (JSON)
    ├── analysis_results/         # Analysis CSVs
    └── visualizations/           # Charts (PNG)
```

---

## 📈 Output

After running the pipeline, the `output/` directory will contain:

### `crawled_data/`
- `all_articles.json` — Raw crawled/demo articles

### `analysis_results/`
- `complete_analysis.csv` — Full dataset with all features
- `analyzed_articles.csv` — Articles with sentiment scores
- `sentiment_summary.csv` — Sentiment labels per article
- `top_tfidf_terms.csv` — Top TF-IDF terms and scores
- `word_frequencies.csv` — Word frequency distribution
- `cluster_assignments.csv` — Cluster labels per article
- `cluster_keywords.csv` — Top keywords per cluster

### `visualizations/`
- `wordcloud_overall.png`
- `wordclouds_by_cluster.png`
- `cluster_scatter_pca.png`
- `elbow_curve.png`
- `silhouette_scores.png`
- `sentiment_distribution.png`
- `source_comparison.png`
- `top_tfidf_terms.png`
- `word_frequencies.png`

---

## 🔬 Methodology

### TF-IDF (Term Frequency — Inverse Document Frequency)
```
TF-IDF(t, d, D) = TF(t, d) × IDF(t, D)
    TF(t, d)  = frequency of term t in document d
    IDF(t, D) = log(N / df(t))
    N         = total number of documents
    df(t)     = number of documents containing term t
```

### K-Means Clustering
```
Objective: Minimize Σᵢ Σₓ∈Cᵢ ||x - μᵢ||²
    Cᵢ = cluster i
    μᵢ = centroid of cluster i
    x  = data point (TF-IDF vector)
```

### VADER Sentiment Analysis
```
Compound Score ∈ [-1, +1]
    compound ≥  0.05 → Positive
    compound ≤ -0.05 → Negative
    otherwise        → Neutral
```

---

## 📦 Libraries Used

| Library | Purpose |
|---------|---------|
| `requests` | HTTP requests for web crawling |
| `beautifulsoup4` | HTML parsing and content extraction |
| `lxml` | Fast HTML/XML parser |
| `nltk` | Tokenization, stopwords, lemmatization |
| `scikit-learn` | TF-IDF vectorizer, K-Means, PCA, silhouette |
| `matplotlib` | Core plotting and chart generation |
| `seaborn` | Statistical visualizations |
| `wordcloud` | Word cloud generation |
| `pandas` | Data manipulation and CSV I/O |
| `numpy` | Numerical operations |
| `vaderSentiment` | Lexicon-based sentiment analysis |

---

## 👥 Credits

- **Data Sources**: BBC Sport, Sky Sports, PremierLeague.com
- **Project Domain**: English Premier League
- **Techniques**: Web Crawling, NLP, Unsupervised Machine Learning
