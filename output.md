# EPL Insight Engine Output Summary

This project crawls English Premier League news, analyzes the text, clusters article themes, and exports the results under the `output/` directory.

## Project dashboard

- Open the interactive/static dashboard: [dashboard.html](./dashboard.html)
- Generated analysis dataset: [output/analysis_results/complete_analysis.csv](./output/analysis_results/complete_analysis.csv)

## Overall dataset output

| Metric | Value |
|---|---:|
| Total articles analyzed | 60 |
| Sources | BBC Sport, Sky Sports, PremierLeague.com |
| Articles per source | 20 each |
| Sentiment labels | Positive: 47, Negative: 12, Neutral: 1 |
| Clusters discovered | 10 |

### Sentiment distribution by source

- BBC Sport: 20 articles
- Sky Sports: 20 articles
- PremierLeague.com: 20 articles

The overall sentiment profile is heavily positive, with 47 positive headlines, 12 negative headlines, and 1 neutral article.

## Most frequent words

The most common words in the processed EPL corpus were:

1. manager (62)
2. tactical (36)
3. injury (35)
4. team (34)
5. goal (33)
6. player (32)
7. back (27)
8. result (24)
9. arsenal (23)
10. three (23)

Source data: [output/analysis_results/word_frequencies.csv](./output/analysis_results/word_frequencies.csv)

## Top TF-IDF terms

The highest-scoring TF-IDF terms were:

1. manager — 0.0468
2. injury — 0.0394
3. player — 0.0350
4. tactical — 0.0348
5. team — 0.0334
6. goal — 0.0308
7. back — 0.0303
8. result — 0.0294
9. three — 0.0282
10. midfielder — 0.0278

Source data: [output/analysis_results/top_tfidf_terms.csv](./output/analysis_results/top_tfidf_terms.csv)

## Cluster insights

The analysis found 10 cluster groups. Example thematic patterns from the generated cluster keywords:

- Cluster 0: transfer-signing / squad-building / Newcastle / deal / million-pound movement
- Cluster 1: Manchester City / Brighton / Champions-level competition / victory / injury focus
- Cluster 2: tactical analysis / expected goals / team shape / centre-back / attacking play
- Cluster 3: Aston Villa / return / injury recovery / player availability

Source data: [output/analysis_results/cluster_keywords.csv](./output/analysis_results/cluster_keywords.csv)

## Generated visualization files

The project exports the outputs to the following chart files in [output/visualizations](./output/visualizations):

- [wordcloud_overall.png](./output/visualizations/wordcloud_overall.png)
- [wordclouds_by_cluster.png](./output/visualizations/wordclouds_by_cluster.png)
- [cluster_scatter_pca.png](./output/visualizations/cluster_scatter_pca.png)
- [elbow_curve.png](./output/visualizations/elbow_curve.png)
- [silhouette_scores.png](./output/visualizations/silhouette_scores.png)
- [sentiment_distribution.png](./output/visualizations/sentiment_distribution.png)
- [source_comparison.png](./output/visualizations/source_comparison.png)
- [top_tfidf_terms.png](./output/visualizations/top_tfidf_terms.png)
- [word_frequencies.png](./output/visualizations/word_frequencies.png)

## Generated dataset files

The raw and processed outputs are stored in:

- [output/crawled_data/all_articles.json](./output/crawled_data/all_articles.json)
- [output/analysis_results/sentiment_summary.csv](./output/analysis_results/sentiment_summary.csv)
- [output/analysis_results/analyzed_articles.csv](./output/analysis_results/analyzed_articles.csv)
- [output/analysis_results/cluster_assignments.csv](./output/analysis_results/cluster_assignments.csv)
- [output/analysis_results/cluster_keywords.csv](./output/analysis_results/cluster_keywords.csv)
- [output/analysis_results/complete_analysis.csv](./output/analysis_results/complete_analysis.csv)

## Executive takeaway

This project demonstrates a complete EPL text-mining pipeline: articles are gathered from multiple football sources, cleaned and analyzed, clustered into thematic groups, and visualized for easy interpretation. The strongest recurring themes are transfer activity, tactical analysis, injuries, and club performance narratives.
