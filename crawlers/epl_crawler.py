"""
===============================================================================
EPL Insight Engine — Premier League Official Site Crawler
===============================================================================
Crawls news articles and match data from premierleague.com.

Target URL : https://www.premierleague.com/news
Data Points: Article title, full text, URL, date, category

Note: premierleague.com is heavily JavaScript-rendered. This crawler
      attempts static HTML extraction first, then falls back to
      extracting data from embedded JSON/script tags.

Author  : EPL Insight Engine Team
===============================================================================
"""

from crawlers.base_crawler import BaseCrawler
from config import EPL_URLS, CRAWL_CONFIG
import json
import re


class EPLCrawler(BaseCrawler):
    """
    Web crawler for the official Premier League website.

    Inherits from BaseCrawler and implements the `crawl()` method
    to extract EPL news articles from premierleague.com.

    Note: This site uses heavy client-side rendering. The crawler
    employs multiple strategies to extract content.
    """

    def __init__(self):
        """Initialize the Premier League crawler with preconfigured URLs."""
        super().__init__(
            source_name="PremierLeague.com",
            base_urls=EPL_URLS,
        )

    def _extract_article_links(self, soup, base_url):
        """
        Extract article links from the Premier League news listing.

        Args:
            soup (BeautifulSoup): Parsed HTML of the listing page.
            base_url (str): Base URL for resolving relative links.

        Returns:
            list: List of absolute article URLs.
        """
        links = []
        seen = set()

        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"]

            if any(
                pattern in href
                for pattern in ["/news/", "/match/", "/stats/"]
            ):
                if href.startswith("/"):
                    full_url = "https://www.premierleague.com" + href
                elif href.startswith("http"):
                    full_url = href
                else:
                    continue

                if full_url not in seen:
                    seen.add(full_url)
                    links.append(full_url)

        self.logger.info(f"Found {len(links)} article links on listing page")
        return links[:CRAWL_CONFIG["max_articles_per_source"]]

    def _try_extract_json_data(self, soup):
        """
        Attempt to extract article data from embedded JSON in script tags.

        Many modern sites embed structured data (JSON-LD, __NEXT_DATA__,
        etc.) in <script> tags for SEO and pre-rendering.

        Args:
            soup (BeautifulSoup): Parsed HTML page.

        Returns:
            list: Articles extracted from JSON, or empty list.
        """
        articles = []

        # Look for JSON-LD structured data
        for script in soup.find_all("script", type="application/ld+json"):
            try:
                data = json.loads(script.string)
                if isinstance(data, dict):
                    if data.get("@type") in ["NewsArticle", "Article"]:
                        articles.append(self.create_article(
                            title=data.get("headline", ""),
                            content=data.get("articleBody", data.get("description", "")),
                            url=data.get("url", ""),
                            date=data.get("datePublished", ""),
                            category="EPL Official",
                        ))
                elif isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict) and item.get("@type") in ["NewsArticle", "Article"]:
                            articles.append(self.create_article(
                                title=item.get("headline", ""),
                                content=item.get("articleBody", item.get("description", "")),
                                url=item.get("url", ""),
                                date=item.get("datePublished", ""),
                                category="EPL Official",
                            ))
            except (json.JSONDecodeError, TypeError):
                continue

        # Look for __NEXT_DATA__ (Next.js apps)
        for script in soup.find_all("script", id="__NEXT_DATA__"):
            try:
                data = json.loads(script.string)
                # Navigate the Next.js data structure
                page_props = data.get("props", {}).get("pageProps", {})
                articles_data = page_props.get("articles", page_props.get("news", []))
                if isinstance(articles_data, list):
                    for item in articles_data[:CRAWL_CONFIG["max_articles_per_source"]]:
                        if isinstance(item, dict):
                            articles.append(self.create_article(
                                title=item.get("title", item.get("headline", "")),
                                content=item.get("body", item.get("content", item.get("description", ""))),
                                url=item.get("url", item.get("path", "")),
                                date=item.get("publishDate", item.get("date", "")),
                                category="EPL Official",
                            ))
            except (json.JSONDecodeError, TypeError, AttributeError):
                continue

        return articles

    def _extract_article_content(self, soup, url):
        """
        Extract article content from a premierleague.com article page.

        Args:
            soup (BeautifulSoup): Parsed HTML of the article page.
            url (str): URL of the article.

        Returns:
            dict or None: Article dict if extraction succeeds.
        """
        # --- Extract Title ---
        title = ""
        h1_tag = soup.find("h1")
        if h1_tag:
            title = h1_tag.get_text(strip=True)

        # --- Extract Content ---
        content_parts = []

        # Strategy 1: Article body container
        for selector in ["article", ".article-body", ".content", ".newsArticle"]:
            container = soup.find(
                "div" if selector.startswith(".") else selector,
                class_=selector.lstrip(".") if selector.startswith(".") else None,
            )
            if container:
                for p_tag in container.find_all("p"):
                    text = p_tag.get_text(strip=True)
                    if text and len(text) > 20:
                        content_parts.append(text)
                if content_parts:
                    break

        # Strategy 2: Broad paragraph search
        if not content_parts:
            for p_tag in soup.find_all("p"):
                text = p_tag.get_text(strip=True)
                if text and len(text) > 40:
                    content_parts.append(text)

        content = " ".join(content_parts)

        # --- Extract Date ---
        date = ""
        time_tag = soup.find("time")
        if time_tag:
            date = time_tag.get("datetime", time_tag.get_text(strip=True))

        if not title or len(content) < 50:
            return None

        return self.create_article(
            title=title,
            content=content,
            url=url,
            date=date,
            category="EPL Official",
        )

    def crawl(self):
        """
        Execute the Premier League website crawling pipeline.

        Uses a multi-strategy approach:
            1. Try to extract JSON-LD / embedded data first.
            2. Fall back to HTML-based extraction.

        Returns:
            list: List of article dicts.
        """
        articles = []

        for listing_url in self.base_urls:
            self.logger.info(f"Processing: {listing_url}")

            soup = self.fetch_page(listing_url)
            if not soup:
                self.logger.warning(f"Failed to fetch: {listing_url}")
                continue

            # Strategy A: Try embedded JSON data first
            json_articles = self._try_extract_json_data(soup)
            if json_articles:
                articles.extend(json_articles)
                self.logger.info(
                    f"Extracted {len(json_articles)} articles from embedded JSON"
                )
                continue

            # Strategy B: Follow article links
            article_links = self._extract_article_links(soup, listing_url)

            for link in article_links:
                article_soup = self.fetch_page(link)
                if not article_soup:
                    continue

                article = self._extract_article_content(article_soup, link)
                if article:
                    articles.append(article)
                    self.logger.info(f"  ✓ Crawled: {article['title'][:60]}...")

                if len(articles) >= CRAWL_CONFIG["max_articles_per_source"]:
                    break

        self.logger.info(f"PremierLeague.com crawl complete. Articles: {len(articles)}")
        return articles
