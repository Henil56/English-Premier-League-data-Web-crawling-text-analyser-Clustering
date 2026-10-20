"""
===============================================================================
EPL Insight Engine — Base Crawler (Abstract)
===============================================================================
Provides a reusable foundation for all web crawlers in this project.

Key Features:
    - HTTP session with custom headers and retry logic
    - Politeness delay between requests (respects server load)
    - HTML fetching and BeautifulSoup parsing
    - Standardized article data structure
    - Error handling and logging

All source-specific crawlers (BBC, Sky, EPL) inherit from this class
and implement the abstract `crawl()` method.

Author  : EPL Insight Engine Team
===============================================================================
"""

import time
import json
import os
import logging
from abc import ABC, abstractmethod
from datetime import datetime

import requests
from bs4 import BeautifulSoup

from config import CRAWL_CONFIG, CRAWLED_DATA_DIR

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


class BaseCrawler(ABC):
    """
    Abstract base class for EPL web crawlers.

    Attributes:
        source_name (str): Human-readable name of the data source.
        base_urls (list): List of seed URLs to crawl.
        session (requests.Session): Persistent HTTP session.
        articles (list): Collected articles (list of dicts).
        logger (logging.Logger): Logger instance for this crawler.

    Usage:
        Subclass this and implement the `crawl()` method.
        Call `run()` to execute the full crawl pipeline.
    """

    def __init__(self, source_name, base_urls):
        """
        Initialize the base crawler.

        Args:
            source_name (str): Name of the data source (e.g., "BBC Sport").
            base_urls (list): Seed URLs for crawling.
        """
        self.source_name = source_name
        self.base_urls = base_urls
        self.articles = []
        self.logger = logging.getLogger(source_name)

        # Set up a persistent HTTP session with headers
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": CRAWL_CONFIG["user_agent"],
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-GB,en;q=0.9",
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive",
        })

    def fetch_page(self, url):
        """
        Fetch a web page with retry logic and politeness delay.

        Args:
            url (str): The URL to fetch.

        Returns:
            BeautifulSoup: Parsed HTML document, or None on failure.

        Methodology:
            1. Sends an HTTP GET request with configured timeout.
            2. Retries up to `max_retries` times on failure.
            3. Waits `delay_between_requests` seconds between attempts.
            4. Returns a BeautifulSoup object for successful responses.
        """
        for attempt in range(1, CRAWL_CONFIG["max_retries"] + 1):
            try:
                self.logger.info(
                    f"Fetching: {url} (attempt {attempt}/{CRAWL_CONFIG['max_retries']})"
                )
                response = self.session.get(
                    url,
                    timeout=CRAWL_CONFIG["request_timeout"],
                )
                response.raise_for_status()

                # Politeness delay — respect server load
                time.sleep(CRAWL_CONFIG["delay_between_requests"])

                # Parse HTML using lxml parser for speed
                soup = BeautifulSoup(response.text, "lxml")
                return soup

            except requests.exceptions.RequestException as e:
                self.logger.warning(f"Attempt {attempt} failed for {url}: {e}")
                if attempt < CRAWL_CONFIG["max_retries"]:
                    wait_time = CRAWL_CONFIG["delay_between_requests"] * attempt
                    self.logger.info(f"Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    self.logger.error(f"All {CRAWL_CONFIG['max_retries']} attempts failed for {url}")

        return None

    def create_article(self, title, content, url="", date="", category=""):
        """
        Create a standardized article dictionary.

        Args:
            title (str): Article headline.
            content (str): Full article text.
            url (str): Source URL of the article.
            date (str): Publication date.
            category (str): Article category (e.g., "Transfer News").

        Returns:
            dict: Standardized article record.
        """
        return {
            "title": title.strip() if title else "",
            "content": content.strip() if content else "",
            "url": url.strip() if url else "",
            "date": date.strip() if date else "",
            "category": category.strip() if category else "General",
            "source": self.source_name,
            "crawled_at": datetime.now().isoformat(),
        }

    @abstractmethod
    def crawl(self):
        """
        Execute the crawling logic for this source.
        Must be implemented by each subclass.

        Returns:
            list: List of article dicts.
        """
        pass

    def save_data(self):
        """
        Save crawled articles to a JSON file in the output directory.

        The file is named after the source (e.g., 'bbc_sport_articles.json').
        Creates the output directory if it doesn't exist.

        Returns:
            str: Path to the saved JSON file.
        """
        os.makedirs(CRAWLED_DATA_DIR, exist_ok=True)

        # Clean filename from source name
        filename = self.source_name.lower().replace(" ", "_") + "_articles.json"
        filepath = os.path.join(CRAWLED_DATA_DIR, filename)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.articles, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Saved {len(self.articles)} articles to {filepath}")
        return filepath

    def run(self):
        """
        Execute the full crawl pipeline:
            1. Log start
            2. Call crawl() (implemented by subclass)
            3. Deduplicate article URLs and enforce the per-source limit
            4. Save results to JSON and log summary

        Returns:
            list: List of crawled article dicts.
        """
        self.logger.info(f"{'='*60}")
        self.logger.info(f"Starting crawler: {self.source_name}")
        self.logger.info(f"Target URLs: {len(self.base_urls)}")
        self.logger.info(f"{'='*60}")

        # Enforce the source-wide limit across all listing pages and remove
        # duplicate article URLs discovered on multiple listings.
        crawled_articles = self.crawl()
        self.articles = []
        seen_urls = set()
        max_articles = CRAWL_CONFIG["max_articles_per_source"]
        for article in crawled_articles:
            if len(self.articles) >= max_articles:
                break
            url = article.get("url", "")
            if url and url in seen_urls:
                continue
            if url:
                seen_urls.add(url)
            self.articles.append(article)

        # Save to disk
        if self.articles:
            self.save_data()
        else:
            self.logger.warning("No articles were crawled.")

        self.logger.info(
            f"Crawler {self.source_name} finished. "
            f"Total articles: {len(self.articles)}"
        )

        return self.articles
