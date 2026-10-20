"""
===============================================================================
EPL Insight Engine — BBC Sport Crawler
===============================================================================
Crawls Premier League news articles from BBC Sport.

Target URL : https://www.bbc.co.uk/sport/football/premier-league
Data Points: Article title, full text, URL, date, category

Methodology:
    1. Fetch the BBC Sport Premier League landing page.
    2. Extract article links from the page layout.
    3. Follow each link to retrieve the full article text.
    4. Structure data into standardized article dicts.

Note: BBC uses server-side rendering, making it suitable for
      requests + BeautifulSoup crawling (no JS rendering needed).

Author  : EPL Insight Engine Team
===============================================================================
"""

from crawlers.base_crawler import BaseCrawler
from config import BBC_URLS, CRAWL_CONFIG


class BBCCrawler(BaseCrawler):
    """
    Web crawler for BBC Sport Premier League content.

    Inherits from BaseCrawler and implements the `crawl()` method
    to extract EPL news articles from BBC Sport pages.

    Attributes:
        Inherited from BaseCrawler.
    """

    def __init__(self):
        """Initialize the BBC Sport crawler with preconfigured URLs."""
        super().__init__(
            source_name="BBC Sport",
            base_urls=BBC_URLS,
        )

    def _extract_article_links(self, soup, base_url):
        """
        Extract article links from a BBC Sport listing page.

        BBC Sport article links typically follow patterns like:
            /sport/football/articles/...
            /sport/football/...

        Args:
            soup (BeautifulSoup): Parsed HTML of the listing page.
            base_url (str): Base URL for resolving relative links.

        Returns:
            list: List of absolute article URLs.
        """
        links = []
        seen = set()

        # BBC uses various link patterns — search broadly
        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"]

            # Filter for football/premier-league article links
            if any(
                pattern in href
                for pattern in [
                    "/sport/football/articles/",
                    "/sport/football/premier-league/",
                    "/sport/articles/",
                ]
            ):
                # Build absolute URL
                if href.startswith("/"):
                    full_url = "https://www.bbc.co.uk" + href
                elif href.startswith("http"):
                    full_url = href
                else:
                    continue

                # Deduplicate
                if full_url not in seen:
                    seen.add(full_url)
                    links.append(full_url)

        self.logger.info(f"Found {len(links)} article links on listing page")
        return links[:CRAWL_CONFIG["max_articles_per_source"]]

    def _extract_article_content(self, soup, url):
        """
        Extract full article content from a BBC article page.

        BBC article pages typically have:
            - <h1> for the headline
            - <article> or <div data-component="text-block"> for body text
            - <time> for publication date

        Args:
            soup (BeautifulSoup): Parsed HTML of the article page.
            url (str): URL of the article (for reference).

        Returns:
            dict or None: Article dict if extraction succeeds, None otherwise.
        """
        # --- Extract Title ---
        title = ""
        h1_tag = soup.find("h1")
        if h1_tag:
            title = h1_tag.get_text(strip=True)

        # --- Extract Article Body ---
        content_parts = []

        # Strategy 1: Look for <article> tag
        article_tag = soup.find("article")
        if article_tag:
            for p_tag in article_tag.find_all("p"):
                text = p_tag.get_text(strip=True)
                if text and len(text) > 20:  # Skip short fragments
                    content_parts.append(text)

        # Strategy 2: Look for data-component text blocks
        if not content_parts:
            for block in soup.find_all("div", attrs={"data-component": "text-block"}):
                for p_tag in block.find_all("p"):
                    text = p_tag.get_text(strip=True)
                    if text and len(text) > 20:
                        content_parts.append(text)

        # Strategy 3: Generic paragraph extraction
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

        # --- Validate ---
        if not title or len(content) < 50:
            self.logger.debug(f"Skipping article (insufficient content): {url}")
            return None

        return self.create_article(
            title=title,
            content=content,
            url=url,
            date=date,
            category="EPL News",
        )

    def crawl(self):
        """
        Execute the BBC Sport crawling pipeline.

        Pipeline:
            1. Fetch each listing page in self.base_urls
            2. Extract article links from the listing
            3. Follow each link to get full article text
            4. Return list of structured article dicts

        Returns:
            list: List of article dicts with keys:
                  title, content, url, date, category, source, crawled_at
        """
        articles = []

        for listing_url in self.base_urls:
            self.logger.info(f"Processing listing page: {listing_url}")

            # Step 1: Fetch listing page
            soup = self.fetch_page(listing_url)
            if not soup:
                self.logger.warning(f"Failed to fetch listing page: {listing_url}")
                continue

            # Step 2: Extract article links
            article_links = self._extract_article_links(soup, listing_url)

            # Step 3: Follow each link
            for link in article_links:
                article_soup = self.fetch_page(link)
                if not article_soup:
                    continue

                article = self._extract_article_content(article_soup, link)
                if article:
                    articles.append(article)
                    self.logger.info(f"  ✓ Crawled: {article['title'][:60]}...")

                # Stop if we've reached the limit
                if len(articles) >= CRAWL_CONFIG["max_articles_per_source"]:
                    break

        self.logger.info(f"BBC Sport crawl complete. Articles: {len(articles)}")
        return articles
