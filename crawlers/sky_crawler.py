"""
===============================================================================
EPL Insight Engine — Sky Sports Crawler
===============================================================================
Crawls Premier League news articles from Sky Sports.

Target URL : https://www.skysports.com/premier-league-news
Data Points: Article title, full text, URL, date, category

Methodology:
    1. Fetch the Sky Sports Premier League news page.
    2. Extract article cards/links from the listing layout.
    3. Follow each link to retrieve the full article text.
    4. Structure data into standardized article dicts.

Author  : EPL Insight Engine Team
===============================================================================
"""

from crawlers.base_crawler import BaseCrawler
from config import SKY_URLS, CRAWL_CONFIG


class SkyCrawler(BaseCrawler):
    """
    Web crawler for Sky Sports Premier League content.

    Inherits from BaseCrawler and implements the `crawl()` method
    to extract EPL news articles from Sky Sports pages.
    """

    def __init__(self):
        """Initialize the Sky Sports crawler with preconfigured URLs."""
        super().__init__(
            source_name="Sky Sports",
            base_urls=SKY_URLS,
        )

    def _extract_article_links(self, soup, base_url):
        """
        Extract article links from a Sky Sports listing page.

        Sky Sports news cards typically have links following:
            /football/news/...

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

            # Filter for football news article links
            if any(
                pattern in href
                for pattern in [
                    "/football/news/",
                    "/premier-league/news/",
                    "/football/report/",
                ]
            ):
                # Build absolute URL
                if href.startswith("/"):
                    full_url = "https://www.skysports.com" + href
                elif href.startswith("http"):
                    full_url = href
                else:
                    continue

                if full_url not in seen:
                    seen.add(full_url)
                    links.append(full_url)

        self.logger.info(f"Found {len(links)} article links on listing page")
        return links[:CRAWL_CONFIG["max_articles_per_source"]]

    def _extract_article_content(self, soup, url):
        """
        Extract full article content from a Sky Sports article page.

        Sky Sports article pages typically have:
            - <h1> or .article-header for the headline
            - .article-body or <article> for body text
            - .article-date or <time> for publication date

        Args:
            soup (BeautifulSoup): Parsed HTML of the article page.
            url (str): URL of the article.

        Returns:
            dict or None: Article dict if extraction succeeds, None otherwise.
        """
        # --- Extract Title ---
        title = ""
        # Try specific Sky Sports selectors first
        title_elem = soup.find("h1", class_="article-header__title")
        if not title_elem:
            title_elem = soup.find("h1")
        if title_elem:
            title = title_elem.get_text(strip=True)

        # --- Extract Article Body ---
        content_parts = []

        # Strategy 1: Sky Sports article body class
        body = soup.find("div", class_="article-body")
        if body:
            for p_tag in body.find_all("p"):
                text = p_tag.get_text(strip=True)
                if text and len(text) > 20:
                    content_parts.append(text)

        # Strategy 2: Generic article tag
        if not content_parts:
            article_tag = soup.find("article")
            if article_tag:
                for p_tag in article_tag.find_all("p"):
                    text = p_tag.get_text(strip=True)
                    if text and len(text) > 20:
                        content_parts.append(text)

        # Strategy 3: Broad paragraph search
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
        else:
            date_elem = soup.find("p", class_="article-header__date")
            if date_elem:
                date = date_elem.get_text(strip=True)

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
        Execute the Sky Sports crawling pipeline.

        Returns:
            list: List of article dicts.
        """
        articles = []

        for listing_url in self.base_urls:
            self.logger.info(f"Processing listing page: {listing_url}")

            soup = self.fetch_page(listing_url)
            if not soup:
                self.logger.warning(f"Failed to fetch listing page: {listing_url}")
                continue

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

        self.logger.info(f"Sky Sports crawl complete. Articles: {len(articles)}")
        return articles
