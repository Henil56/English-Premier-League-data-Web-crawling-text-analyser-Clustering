import unittest
from unittest.mock import patch

from config import CRAWL_CONFIG
from crawlers.base_crawler import BaseCrawler


class StubCrawler(BaseCrawler):
    def __init__(self, articles):
        super().__init__("Test Source", [])
        self.test_articles = articles

    def crawl(self):
        return self.test_articles

    def save_data(self):
        return "test-output.json"


class BaseCrawlerTests(unittest.TestCase):
    def test_run_deduplicates_urls_and_enforces_source_limit(self):
        articles = [
            {"url": "https://example.com/one"},
            {"url": "https://example.com/one"},
            {"url": "https://example.com/two"},
            {"url": "https://example.com/three"},
        ]
        crawler = StubCrawler(articles)

        with patch.dict(CRAWL_CONFIG, {"max_articles_per_source": 2}):
            results = crawler.run()

        self.assertEqual(
            [article["url"] for article in results],
            ["https://example.com/one", "https://example.com/two"],
        )

    def test_run_resets_previous_results(self):
        crawler = StubCrawler([{"url": "https://example.com/one"}])

        first_run = crawler.run()
        second_run = crawler.run()

        self.assertEqual(len(first_run), 1)
        self.assertEqual(len(second_run), 1)


if __name__ == "__main__":
    unittest.main()
