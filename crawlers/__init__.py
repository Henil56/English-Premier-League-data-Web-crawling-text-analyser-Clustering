"""
Crawlers Package — Web crawlers for EPL data sources.
Exports: BBCCrawler, SkyCrawler, EPLCrawler
"""

from crawlers.base_crawler import BaseCrawler
from crawlers.bbc_crawler import BBCCrawler
from crawlers.sky_crawler import SkyCrawler
from crawlers.epl_crawler import EPLCrawler

__all__ = ["BaseCrawler", "BBCCrawler", "SkyCrawler", "EPLCrawler"]
