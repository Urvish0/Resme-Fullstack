# app/services/universal_scraper.py
import asyncio
import json
import re
import logging
from typing import Dict
from urllib.parse import urlparse
from playwright.async_api import async_playwright, Page
from playwright_stealth import stealth_async

logger = logging.getLogger(__name__)


class UniversalJobScraper:
    """A robust scraper for job descriptions from ANY website."""

    def __init__(
        self, headless: bool = True, timeout: int = 45000, max_retries: int = 2
    ):
        self.headless = headless
        self.timeout = timeout
        self.max_retries = max_retries
        self.playwright = None
        self.browser = None

    # async def __aenter__(self):
    #     """Context manager entry."""
    #     self.playwright = await async_playwright().start()

    #     # Use stealth to avoid bot detection
    #     self.browser = await self.playwright.chromium.launch(
    #         headless=self.headless,
    #         args=[
    #             '--disable-blink-features=AutomationControlled',
    #             '--disable-dev-shm-usage',
    #             '--no-sandbox',
    #             '--disable-setuid-sandbox',
    #         ]
    #     )
    #     return self
    async def __aenter__(self):
        """Context manager entry."""
        self.playwright = await async_playwright().start()

        self.browser = await self.playwright.chromium.launch(
            headless=self.headless,
            args=["--disable-blink-features=AutomationControlled"],
        )

        # Create a context with English (US) locale and timezone
        self.context = await self.browser.new_context(
            locale="en-US",
            timezone_id="America/New_York",
            viewport={"width": 1920, "height": 1080},
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self.context:
            await self.context.close()
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()
        if self.playwright:
            await self.playwright.stop()

    async def scrape_job_description(self, url: str) -> Dict[str, str]:
        """
        Main method to scrape a job description from ANY URL.

        Returns:
            Dictionary with: url, job_title, job_description, company, location, source
        """

        for attempt in range(self.max_retries):
            try:
                logger.info(
                    f"[UNIVERSAL_SCRAPER] Attempt {attempt + 1}/{self.max_retries} for {url}"
                )

                # page = await self.browser.new_page()
                page = await self.context.new_page()

                # Set realistic viewport and user agent
                await page.set_viewport_size({"width": 1280, "height": 800})
                await page.set_extra_http_headers(
                    {
                        "Accept-Language": "en-US,en;q=0.9",
                        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                        "Referer": "https://www.google.com/",
                    }
                )

                # Apply stealth techniques to avoid detection
                await stealth_async(page)

                # Navigate to URL with aggressive waiting
                await page.goto(url, wait_until="networkidle", timeout=self.timeout)

                # Wait longer for dynamic content
                await page.wait_for_timeout(
                    3000 + (attempt * 2000)
                )  # Progressive delay

                # Wait for job-related content to appear
                await self._wait_for_job_content(page, attempt)

                # Get content after waiting
                page_content = await page.content()

                # DEBUG: Save page for inspection
                if attempt == 0:
                    try:
                        body_text = await page.inner_text("body")
                        logger.info(f"[DEBUG] Body text length: {len(body_text)} chars")
                        logger.info(f"[DEBUG] First 500 chars: {body_text[:500]}")
                    except Exception:
                        pass

                # Try different extraction strategies
                result = await self._extract_structured_data(page, page_content)

                if not result or len(result.get("job_description", "")) < 300:
                    result = await self._aggressive_extraction(
                        page, page_content, attempt
                    )

                if not result or len(result.get("job_description", "")) < 300:
                    result = await self._fallback_extraction(page)

                # Clean and return if we got enough content
                if result and len(result.get("job_description", "")) >= 100:
                    result["url"] = url
                    result["job_description"] = self.clean_text(
                        result["job_description"]
                    )

                    logger.info(
                        f"[UNIVERSAL_SCRAPER] SUCCESS - Extracted {len(result.get('job_description', ''))} chars from {url}"
                    )
                    await page.close()
                    return result

                await page.close()
                logger.warning(
                    f"[UNIVERSAL_SCRAPER] Attempt {attempt + 1} failed - insufficient content"
                )

            except Exception as e:
                logger.error(f"[UNIVERSAL_SCRAPER] Attempt {attempt + 1} error: {e}")
                if attempt < self.max_retries - 1:
                    logger.info("[UNIVERSAL_SCRAPER] Retrying...")
                    await asyncio.sleep(2)  # Wait before retry
                else:
                    if "page" in locals():
                        await page.close()

        # All attempts failed
        logger.error(f"[UNIVERSAL_SCRAPER] All attempts failed for {url}")
        return {"url": url, "job_description": "", "source": "all_attempts_failed"}

    async def _wait_for_job_content(self, page: Page, attempt: int):
        """Wait for job-related content to load."""
        try:
            # Wait for either text content or specific elements
            await page.wait_for_function(
                """
                () => {
                    const bodyText = document.body.innerText;
                    return bodyText.length > 1500 || 
                           bodyText.toLowerCase().includes('responsibilities') || 
                           bodyText.toLowerCase().includes('qualifications') ||
                           bodyText.toLowerCase().includes('experience') ||
                           bodyText.toLowerCase().includes('requirements') ||
                           bodyText.toLowerCase().includes('job') ||
                           bodyText.toLowerCase().includes('description');
                }
            """,
                timeout=10000,
            )
        except Exception:
            # Fallback: wait for body or main content
            try:
                await page.wait_for_selector(
                    "body, main, article, [role='main']", timeout=5000
                )
            except Exception:
                pass

        # Additional wait for dynamic content
        await page.wait_for_timeout(2000)

    async def _extract_structured_data(
        self, page: Page, content: str
    ) -> Dict[str, str]:
        """Extract from JSON-LD, microdata, or other structured formats."""
        result = {}

        try:
            # Try JSON-LD structured data
            json_ld_script = await page.query_selector(
                'script[type="application/ld+json"]'
            )
            if json_ld_script:
                json_text = await json_ld_script.inner_text()
                try:
                    data = json.loads(json_text)
                    if isinstance(data, dict) and data.get("@type") == "JobPosting":
                        result.update(
                            {
                                "job_title": data.get("title", ""),
                                "job_description": data.get("description", ""),
                                "company": data.get("hiringOrganization", {}).get(
                                    "name", ""
                                ),
                                "location": data.get("jobLocation", {})
                                .get("address", {})
                                .get("addressLocality", ""),
                                "source": "json_ld",
                            }
                        )
                        logger.info(
                            f"[UNIVERSAL_SCRAPER] Found JSON-LD data: {len(result.get('job_description', ''))} chars"
                        )
                        return result
                except json.JSONDecodeError as e:
                    logger.debug(f"[UNIVERSAL_SCRAPER] JSON parse error: {e}")

        except Exception as e:
            logger.debug(f"[UNIVERSAL_SCRAPER] Structured data extraction failed: {e}")

        return result

    async def _aggressive_extraction(
        self, page: Page, content: str, attempt: int
    ) -> Dict[str, str]:
        """Aggressive extraction for any website."""
        result = {}

        try:
            # Strategy 1: Try common job description containers
            selectors = [
                # Common job description selectors
                "[class*='description']",
                "[class*='job-description']",
                "[class*='job-details']",
                "[class*='position-description']",
                "[class*='job-content']",
                "[class*='description-content']",
                "[class*='jobDescription']",
                "[id*='description']",
                "[id*='job-description']",
                "[data-qa*='description']",
                "[data-test*='description']",
                # Generic content areas
                "main",
                "article",
                "[role='main']",
                "#main",
                ".main",
                ".content",
                ".container",
                ".job-view",
                ".position-view",
                ".careers-body",
                # Eightfold.ai specific (Qualcomm)
                "[class*='ef-job-description']",
                "[class*='ef-description']",
                ".description-container",
                # Greenhouse.io specific
                "[class*='greenhouse']",
                "[id*='content']",
                # LinkedIn specific
                "[class*='description__text']",
                "[class*='jobs-description']",
                # Indeed specific
                "#jobDescriptionText",
                "[class*='jobsearch-JobComponent']",
            ]

            best_text = ""
            best_selector = ""

            for selector in selectors:
                try:
                    elements = await page.query_selector_all(selector)
                    for element in elements:
                        text = await element.inner_text()
                        text = text.strip()

                        # Check if this looks like a job description
                        if self._looks_like_job_description(text) and len(text) > len(
                            best_text
                        ):
                            best_text = text
                            best_selector = selector
                            logger.debug(
                                f"[UNIVERSAL_SCRAPER] Found content with selector '{selector}': {len(text)} chars"
                            )
                except Exception:
                    continue  # Skip failed selectors

            if best_text:
                result["job_description"] = best_text
                result["source"] = f"selector:{best_selector}"

                # Try to extract title
                if not result.get("job_title"):
                    title = await self._extract_title(page)
                    if title:
                        result["job_title"] = title

            # Strategy 2: Get all text and filter
            if not result or len(result.get("job_description", "")) < 300:
                all_text = await page.inner_text("body")
                filtered = self._filter_job_text(all_text)
                if len(filtered) > len(result.get("job_description", "")):
                    result["job_description"] = filtered
                    result["source"] = "filtered_body_text"

        except Exception as e:
            logger.debug(f"[UNIVERSAL_SCRAPER] Aggressive extraction failed: {e}")

        return result

    async def _extract_title(self, page: Page) -> str:
        """Extract job title from page."""
        try:
            # Try common title selectors
            title_selectors = [
                "h1",
                "h2",
                "[class*='title']",
                "[class*='job-title']",
                ".job-title",
                ".position-title",
                ".title",
                "[data-qa*='title']",
                "[data-test*='title']",
                "title",  # HTML title tag
            ]

            for selector in title_selectors:
                try:
                    element = await page.query_selector(selector)
                    if element:
                        text = await element.inner_text()
                        text = text.strip()
                        if text and len(text) < 200:  # Reasonable title length
                            return text
                except Exception:
                    continue

            # Fallback: try to get from URL or page title
            title = await page.title()
            if title and len(title) < 200:
                return title

        except Exception as e:
            logger.debug(f"[UNIVERSAL_SCRAPER] Title extraction failed: {e}")

        return ""

    def _looks_like_job_description(self, text: str) -> bool:
        """Check if text looks like a job description."""
        if not text or len(text) < 200:
            return False

        text_lower = text.lower()

        # Check for job-related keywords
        job_keywords = [
            "responsibilities",
            "qualifications",
            "requirements",
            "experience",
            "skills",
            "education",
            "duties",
            "job description",
            "position overview",
            "about the role",
            "what you will do",
            "what you'll do",
            "you will",
            "we are looking for",
            "the ideal candidate",
            "minimum qualifications",
            "preferred qualifications",
            "benefits",
            "compensation",
            "location",
        ]

        keyword_count = sum(1 for keyword in job_keywords if keyword in text_lower)

        # If it has at least 2 job-related keywords OR is long text, it's likely a job description
        return keyword_count >= 2 or len(text) > 1000

    def _filter_job_text(self, text: str) -> str:
        """Filter job-related text from all body text."""
        if not text:
            return ""

        lines = [line.strip() for line in text.split("\n") if line.strip()]

        # Filter lines that look like job content
        job_lines = []
        for line in lines:
            if len(line) < 30:
                continue  # Skip very short lines

            # Skip common non-job content
            lower_line = line.lower()
            non_job_patterns = [
                "privacy",
                "cookie",
                "copyright",
                "terms",
                "sign in",
                "log in",
                "sign up",
                "subscribe",
                "facebook",
                "twitter",
                "linkedin",
                "instagram",
                "navigation",
                "menu",
                "home",
                "about",
                "contact",
                "careers",
                "jobs",
                "search",
                "filter",
                "apply now",
                "share",
                "save",
                "follow",
                "subscribe",
            ]

            if any(pattern in lower_line for pattern in non_job_patterns):
                continue

            # Keep lines that look like job content
            if self._looks_like_job_description(line) or len(line) > 100:
                job_lines.append(line)

        # Limit to reasonable size
        return "\n\n".join(job_lines[:50])

    async def _fallback_extraction(self, page: Page) -> Dict[str, str]:
        """Final fallback - extract all meaningful text."""
        result = {}

        try:
            # Get all text from main content areas
            main_selectors = ["main", "article", "section", "div"]
            all_text = ""

            for selector in main_selectors:
                try:
                    elements = await page.query_selector_all(selector)
                    for element in elements:
                        text = await element.inner_text()
                        if len(text) > len(all_text):
                            all_text = text
                except Exception:
                    continue

            if all_text:
                # Clean and truncate
                cleaned = self.clean_text(all_text)
                result["job_description"] = cleaned[:10000]  # Limit
                result["source"] = "fallback_all_text"

        except Exception as e:
            logger.debug(f"[UNIVERSAL_SCRAPER] Fallback extraction failed: {e}")

        return result

    def _extract_company_from_url(self, url: str) -> str:
        """Extract company name from URL."""
        try:
            domain = urlparse(url).netloc.lower()
            # Remove common subdomains
            for prefix in ["www.", "careers.", "jobs.", "recruiting.", "talent."]:
                domain = domain.replace(prefix, "")

            # Get company name (first part of domain)
            company = domain.split(".")[0]
            return company.title() if company else ""
        except Exception:
            return ""

    def clean_text(self, text: str) -> str:
        """Clean extracted text."""
        if not text:
            return ""

        # Remove excessive whitespace
        text = re.sub(r"\s+", " ", text)
        # Remove HTML tags
        text = re.sub(r"<[^>]+>", " ", text)
        # Remove URLs
        text = re.sub(r"https?://\S+", "", text)
        # Remove email addresses
        text = re.sub(r"\S+@\S+", "", text)
        # Remove special characters but keep basic punctuation
        text = re.sub(r"[^\w\s.,!?-]", " ", text)

        return text.strip()


# Async utility function for easy use in FastAPI
async def get_job_description_from_url(url: str) -> Dict[str, str]:
    """
    One-line function to get job description from any URL.
    Use this in your FastAPI endpoints.

    Example:
        result = await get_job_description_from_url("https://example.com/job")
    """
    # Use non-headless for debugging if needed
    headless_mode = True  # Set to False to see browser for debugging

    async with UniversalJobScraper(
        headless=headless_mode, timeout=45000, max_retries=2
    ) as scraper:
        return await scraper.scrape_job_description(url)
