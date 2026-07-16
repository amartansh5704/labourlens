# api/rag/web_search.py
# Web search fallback using Tavily API
# Used when Qdrant has no relevant documents

from tavily import TavilyClient
from loguru import logger
from typing import List, Dict, Optional
import os
from dotenv import load_dotenv

load_dotenv()


class WebSearcher:
    """
    Searches the internet for Indian labor law information.
    Used as fallback when Qdrant has no relevant documents.

    Uses Tavily - a search API built specifically for AI apps.
    Returns clean text snippets, not just links.
    Free tier: 1000 searches/month
    """

    def __init__(self):
        api_key = os.getenv("TAVILY_API_KEY", "")

        if not api_key:
            logger.warning(
                "TAVILY_API_KEY not set. "
                "Web search fallback disabled."
            )
            self.enabled = False
            self.client = None
        else:
            self.client = TavilyClient(api_key=api_key)
            self.enabled = True
            logger.info("Web search enabled via Tavily")

    def search(
        self,
        question: str,
        jurisdiction: Optional[str] = None,
        topic: Optional[str] = None,
        max_results: int = 5,
    ) -> List[Dict]:
        """
        Search the web for relevant labor law information.

        Args:
            question:     user's original question
            jurisdiction: state name to focus search
            topic:        topic to focus search
            max_results:  max results to return

        Returns:
            List of result dicts with title, content, url
        """

        if not self.enabled:
            logger.warning("Web search is disabled")
            return []

        # build a focused search query
        search_query = self._build_search_query(
            question=question,
            jurisdiction=jurisdiction,
            topic=topic
        )

        logger.info(f"Web search query: {search_query}")

        try:
            response = self.client.search(
                query=search_query,
                search_depth="advanced",
                max_results=max_results,
                include_answer=True,
                include_raw_content=False,

                # focus on official and reliable sources
                include_domains=[
                    "labour.gov.in",
                    "epfindia.gov.in",
                    "esic.gov.in",
                    "indiacode.nic.in",
                    "labour.delhi.gov.in",
                    "mahakamgar.maharashtra.gov.in",
                    "labour.kar.nic.in",
                    "labour.tn.gov.in",
                    "labour.telangana.gov.in",
                    "clc.gov.in",
                    "pib.gov.in",
                    "economictimes.indiatimes.com",
                    "cleartax.in",
                    "taxguru.in",
                    "hrline.in",
                    "simplepayroll.in",
                ],
            )

            results = self._parse_results(response)

            logger.info(
                f"Web search returned {len(results)} results"
            )
            return results

        except Exception as e:
            logger.error(f"Web search failed: {e}")
            return []

    def search_general(
        self,
        question: str,
        max_results: int = 3,
    ) -> List[Dict]:
        """
        General web search without domain restrictions.
        Used when focused search returns nothing.
        """

        if not self.enabled:
            return []

        query = f"India employment labor law {question}"

        try:
            response = self.client.search(
                query=query,
                search_depth="basic",
                max_results=max_results,
                include_answer=True,
            )

            return self._parse_results(response)

        except Exception as e:
            logger.error(f"General web search failed: {e}")
            return []

    def _build_search_query(
        self,
        question: str,
        jurisdiction: Optional[str],
        topic: Optional[str],
    ) -> str:
        """
        Build a focused search query for Indian labor law.
        Better query = better results.
        """

        # topic keywords to add
        topic_keywords = {
            "minimum_wage": "minimum wages notification",
            "working_hours": "working hours overtime rules",
            "epf_esi": "EPF ESI contribution rates",
            "leave_policy": "leave entitlement rules",
            "worker_classification": "contract labour rules",
        }

        # jurisdiction mapping
        jurisdiction_terms = {
            "Central": "India central government",
            "Delhi": "Delhi NCR",
            "Maharashtra": "Maharashtra",
            "Karnataka": "Karnataka Bangalore",
            "Tamil Nadu": "Tamil Nadu Chennai",
            "Telangana": "Telangana Hyderabad",
        }

        # build query parts
        parts = []

        # add jurisdiction
        if jurisdiction:
            parts.append(
                jurisdiction_terms.get(jurisdiction, jurisdiction)
            )

        # add topic keyword
        if topic and topic in topic_keywords:
            parts.append(topic_keywords[topic])

        # add original question
        parts.append(question)

        # always add India labor law context
        parts.append("India labor law 2024")

        query = " ".join(parts)

        # limit query length
        if len(query) > 200:
            query = query[:200]

        return query

    def _parse_results(self, response: dict) -> List[Dict]:
        """Parse Tavily response into clean result dicts"""

        results = []

        # tavily sometimes returns a direct answer
        if response.get("answer"):
            results.append({
                "title": "Web Search Summary",
                "content": response["answer"],
                "url": "https://tavily.com",
                "source": "web_search_answer",
                "score": 0.9,
            })

        # parse individual results
        for item in response.get("results", []):
            content = item.get("content", "").strip()

            # skip empty or very short content
            if len(content) < 50:
                continue

            results.append({
                "title": item.get("title", "Web Result"),
                "content": content,
                "url": item.get("url", ""),
                "source": "web_search",
                "score": item.get("score", 0.5),
            })

        return results

    def format_for_prompt(
        self,
        results: List[Dict]
    ) -> str:
        """
        Format web search results for inclusion in prompt.
        """
        if not results:
            return "No web search results found."

        parts = []
        for i, result in enumerate(results, 1):
            part = f"""[Web Result {i}]
Title: {result.get('title', 'Unknown')}
Source: {result.get('url', 'Unknown')}
Content: {result.get('content', '')}"""
            parts.append(part)

        return "\n\n---\n\n".join(parts)