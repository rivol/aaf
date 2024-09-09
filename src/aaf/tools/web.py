import asyncio

from duckduckgo_search import AsyncDDGS


async def search(query: str) -> list[dict]:
    """Search the web for the given query."""

    results = await AsyncDDGS().atext(query, max_results=10)
    return results


async def search_multi(queries: list[str]) -> list[list[dict]]:
    """Execute multiple web searches at once.

    Behaves same as "search" tool, but runs multiple queries in parallel.
    Returned results are in the same order as the input queries.
    """

    return await asyncio.gather(*[search(query) for query in queries])
