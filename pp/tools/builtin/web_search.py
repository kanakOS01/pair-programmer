from ddgs import DDGS
from pydantic import BaseModel, Field

from pp.domain import ToolInvocation, ToolResult, ToolType
from pp.tools.base import Tool


class WebSearchParams(BaseModel):
    query: str = Field(..., description="Search query")
    max_results: int = Field(10, ge=1, le=20, description="Maximum number of results to return. Defaults to 10.")


class WebSearchTool(Tool):
    name = "web_search"
    description = "Search the web. Returns search results with titles, URLs and snippets"
    type = ToolType.NETWORK
    schema = WebSearchParams

    TIMEOUT = 5

    async def execute(self, invocation: ToolInvocation) -> ToolResult:
        params = WebSearchParams(**invocation.params)

        try:
            results = DDGS(timeout=self.TIMEOUT).text(
                params.query, region="us-en", safesearch="off", timelimit="y", page=1, backend="auto"
            )
        except Exception as e:
            return ToolResult.error_result(f"Search failed: {e}")

        if not results:
            return ToolResult.success_result(f"No search results found for {params.query}", metadata={"results": 0})

        output = [f"Search results for: {params.query}"]

        for i, result in enumerate(results, start=1):
            output.append(f"{i}. Title: {result['title']}")
            output.append(f"   URL: {result['href']}")
            if result.get("body"):
                output.append(f"   Snippet: {result['body']}")
            output.append("")

        return ToolResult.success_result(
            "\n".join(output),
            metadata={"results": len(results)},
        )
