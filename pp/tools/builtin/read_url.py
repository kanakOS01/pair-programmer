import ipaddress
import socket
from urllib.parse import urlparse

import httpx
from pydantic import BaseModel, Field

from pp.domain import ToolInvocation, ToolResult, ToolType
from pp.tools.base import Tool


class ReadUrlParams(BaseModel):
    url: str = Field(..., description="Url to fetch (http/https only)")
    timeout: float = Field(10, le=60, description="Request timeout in seconds. Defaults to 10.0")


class ReadUrlTool(Tool):
    name = "read_url"
    description = "Fetch text content from a URL"
    type = ToolType.NETWORK
    schema = ReadUrlParams

    MAX_BYTES = 100_000

    async def execute(self, invocation: ToolInvocation) -> ToolResult:
        params = ReadUrlParams(**invocation.params)

        try:
            self._validate_url(params.url)
        except ValueError as e:
            return ToolResult.error_result(str(e))

        try:
            async with httpx.AsyncClient(timeout=params.timeout, follow_redirects=False) as client:
                resp = await client.get(params.url)
                resp.raise_for_status()
        except httpx.TimeoutException:
            return ToolResult.error_result("Request timed out")
        except httpx.HTTPStatusError as e:
            return ToolResult.error_result(
                f"HTTP {e.response.status_code}: {e.response.reason_phrase}",
            )
        except Exception as e:
            return ToolResult.error_result(f"Request failed: {e}")

        content = resp.text[: self.MAX_BYTES]
        truncated = len(resp.text) > self.MAX_BYTES
        if truncated:
            content += "\n... [truncated]"

        return ToolResult.success_result(
            content, metadata={"status": resp.status_code, "truncated": truncated, "bytes": len(content)}
        )

    # ======= Helpers ======= #

    def _validate_url(self, url: str) -> None:
        parsed = urlparse(url)

        if parsed.scheme not in ("http", "https"):
            raise ValueError("Only http/https urls allowd")

        if not parsed.hostname:
            raise ValueError("Invalid url")

        # resolve DNS
        try:
            ip = socket.gethostbyname(parsed.hostname)
        except Exception:
            raise ValueError("Unable to resolve host") from None

        ip_obj = ipaddress.ip_address(ip)

        # block private networks (SSRF protection)
        if ip_obj.is_private or ip_obj.is_loopback or ip_obj.is_link_local or ip_obj.is_reserved:
            raise ValueError("Access to private networks is blocked")
