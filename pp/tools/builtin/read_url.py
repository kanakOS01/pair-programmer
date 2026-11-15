import asyncio
import ipaddress
import socket
from urllib.parse import urljoin, urlparse

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
    MAX_REDIRECTS = 5

    async def execute(self, invocation: ToolInvocation) -> ToolResult:
        params = ReadUrlParams(**invocation.params)

        try:
            # Follow redirects manually, so that the private-network check runs on every hop.
            async with httpx.AsyncClient(timeout=params.timeout, follow_redirects=False) as client:
                url = params.url
                for _ in range(self.MAX_REDIRECTS + 1):
                    await self._validate_url(url)
                    resp = await client.get(url)
                    if not resp.is_redirect:
                        break
                    url = urljoin(str(resp.url), resp.headers["location"])
                else:
                    return ToolResult.error_result(f"Too many redirects (more than {self.MAX_REDIRECTS})")
                resp.raise_for_status()
        except ValueError as e:
            return ToolResult.error_result(str(e))
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

    async def _validate_url(self, url: str) -> None:
        parsed = urlparse(url)

        if parsed.scheme not in ("http", "https"):
            raise ValueError("Only http/https urls allowed")

        if not parsed.hostname:
            raise ValueError("Invalid url")

        # Resolve DNS in a thread, so that the event loop does not block. Check all addresses (IPv4 and IPv6).
        # Note: httpx resolves the host again, so a DNS rebinding attack can still pass this check.
        try:
            infos = await asyncio.to_thread(socket.getaddrinfo, parsed.hostname, parsed.port or None)
        except Exception:
            raise ValueError("Unable to resolve host") from None

        for info in infos:
            ip_obj = ipaddress.ip_address(info[4][0].split("%")[0])
            if isinstance(ip_obj, ipaddress.IPv6Address) and ip_obj.ipv4_mapped:
                ip_obj = ip_obj.ipv4_mapped

            # block private networks (SSRF protection)
            if (
                ip_obj.is_private
                or ip_obj.is_loopback
                or ip_obj.is_link_local
                or ip_obj.is_reserved
                or ip_obj.is_multicast
                or ip_obj.is_unspecified
            ):
                raise ValueError("Access to private networks is blocked")
