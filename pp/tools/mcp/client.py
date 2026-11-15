from __future__ import annotations

import asyncio
import collections
import json
import logging
import os
from typing import Any

from pp.utils.process import stop_process_group

logger = logging.getLogger(__name__)


class MCPClient:
    STDERR_TAIL_LINES = 10

    def __init__(self, name: str, command: str, args: list[str] | None = None, env: dict[str, str] | None = None) -> None:
        self.name = name
        self.command = command
        self.args = args or []
        self.env = env or {}
        self._process: asyncio.subprocess.Process | None = None
        self._pending_requests: dict[int | str, asyncio.Future[dict[str, Any]]] = {}
        self._next_id = 1
        self._read_task: asyncio.Task | None = None
        self._stderr_task: asyncio.Task | None = None
        self._connected = False
        # The last stderr lines of the server, to explain why it stopped.
        self._stderr_tail: collections.deque[str] = collections.deque(maxlen=self.STDERR_TAIL_LINES)

    async def connect(self, timeout: float = 15.0) -> None:
        if self._connected:
            return

        logger.info(f"Connecting to MCP server '{self.name}' using command: {self.command} {self.args}...")

        # Merge system environment with configured environment
        env = os.environ.copy()
        env.update(self.env)

        try:
            self._process = await asyncio.create_subprocess_exec(
                self.command,
                *self.args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                # Own process group: a Ctrl-C in the terminal goes to pp only, not to the server.
                # disconnect() stops the group, which includes children such as the server that uvx starts.
                start_new_session=True,
            )
        except Exception as e:
            logger.error(f"Failed to start MCP server process '{self.name}': {e}")
            raise RuntimeError(f"Failed to start MCP server process '{self.name}': {e}") from e

        self._read_task = asyncio.create_task(self._read_stdout())
        self._stderr_task = asyncio.create_task(self._read_stderr())

        try:
            # Step 1: send initialize request
            init_params = {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {
                    "name": "pair-programmer",
                    "version": "0.1.0",
                },
            }
            _ = await self._send_request("initialize", init_params, timeout=timeout)

            # Step 2: send initialized notification
            await self._send_notification("notifications/initialized")

            self._connected = True
            logger.info(f"Successfully connected and initialized MCP server '{self.name}'.")
        except Exception as e:
            # MCPManager logs the failure. Log here at debug level only, so it is not shown twice.
            logger.debug(f"MCP server '{self.name}' initialization handshake failed: {e}")
            await self.disconnect()
            raise

    @property
    def connected(self) -> bool:
        return self._connected

    async def disconnect(self) -> None:
        self._connected = False

        if self._read_task:
            self._read_task.cancel()
            self._read_task = None

        if self._stderr_task:
            self._stderr_task.cancel()
            self._stderr_task = None

        if self._process:
            try:
                await stop_process_group(self._process)
            except Exception as e:
                logger.debug(f"[MCP {self.name}] Error while stopping the server: {e}")
            self._process = None

        # Cancel/reject all pending futures
        for future in self._pending_requests.values():
            if not future.done():
                future.set_exception(RuntimeError("MCP server disconnected"))
        self._pending_requests.clear()

    async def list_tools(self, timeout: float = 10.0) -> list[dict[str, Any]]:
        if not self._connected:
            raise RuntimeError("MCP client is not connected")
        resp = await self._send_request("tools/list", {}, timeout=timeout)
        result = resp.get("result", {})
        return result.get("tools", [])

    async def call_tool(self, tool_name: str, arguments: dict[str, Any], timeout: float = 60.0) -> dict[str, Any]:
        if not self._connected:
            raise RuntimeError("MCP client is not connected")
        params = {
            "name": tool_name,
            "arguments": arguments,
        }
        resp = await self._send_request("tools/call", params, timeout=timeout)
        if "error" in resp:
            raise RuntimeError(f"MCP tool call returned JSON-RPC error: {resp['error']}")
        return resp.get("result", {})

    async def _send_request(self, method: str, params: dict[str, Any], timeout: float) -> dict[str, Any]:
        if not self._process or not self._process.stdin:
            raise RuntimeError("MCP client connection is not active")

        req_id = self._next_id
        self._next_id += 1

        future = asyncio.get_running_loop().create_future()
        self._pending_requests[req_id] = future

        message = {
            "jsonrpc": "2.0",
            "id": req_id,
            "method": method,
            "params": params,
        }

        line = json.dumps(message) + "\n"
        try:
            self._process.stdin.write(line.encode("utf-8"))
            await self._process.stdin.drain()
        except Exception as e:
            self._pending_requests.pop(req_id, None)
            raise RuntimeError(f"Failed to write request to MCP server stdin: {e}") from e

        try:
            return await asyncio.wait_for(future, timeout=timeout)
        except asyncio.TimeoutError as e:
            self._pending_requests.pop(req_id, None)
            raise TimeoutError(f"MCP request '{method}' (id={req_id}) timed out after {timeout} seconds") from e
        except Exception as e:
            self._pending_requests.pop(req_id, None)
            raise e

    async def _send_notification(self, method: str, params: dict[str, Any] | None = None) -> None:
        if not self._process or not self._process.stdin:
            raise RuntimeError("MCP client connection is not active")

        message: dict[str, Any] = {
            "jsonrpc": "2.0",
            "method": method,
        }
        if params is not None:
            message["params"] = params

        line = json.dumps(message) + "\n"
        try:
            self._process.stdin.write(line.encode("utf-8"))
            await self._process.stdin.drain()
        except Exception as e:
            logger.error(f"Failed to write notification to MCP server stdin: {e}")

    async def _read_stdout(self) -> None:
        try:
            while self._process and self._process.stdout:
                line = await self._process.stdout.readline()
                if not line:
                    break

                try:
                    msg = json.loads(line.decode("utf-8"))
                except Exception as e:
                    logger.error(f"[MCP {self.name}] Failed to parse stdout JSON: {e}")
                    continue

                if "id" in msg:
                    # Request/Response format
                    if "result" in msg or "error" in msg:
                        # This is a response to our request
                        req_id = msg["id"]
                        future = self._pending_requests.pop(req_id, None)
                        if future and not future.done():
                            future.set_result(msg)
                    else:
                        # Server-to-client request (currently unsupported, reply with method not found)
                        req_id = msg["id"]
                        method = msg.get("method")
                        logger.debug(f"[MCP {self.name}] Server request received: {method}")
                        error_resp = {
                            "jsonrpc": "2.0",
                            "id": req_id,
                            "error": {
                                "code": -32601,
                                "message": f"Method {method} not found",
                            },
                        }
                        try:
                            if not self._process.stdin:
                                raise RuntimeError("stdin is closed")
                            self._process.stdin.write((json.dumps(error_resp) + "\n").encode("utf-8"))
                            await self._process.stdin.drain()
                        except Exception as e:
                            logger.error(f"[MCP {self.name}] Failed to send error response: {e}")
                else:
                    # Notification
                    method = msg.get("method")
                    if method == "notifications/message":
                        params = msg.get("params", {})
                        level = params.get("level", "info")
                        data = params.get("data", {})
                        message = data.get("message", "")
                        logger.info(f"[MCP {self.name}] {level.upper()}: {message}")
                    else:
                        logger.debug(f"[MCP {self.name}] Received notification: {method}")
        except asyncio.CancelledError:
            # disconnect() stopped the reader. It fails the pending requests itself.
            self._connected = False
            return
        except Exception as e:
            logger.error(f"[MCP {self.name}] Error reading stdout: {e}")

        # stdout closed: the server stopped. Fail the pending requests now, not after their timeout.
        self._connected = False
        await self._fail_pending_requests()

    async def _fail_pending_requests(self) -> None:
        # Give the stderr reader and the process a moment to finish, so that the message is complete.
        if self._stderr_task:
            try:
                await asyncio.wait_for(asyncio.shield(self._stderr_task), timeout=1.0)
            except (asyncio.TimeoutError, asyncio.CancelledError, Exception):
                pass

        exit_code = None
        if self._process:
            try:
                exit_code = await asyncio.wait_for(self._process.wait(), timeout=1.0)
            except asyncio.TimeoutError:
                pass

        message = f"MCP server '{self.name}' stopped"
        if exit_code is not None:
            message += f" with exit code {exit_code}"
        if self._stderr_tail:
            message += ". Last stderr lines:\n" + "\n".join(self._stderr_tail)

        for future in self._pending_requests.values():
            if not future.done():
                future.set_exception(RuntimeError(message))
        self._pending_requests.clear()

    async def _read_stderr(self) -> None:
        try:
            while self._process and self._process.stderr:
                line = await self._process.stderr.readline()
                if not line:
                    break
                text = line.decode("utf-8", errors="replace").rstrip()
                self._stderr_tail.append(text)
                logger.debug(f"[MCP {self.name}] stderr: {text}")
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"[MCP {self.name}] Error reading stderr: {e}")
