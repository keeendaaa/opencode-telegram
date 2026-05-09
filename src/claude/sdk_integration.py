"""OpenCode server integration.

This module intentionally keeps the historical Claude* class names because the
rest of the bot uses them as its agent integration contract. Internally, the
implementation talks to a local ``opencode serve`` HTTP server.
"""

import asyncio
import json
import os
import secrets
import socket
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import httpx
import structlog

from ..config.settings import Settings
from ..security.validators import SecurityValidator
from .exceptions import ClaudeParsingError, ClaudeProcessError, ClaudeTimeoutError

logger = structlog.get_logger()

TASK_COMPLETED_MSG = "Task completed. Tools used: {tools_summary}"


@dataclass
class ClaudeResponse:
    """Response from the agent backend."""

    content: str
    session_id: str
    cost: float
    duration_ms: int
    num_turns: int
    is_error: bool = False
    error_type: Optional[str] = None
    tools_used: List[Dict[str, Any]] = field(default_factory=list)
    interrupted: bool = False


@dataclass
class StreamUpdate:
    """Streaming/progress update normalized for bot handlers."""

    type: str
    content: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    metadata: Optional[Dict[str, Any]] = None
    progress: Optional[Dict[str, Any]] = None

    def get_tool_names(self) -> List[str]:
        """Return tool names from the stream payload."""
        names: List[str] = []
        if self.tool_calls:
            for tool_call in self.tool_calls:
                name = tool_call.get("name") if isinstance(tool_call, dict) else None
                if isinstance(name, str) and name:
                    names.append(name)
        if self.metadata:
            tool_name = self.metadata.get("tool_name")
            if isinstance(tool_name, str) and tool_name:
                names.append(tool_name)
            metadata_tools = self.metadata.get("tools")
            if isinstance(metadata_tools, list):
                for tool in metadata_tools:
                    if isinstance(tool, dict):
                        name = tool.get("name")
                    elif isinstance(tool, str):
                        name = tool
                    else:
                        name = None
                    if isinstance(name, str) and name:
                        names.append(name)
        return list(dict.fromkeys(names))

    def is_error(self) -> bool:
        """Check whether this stream update represents an error."""
        if self.type == "error":
            return True
        if self.metadata:
            if self.metadata.get("is_error") is True:
                return True
            status = self.metadata.get("status")
            if isinstance(status, str) and status.lower() == "error":
                return True
            for key in ("error", "error_message"):
                value = self.metadata.get(key)
                if isinstance(value, str) and value:
                    return True
        if self.progress:
            status = self.progress.get("status")
            if isinstance(status, str) and status.lower() == "error":
                return True
        return False

    def get_error_message(self) -> str:
        """Get the best available error message from the stream payload."""
        if self.metadata:
            for key in ("error_message", "error", "message"):
                value = self.metadata.get(key)
                if isinstance(value, str) and value.strip():
                    return value
        if isinstance(self.content, str) and self.content.strip():
            return self.content
        if self.progress:
            value = self.progress.get("error")
            if isinstance(value, str) and value.strip():
                return value
        return "Unknown error"

    def get_progress_percentage(self) -> Optional[int]:
        """Extract progress percentage if present."""

        def _to_int(value: Any) -> Optional[int]:
            if isinstance(value, (int, float)):
                return int(value)
            if isinstance(value, str) and value.strip():
                try:
                    return int(float(value))
                except ValueError:
                    return None
            return None

        if self.progress:
            for key in ("percentage", "percent", "progress"):
                percentage = _to_int(self.progress.get(key))
                if percentage is not None:
                    return max(0, min(100, percentage))
            step = _to_int(self.progress.get("step"))
            total_steps = _to_int(self.progress.get("total_steps"))
            if step is not None and total_steps and total_steps > 0:
                return max(0, min(100, int((step / total_steps) * 100)))
        if self.metadata:
            percentage = _to_int(self.metadata.get("progress_percentage"))
            if percentage is not None:
                return max(0, min(100, percentage))
        return None


@dataclass
class _OpenCodeServer:
    """Running opencode server metadata."""

    cwd: Path
    base_url: str
    username: str
    password: str
    process: asyncio.subprocess.Process


class ClaudeSDKManager:
    """Manage OpenCode server integration."""

    def __init__(
        self,
        config: Settings,
        security_validator: Optional[SecurityValidator] = None,
    ):
        """Initialize OpenCode manager with configuration."""
        self.config = config
        self.security_validator = security_validator
        self._servers: Dict[str, _OpenCodeServer] = {}
        self._lock = asyncio.Lock()

    async def execute_command(
        self,
        prompt: str,
        working_directory: Path,
        session_id: Optional[str] = None,
        continue_session: bool = False,
        stream_callback: Optional[Callable[[StreamUpdate], Any]] = None,
        interrupt_event: Optional[asyncio.Event] = None,
        images: Optional[List[Dict[str, str]]] = None,
    ) -> ClaudeResponse:
        """Execute an agent prompt through ``opencode serve``."""
        start_time = asyncio.get_event_loop().time()
        working_directory = working_directory.resolve()

        logger.info(
            "Starting OpenCode command",
            working_directory=str(working_directory),
            session_id=session_id,
            continue_session=continue_session,
        )

        try:
            server = await self._get_server(working_directory)
            async with self._client(server) as client:
                final_session_id = session_id if session_id and continue_session else None
                if not final_session_id:
                    final_session_id = await self._create_session(client, prompt)

                body = self._build_message_body(prompt, images)
                message_task = asyncio.create_task(
                    self._post_message(client, final_session_id, body)
                )
                interrupt_task: Optional[asyncio.Task[None]] = None
                interrupted = False

                if interrupt_event is not None:

                    async def _abort_on_interrupt() -> None:
                        await interrupt_event.wait()
                        await self._abort_session(client, final_session_id)

                    interrupt_task = asyncio.create_task(_abort_on_interrupt())

                try:
                    if interrupt_task is None:
                        response_data = await asyncio.wait_for(
                            message_task,
                            timeout=self.config.opencode_timeout_seconds,
                        )
                    else:
                        done, _pending = await asyncio.wait(
                            {message_task, interrupt_task},
                            timeout=self.config.opencode_timeout_seconds,
                            return_when=asyncio.FIRST_COMPLETED,
                        )
                        if not done:
                            message_task.cancel()
                            await self._abort_session(client, final_session_id)
                            raise asyncio.TimeoutError
                        if interrupt_task in done:
                            interrupted = True
                            message_task.cancel()
                            response_data = {}
                        else:
                            response_data = await message_task
                finally:
                    if interrupt_task is not None:
                        interrupt_task.cancel()

                content = self._extract_text(response_data).strip()
                tools_used = self._extract_tools(response_data)
                if not content and tools_used:
                    tool_names = [tool["name"] for tool in tools_used if tool.get("name")]
                    content = TASK_COMPLETED_MSG.format(
                        tools_summary=", ".join(list(dict.fromkeys(tool_names)))
                    )
                if interrupted and not content:
                    content = "Request stopped."

                if stream_callback and (content or tools_used):
                    await stream_callback(
                        StreamUpdate(
                            type="assistant",
                            content=content or None,
                            tool_calls=tools_used or None,
                        )
                    )

                duration_ms = int(
                    (asyncio.get_event_loop().time() - start_time) * 1000
                )
                return ClaudeResponse(
                    content=content,
                    session_id=final_session_id,
                    cost=0.0,
                    duration_ms=duration_ms,
                    num_turns=1,
                    tools_used=tools_used,
                    interrupted=interrupted,
                )

        except asyncio.TimeoutError:
            logger.error(
                "OpenCode command timed out",
                timeout_seconds=self.config.opencode_timeout_seconds,
            )
            raise ClaudeTimeoutError(
                f"OpenCode timed out after {self.config.opencode_timeout_seconds}s"
            )
        except httpx.HTTPStatusError as e:
            detail = e.response.text[:2000]
            logger.error(
                "OpenCode HTTP error",
                status_code=e.response.status_code,
                detail=detail,
            )
            raise ClaudeProcessError(
                f"OpenCode HTTP {e.response.status_code}: {detail}"
            )
        except httpx.HTTPError as e:
            logger.error("OpenCode connection error", error=str(e))
            raise ClaudeProcessError(f"Failed to connect to OpenCode: {e}")
        except json.JSONDecodeError as e:
            logger.error("OpenCode JSON decode error", error=str(e))
            raise ClaudeParsingError(f"Failed to decode OpenCode response: {e}")
        except Exception as e:
            logger.error(
                "Unexpected OpenCode error",
                error=str(e),
                error_type=type(e).__name__,
            )
            raise ClaudeProcessError(f"Unexpected OpenCode error: {e}")

    async def _get_server(self, working_directory: Path) -> _OpenCodeServer:
        key = str(working_directory)
        async with self._lock:
            existing = self._servers.get(key)
            if existing and existing.process.returncode is None:
                return existing

            server = await self._start_server(working_directory)
            self._servers[key] = server
            return server

    async def _start_server(self, working_directory: Path) -> _OpenCodeServer:
        port = self.config.opencode_server_port or self._find_free_port()
        hostname = self.config.opencode_server_host
        username = self.config.opencode_server_username
        password = self.config.opencode_server_password or secrets.token_urlsafe(24)
        base_url = f"http://{hostname}:{port}"

        env = os.environ.copy()
        env["OPENCODE_SERVER_USERNAME"] = username
        env["OPENCODE_SERVER_PASSWORD"] = password
        env["OPENCODE_PERMISSION"] = json.dumps(self._permission_config())

        command = [
            self.config.opencode_cli_path or "opencode",
            "serve",
            "--hostname",
            hostname,
            "--port",
            str(port),
        ]
        logger.info("Starting OpenCode server", cwd=str(working_directory), port=port)

        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                cwd=str(working_directory),
                env=env,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
        except FileNotFoundError:
            raise ClaudeProcessError(
                "OpenCode CLI not found. Install it and/or set OPENCODE_CLI_PATH."
            )

        server = _OpenCodeServer(
            cwd=working_directory,
            base_url=base_url,
            username=username,
            password=password,
            process=process,
        )
        await self._wait_until_ready(server)
        return server

    async def _wait_until_ready(self, server: _OpenCodeServer) -> None:
        deadline = asyncio.get_event_loop().time() + self.config.opencode_start_timeout
        last_error = ""
        while asyncio.get_event_loop().time() < deadline:
            if server.process.returncode is not None:
                stderr = await self._read_process_stderr(server.process)
                raise ClaudeProcessError(
                    f"OpenCode server exited early with code "
                    f"{server.process.returncode}: {stderr}"
                )
            try:
                async with self._client(server, timeout=2.0) as client:
                    response = await client.get("/global/health")
                    if response.status_code == 200:
                        return
                    last_error = response.text[:500]
            except httpx.HTTPError as e:
                last_error = str(e)
            await asyncio.sleep(0.2)
        raise ClaudeTimeoutError(
            f"OpenCode server did not become ready: {last_error or 'timeout'}"
        )

    def _client(
        self, server: _OpenCodeServer, timeout: Optional[float] = None
    ) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=server.base_url,
            auth=(server.username, server.password),
            timeout=timeout or self.config.opencode_timeout_seconds,
        )

    async def _create_session(self, client: httpx.AsyncClient, prompt: str) -> str:
        title = prompt.strip().splitlines()[0][:80] or "Telegram session"
        response = await client.post("/session", json={"title": title})
        response.raise_for_status()
        data = response.json()
        session_id = data.get("id")
        if not isinstance(session_id, str) or not session_id:
            raise ClaudeParsingError("OpenCode session create response has no id")
        return session_id

    async def _post_message(
        self,
        client: httpx.AsyncClient,
        session_id: str,
        body: Dict[str, Any],
    ) -> Dict[str, Any]:
        response = await client.post(f"/session/{session_id}/message", json=body)
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict):
            raise ClaudeParsingError("OpenCode message response is not an object")
        return data

    async def _abort_session(
        self, client: httpx.AsyncClient, session_id: str
    ) -> None:
        try:
            response = await client.post(f"/session/{session_id}/abort")
            response.raise_for_status()
        except httpx.HTTPError as e:
            logger.warning("Failed to abort OpenCode session", error=str(e))

    def _build_message_body(
        self, prompt: str, images: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        system_prompt = (
            f"All file operations must stay within {self.config.approved_directory}. "
            "Use relative paths."
        )
        parts: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]

        # OpenCode server accepts structured parts, but image part details can vary
        # between versions. Preserve image context as text for now rather than
        # dropping it silently.
        if images:
            parts.append(
                {
                    "type": "text",
                    "text": f"\n\nAttached images: {len(images)} image(s).",
                }
            )

        body: Dict[str, Any] = {
            "parts": parts,
            "system": system_prompt,
        }
        model = self._model_payload(self.config.opencode_model)
        if model:
            body["model"] = model
        if self.config.opencode_agent:
            body["agent"] = self.config.opencode_agent
        return body

    def _model_payload(self, model: Optional[str]) -> Optional[Dict[str, str]]:
        if not model:
            return None
        provider, sep, model_id = model.partition("/")
        if not sep or not provider or not model_id:
            logger.warning(
                "Ignoring invalid OpenCode model, expected provider/model",
                model=model,
            )
            return None
        return {"providerID": provider, "modelID": model_id}

    def _permission_config(self) -> Any:
        if self.config.disable_tool_validation:
            return "allow"
        return {
            "read": "allow",
            "edit": "allow",
            "bash": "allow",
            "glob": "allow",
            "grep": "allow",
            "task": "allow",
            "todowrite": "allow",
            "webfetch": "allow",
            "websearch": "allow",
            "external_directory": "deny",
            "doom_loop": "deny",
        }

    def _extract_text(self, data: Any) -> str:
        chunks: List[str] = []

        def walk(value: Any) -> None:
            if isinstance(value, dict):
                value_type = value.get("type")
                text = value.get("text")
                if isinstance(text, str) and value_type in {"text", "step-start"}:
                    chunks.append(text)
                elif isinstance(text, str) and "parts" not in value:
                    chunks.append(text)
                for key in ("parts", "content", "children"):
                    child = value.get(key)
                    if child is not None:
                        walk(child)
            elif isinstance(value, list):
                for item in value:
                    walk(item)

        walk(data.get("parts", data) if isinstance(data, dict) else data)
        return "\n".join(chunk for chunk in chunks if chunk.strip())

    def _extract_tools(self, data: Any) -> List[Dict[str, Any]]:
        tools: List[Dict[str, Any]] = []

        def maybe_add(value: Dict[str, Any]) -> None:
            value_type = str(value.get("type", "")).lower()
            call = value.get("call") if isinstance(value.get("call"), dict) else {}
            name = (
                value.get("tool")
                or value.get("toolID")
                or value.get("toolId")
                or value.get("name")
                or call.get("tool")
                or call.get("name")
            )
            if isinstance(name, str) and ("tool" in value_type or call):
                tools.append(
                    {
                        "name": name,
                        "input": value.get("input") or call.get("input") or {},
                    }
                )

        def walk(value: Any) -> None:
            if isinstance(value, dict):
                maybe_add(value)
                for child in value.values():
                    if isinstance(child, (dict, list)):
                        walk(child)
            elif isinstance(value, list):
                for item in value:
                    walk(item)

        walk(data)
        unique: Dict[str, Dict[str, Any]] = {}
        for idx, tool in enumerate(tools):
            key = f"{tool.get('name')}:{idx}"
            unique[key] = tool
        return list(unique.values())

    @staticmethod
    def _find_free_port() -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            return int(sock.getsockname()[1])

    @staticmethod
    async def _read_process_stderr(process: asyncio.subprocess.Process) -> str:
        if process.stderr is None:
            return ""
        try:
            data = await asyncio.wait_for(process.stderr.read(4096), timeout=1)
        except asyncio.TimeoutError:
            return ""
        return data.decode(errors="replace")
