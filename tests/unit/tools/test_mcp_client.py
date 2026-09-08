import asyncio
from pathlib import Path

import pytest

from coding_agent.config import FilesystemMcpSettings
from coding_agent.tools.mcp import StdioMcpClient
from coding_agent.tools.mcp import client as client_module


class FakeSdkClient:
    instances: list["FakeSdkClient"] = []
    fail_enter: BaseException | None = None
    fail_exit = False

    def __init__(self, *args: object, **kwargs: object) -> None:
        self.entered = False
        self.exited = False
        self.__class__.instances.append(self)

    async def __aenter__(self) -> "FakeSdkClient":
        self.entered = True
        if self.fail_enter is not None:
            raise self.fail_enter
        return self

    async def __aexit__(self, *args: object) -> None:
        self.exited = True
        if self.fail_exit:
            raise RuntimeError("cleanup failed")


def make_client(tmp_path: Path) -> StdioMcpClient:
    settings = FilesystemMcpSettings(workspace_root=tmp_path, mcp_command="fake")
    return StdioMcpClient(settings)


def test_stdio_client_rejects_double_enter(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    FakeSdkClient.instances = []
    monkeypatch.setattr(client_module, "Client", FakeSdkClient)
    client = make_client(tmp_path)

    async def scenario() -> None:
        await client.__aenter__()
        with pytest.raises(RuntimeError, match="already connected"):
            await client.__aenter__()
        await client.__aexit__(None, None, None)

    asyncio.run(scenario())
    assert len(FakeSdkClient.instances) == 1
    assert FakeSdkClient.instances[0].exited is True


def test_stdio_client_clears_reference_when_exit_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    FakeSdkClient.instances = []
    FakeSdkClient.fail_exit = True
    monkeypatch.setattr(client_module, "Client", FakeSdkClient)
    client = make_client(tmp_path)

    async def scenario() -> None:
        await client.__aenter__()
        with pytest.raises(RuntimeError, match="cleanup failed"):
            await client.__aexit__(None, None, None)
        with pytest.raises(Exception, match="not connected"):
            await client.list_tools()

    try:
        asyncio.run(scenario())
    finally:
        FakeSdkClient.fail_exit = False


def test_stdio_client_cleans_up_when_startup_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    FakeSdkClient.instances = []
    FakeSdkClient.fail_enter = RuntimeError("startup failed")
    monkeypatch.setattr(client_module, "Client", FakeSdkClient)
    client = make_client(tmp_path)

    async def scenario() -> None:
        with pytest.raises(RuntimeError, match="startup failed"):
            await client.__aenter__()

    try:
        asyncio.run(scenario())
    finally:
        FakeSdkClient.fail_enter = None

    assert len(FakeSdkClient.instances) == 1
    assert FakeSdkClient.instances[0].exited is True
