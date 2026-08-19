from __future__ import annotations

import pytest

from denoiseapt import __version__
from denoiseapt.api import ApiError
from server import DemoHandler


def test_server_version_tracks_package_version() -> None:
    assert DemoHandler.server_version == f"DenoiseAPT/{__version__}"


def test_read_json_rejects_non_integer_content_length() -> None:
    handler = object.__new__(DemoHandler)
    handler.headers = {"Content-Length": "not-an-integer"}
    with pytest.raises(ApiError, match="must be an integer"):
        handler._read_json()
