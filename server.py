"""Dependency-light local HTTP server for the DenoiseAPT demonstration."""

from __future__ import annotations

import argparse
import json
import sys
import webbrowser
from http import HTTPStatus
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from denoiseapt import __version__  # noqa: E402
from denoiseapt.api import ApiError, DemoService  # noqa: E402


class DemoHandler(SimpleHTTPRequestHandler):
    server_version = f"DenoiseAPT/{__version__}"
    service: DemoService
    web_root = ROOT / "web"
    max_request_bytes = 8 * 1024 * 1024

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(self.web_root), **kwargs)

    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write("[%s] %s\n" % (self.log_date_time_string(), fmt % args))

    def end_headers(self) -> None:
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Cache-Control", "no-store")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; style-src 'self'; script-src 'self'; "
            "img-src 'self' data:; connect-src 'self'; object-src 'none'; base-uri 'none'",
        )
        super().end_headers()

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/health":
            return self._json(HTTPStatus.OK, self.service.health())
        if path == "/api/cases":
            return self._json(HTTPStatus.OK, self.service.list_cases())
        if path.startswith("/api/"):
            return self._json(HTTPStatus.NOT_FOUND, {"error": "Unknown API endpoint."})
        if path == "/":
            self.path = "/index.html"
        return super().do_GET()

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        try:
            payload = self._read_json()
            if path == "/api/analyze":
                return self._json(HTTPStatus.OK, self.service.analyze(payload))
            if path == "/api/intervene":
                return self._json(HTTPStatus.OK, self.service.intervene(payload))
            return self._json(HTTPStatus.NOT_FOUND, {"error": "Unknown API endpoint."})
        except json.JSONDecodeError:
            return self._json(HTTPStatus.BAD_REQUEST, {"error": "Invalid JSON request."})
        except (ApiError, ValueError) as exc:
            return self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
        except Exception as exc:  # do not expose tracebacks through the API
            self.log_error("Unhandled API error: %r", exc)
            return self._json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"error": "The analysis failed. Check the server log for details."},
            )

    def _read_json(self) -> dict:
        raw_length = self.headers.get("Content-Length")
        if raw_length is None:
            raise ApiError("Content-Length is required.")
        try:
            length = int(raw_length)
        except ValueError as exc:
            raise ApiError("Content-Length must be an integer.") from exc
        if length < 0 or length > self.max_request_bytes:
            raise ApiError("Request is too large.")
        body = self.rfile.read(length)
        value = json.loads(body.decode("utf-8"))
        if not isinstance(value, dict):
            raise ApiError("The request body must be an object.")
        return value

    def _json(self, status: HTTPStatus, value: dict) -> None:
        body = json.dumps(value, separators=(",", ":"), allow_nan=False).encode("utf-8")
        self.send_response(status.value)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the local DenoiseAPT web demonstration.")
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help=(
            "Bind address (default: loopback only). A non-loopback address "
            "exposes this unauthenticated research service to the network."
        ),
    )
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--open",
        action="store_true",
        help="Open the local URL in the default browser.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    service = DemoService(ROOT)
    DemoHandler.service = service
    DemoHandler.max_request_bytes = int(service.config["server"]["max_request_bytes"])
    server = ThreadingHTTPServer((args.host, args.port), DemoHandler)
    url = f"http://{args.host}:{server.server_address[1]}"
    print(f"DenoiseAPT is available at {url}", flush=True)
    if args.host not in {"127.0.0.1", "localhost", "::1"}:
        print(
            "WARNING: the demo has no authentication and is bound beyond loopback.",
            file=sys.stderr,
            flush=True,
        )
    print("Press Ctrl+C to stop the local server.", flush=True)
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
