#!/usr/bin/env python3
"""
Checks the GDPR / offline guarantees:

  1. offline_guard blocks outbound connections and DNS lookups, and still
     allows loopback (Python internals on Windows need it).
  2. ffmpeg and ffprobe are restricted to the local "file" protocol.
  3. No module in src/ imports a network library.

No network, model files or GPU needed. Run from project root:
    python tests/test_offline_guard.py
"""
import ast
import socket
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

import offline_guard  # noqa: E402

failures: list[str] = []


def check(name: str, ok: bool) -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    if not ok:
        failures.append(name)


def expect_blocked(name: str, fn) -> None:  # type: ignore[no-untyped-def]
    try:
        fn()
    except offline_guard.NetworkBlockedError:
        check(name, True)
        return
    except Exception as exc:  # pragma: no cover - diagnostic only
        check(f"{name} (raised {type(exc).__name__}: {exc})", False)
        return
    check(f"{name} (was not blocked)", False)


offline_guard.install()
offline_guard.install()  # idempotent

print("Outbound access is blocked:")
expect_blocked("create_connection to a public IP", lambda: socket.create_connection(("1.1.1.1", 443), timeout=1))
expect_blocked("connect to a public IP", lambda: socket.socket().connect(("8.8.8.8", 53)))
expect_blocked("connect_ex to a public IP", lambda: socket.socket().connect_ex(("8.8.8.8", 53)))
expect_blocked("UDP sendto a public IP", lambda: socket.socket(socket.AF_INET, socket.SOCK_DGRAM).sendto(b"x", ("8.8.8.8", 53)))
expect_blocked("DNS lookup (getaddrinfo)", lambda: socket.getaddrinfo("huggingface.co", 443))
expect_blocked("DNS lookup (gethostbyname)", lambda: socket.gethostbyname("example.com"))
expect_blocked("connect by host name", lambda: socket.socket().connect(("example.com", 80)))


def _urlopen() -> None:
    import urllib.request
    urllib.request.urlopen("https://example.com", timeout=2)


try:
    _urlopen()
    check("urllib request (was not blocked)", False)
except OSError:
    # urllib wraps the error in URLError (an OSError subclass)
    check("urllib request", True)

print("Loopback still works:")
server = socket.socket()
server.bind(("127.0.0.1", 0))
server.listen(1)
port = server.getsockname()[1]
threading.Thread(target=lambda: server.accept()[0].close(), daemon=True).start()
try:
    socket.create_connection(("127.0.0.1", port), timeout=2).close()
    socket.getaddrinfo("localhost", port)
    check("loopback connection and localhost lookup", True)
except Exception as exc:
    check(f"loopback connection ({exc})", False)
finally:
    server.close()
try:
    a, b = socket.socketpair()
    a.close()
    b.close()
    check("socket.socketpair", True)
except Exception as exc:
    check(f"socket.socketpair ({exc})", False)

print("ffmpeg/ffprobe read local files only:")
src = (ROOT / "src" / "ffmpeg_wrapper.py").read_text(encoding="utf-8")
check('whitelist is "file" only', '_LOCAL_FILES_ONLY: tuple[str, ...] = ("-protocol_whitelist", "file")' in src)
check("used by ffprobe and ffmpeg", src.count("*_LOCAL_FILES_ONLY,") == 2)
check('applied before "-i"', src.index("*_LOCAL_FILES_ONLY,\n        \"-i\"") > 0 if "*_LOCAL_FILES_ONLY,\n        \"-i\"" in src else False)

print("No network libraries imported in src/ or launcher/:")
_NETWORK_MODULES = {
    "requests", "urllib", "urllib3", "http", "httpx", "aiohttp", "ftplib",
    "smtplib", "webbrowser", "huggingface_hub", "socketserver", "xmlrpc",
}
bad: list[str] = []
for path in sorted([*(ROOT / "src").glob("*.py"), *(ROOT / "launcher").glob("*.py")]):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        names: list[str] = []
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            names = [node.module]
        for name in names:
            if name.split(".")[0] in _NETWORK_MODULES:
                bad.append(f"{path.name}: {name}")
check("no network imports" + (f" (found: {', '.join(bad)})" if bad else ""), not bad)

if failures:
    print(f"\nFAILED: {len(failures)} check(s)", file=sys.stderr)
    sys.exit(1)
print("\nAll offline checks passed.")
