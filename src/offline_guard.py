"""
offline_guard.py - Block all outbound network access from the EasyScribe process.

EasyScribe is an offline tool: every model is bundled and every file stays on
the user's computer (GDPR). No module in src/ opens a network connection, but a
third-party library could try one in a future version. This guard makes that
fail closed instead of silently sending data.

What it does:
  - Python sockets may connect only to loopback addresses (127.0.0.0/8, ::1,
    "localhost") or use AF_UNIX. Python's own internals on Windows (for example
    socket.socketpair) need loopback, so loopback stays allowed.
  - DNS lookups for any other host name are refused, so no name leaks either.
  - A blocked attempt raises NetworkBlockedError (an OSError subclass, so
    library code that already handles network errors degrades gracefully) and
    is written to the local log.
  - Proxies are disabled (NO_PROXY=*), because a proxy on loopback would
    otherwise forward traffic out.
  - Standard opt-out environment variables are set for libraries that honour
    them.

What it cannot do: native subprocesses (ffmpeg, whisper-cli) do not use Python
sockets. ffmpeg_wrapper.py restricts ffmpeg/ffprobe to the "file" protocol
instead, and whisper-cli only reads the local WAV and model it is given.

Call install() once, as early as possible (main.py does this).
"""

import ipaddress
import logging
import os
import socket

logger = logging.getLogger(__name__)

_installed = False

# Opt-out signals honoured by common Python libraries. None of these libraries
# ship in EasyScribe today; this is defence in depth.
_OFFLINE_ENV = {
    "DO_NOT_TRACK": "1",
    "HF_HUB_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    # A local proxy (127.0.0.1:port, set in the environment or in Windows
    # Internet Options) would pass the loopback check and then forward traffic
    # out. "*" makes urllib/requests bypass every proxy, so the request hits
    # the DNS/connect block above instead.
    "NO_PROXY": "*",
    "no_proxy": "*",
}


class NetworkBlockedError(ConnectionRefusedError):
    """Raised when code in this process tries to reach the network."""


def _is_local_host(host: object) -> bool:
    if host is None:
        return True
    if isinstance(host, bytes):
        host = host.decode("ascii", errors="replace")
    if not isinstance(host, str):
        return False
    host = host.strip().strip("[]").split("%", 1)[0].lower()
    if host in ("", "localhost", "localhost.", "ip6-localhost"):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _is_allowed_address(family: int, address: object) -> bool:
    if family == getattr(socket, "AF_UNIX", object()):
        return True
    if isinstance(address, tuple) and address:
        return _is_local_host(address[0])
    return False


def _blocked(what: str) -> NetworkBlockedError:
    msg = f"Network access blocked (EasyScribe runs offline): {what}"
    logger.warning(msg)
    return NetworkBlockedError(msg)


def install() -> None:
    """Install the guard. Safe to call more than once."""
    global _installed
    if _installed:
        return

    for key, value in _OFFLINE_ENV.items():
        os.environ[key] = value

    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex
    original_sendto = socket.socket.sendto
    original_getaddrinfo = socket.getaddrinfo
    original_gethostbyname = socket.gethostbyname
    original_gethostbyname_ex = socket.gethostbyname_ex

    def connect(self, address):  # type: ignore[no-untyped-def]
        if not _is_allowed_address(self.family, address):
            raise _blocked(f"connect to {address!r}")
        return original_connect(self, address)

    def connect_ex(self, address):  # type: ignore[no-untyped-def]
        if not _is_allowed_address(self.family, address):
            raise _blocked(f"connect to {address!r}")
        return original_connect_ex(self, address)

    def sendto(self, data, *args):  # type: ignore[no-untyped-def]
        address = args[-1] if args else None
        if not _is_allowed_address(self.family, address):
            raise _blocked(f"send to {address!r}")
        return original_sendto(self, data, *args)

    def getaddrinfo(host, *args, **kwargs):  # type: ignore[no-untyped-def]
        if not _is_local_host(host):
            raise _blocked(f"DNS lookup for {host!r}")
        return original_getaddrinfo(host, *args, **kwargs)

    def gethostbyname(host):  # type: ignore[no-untyped-def]
        if not _is_local_host(host):
            raise _blocked(f"DNS lookup for {host!r}")
        return original_gethostbyname(host)

    def gethostbyname_ex(host):  # type: ignore[no-untyped-def]
        if not _is_local_host(host):
            raise _blocked(f"DNS lookup for {host!r}")
        return original_gethostbyname_ex(host)

    socket.socket.connect = connect  # type: ignore[method-assign]
    socket.socket.connect_ex = connect_ex  # type: ignore[method-assign]
    socket.socket.sendto = sendto  # type: ignore[method-assign]
    socket.getaddrinfo = getaddrinfo  # type: ignore[assignment]
    socket.gethostbyname = gethostbyname  # type: ignore[assignment]
    socket.gethostbyname_ex = gethostbyname_ex  # type: ignore[assignment]

    _installed = True
    logger.info("Offline guard active: outbound network access is blocked")


def is_installed() -> bool:
    """Return True if install() has run.

    main.py calls install() before logging is set up, so the log line in
    install() is lost there. main() logs this flag again afterwards.
    """
    return _installed
