"""Documents of blocked web hosts, by their URL: a host on the list or any subdomain of it.

hebban.nl is the source of DBRD's book reviews, so its pages stay out of the pretraining text of a model that is
evaluated on DBRD. This module imports nothing but the standard library: a scan of a staged corpus next to a running
trainer must not import JAX.
"""
from __future__ import annotations

import re
from typing import Iterable, Optional, Tuple

_AFTER_NETLOC = re.compile(r"[/?#\\]")


def url_host(url: Optional[str]) -> Optional[str]:
    """The host of url in lower case, without scheme, user info, port or a final dot; None when there is none.

    Parsed by hand instead of urllib.parse.urlsplit, which raises on some malformed URLs of web crawls (an unclosed
    "[" in the netloc), and takes a URL without a scheme for a path.
    """
    if not url:
        return None
    rest = url.strip()
    scheme_end = rest.find("://")
    rest = rest[scheme_end + 3:] if scheme_end >= 0 else rest.lstrip("/")
    netloc = _AFTER_NETLOC.split(rest, maxsplit=1)[0]
    host = netloc.rsplit("@", 1)[-1]
    if host.startswith("["):
        end = host.find("]")
        host = host if end < 0 else host[: end + 1]
    else:
        host = host.split(":", 1)[0]
    host = host.strip().rstrip(".").lower()
    return host or None


def parse_blocklist(text: str) -> Tuple[str, ...]:
    """The hosts of a comma-separated list, in lower case, sorted; a host with a scheme, path or port is refused."""
    hosts = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if url_host(item) != item.lower().rstrip("."):
            raise ValueError(f"url blocklist entry '{item}' is not a bare host name such as hebban.nl")
        hosts.append(item.lower().rstrip("."))
    if not hosts:
        raise ValueError(f"url blocklist '{text}' has no host")
    return tuple(sorted(set(hosts)))


def host_is_blocked(host: Optional[str], blocked: Iterable[str]) -> bool:
    """True if host is one of blocked or a subdomain of one: www.hebban.nl for hebban.nl, not nothebban.nl."""
    if not host:
        return False
    return any(host == b or host.endswith("." + b) for b in blocked)


def url_is_blocked(url: Optional[str], blocked: Iterable[str]) -> bool:
    return host_is_blocked(url_host(url), blocked)
