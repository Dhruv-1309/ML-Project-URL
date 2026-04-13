"""Feature extraction utilities for phishing URL detection.

This module only uses Python standard library tools and regex.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from urllib.parse import urlparse

# Simple IPv4 regex pattern (e.g., 192.168.0.1)
IPV4_PATTERN = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
URL_SHORTENER_DOMAINS = {
    "bit.ly",
    "tinyurl.com",
    "goo.gl",
    "t.co",
    "ow.ly",
    "is.gd",
    "buff.ly",
    "cutt.ly",
    "rebrand.ly",
    "tiny.cc",
}
SUSPICIOUS_TLDS = {
    "tk",
    "ml",
    "ga",
    "cf",
    "gq",
    "xyz",
    "top",
    "work",
    "support",
    "click",
    "country",
    "stream",
}
SUSPICIOUS_KEYWORDS = (
    "login",
    "verify",
    "secure",
    "account",
    "update",
    "bank",
    "confirm",
    "signin",
    "password",
    "billing",
    "wallet",
    "recover",
    "alert",
    "reset",
    "bonus",
    "free",
    "gift",
)


def _ensure_scheme(url: str) -> str:
    """Add a default scheme if missing so urlparse can split reliably."""
    if not isinstance(url, str):
        return ""
    url = url.strip()
    if not url:
        return ""
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", url):
        return "http://" + url
    return url


def _shannon_entropy(text: str) -> float:
    """Compute Shannon entropy of a string."""
    if not text:
        return 0.0

    counts = Counter(text)
    length = len(text)
    entropy = 0.0

    for count in counts.values():
        p = count / length
        entropy -= p * math.log2(p)

    return entropy


def _count_subdomains(hostname: str) -> int:
    """Count subdomains as parts before the main domain.

    Example:
    - mail.google.com -> 1 subdomain (mail)
    - a.b.example.com -> 2 subdomains (a, b)
    """
    if not hostname:
        return 0

    host = hostname.lower().strip(".")

    # If host is likely an IP, we do not treat it as subdomains.
    if IPV4_PATTERN.fullmatch(host):
        return 0

    parts = [p for p in host.split(".") if p]
    if len(parts) <= 2:
        return 0

    return len(parts) - 2


def _tld_from_hostname(hostname: str) -> str:
    """Return the top-level domain if present."""
    if not hostname:
        return ""

    parts = [part for part in hostname.lower().strip(".").split(".") if part]
    if len(parts) < 2:
        return ""
    return parts[-1]


def _count_keyword_hits(text: str) -> int:
    """Count suspicious phishing-related terms in the URL."""
    lowered = text.lower()
    return sum(lowered.count(keyword) for keyword in SUSPICIOUS_KEYWORDS)


def extract_features(url: str) -> dict:
    """Extract beginner-friendly URL features for phishing detection.

    Returns a dictionary with numeric feature values.
    """
    # Handle missing/invalid values gracefully.
    raw_url = "" if url is None else str(url).strip()
    normalized_url = _ensure_scheme(raw_url)
    parsed = urlparse(normalized_url)

    path = parsed.path or ""
    hostname = parsed.hostname or ""
    query = parsed.query or ""
    full_path = f"{path}?{query}" if query else path

    # Basic counts from raw URL string.
    url_length = len(raw_url)
    hostname_length = len(hostname)
    path_length = len(path)
    query_length = len(query)
    num_dots = raw_url.count(".")
    num_hyphens = raw_url.count("-")
    num_at = raw_url.count("@")
    num_digits = sum(char.isdigit() for char in raw_url)
    num_slashes = raw_url.count("/")
    num_underscores = raw_url.count("_")
    num_question_marks = raw_url.count("?")
    num_equals = raw_url.count("=")
    num_ampersands = raw_url.count("&")
    num_percent = raw_url.count("%")

    # Security/suspicious indicators.
    has_ip = 1 if IPV4_PATTERN.search(raw_url) else 0
    has_https = 1 if raw_url.lower().startswith("https://") else 0
    has_suspicious_tld = 1 if _tld_from_hostname(hostname) in SUSPICIOUS_TLDS else 0
    has_shortener = 1 if hostname.lower() in URL_SHORTENER_DOMAINS else 0
    has_double_slash_path = 1 if "//" in full_path else 0
    num_keywords = _count_keyword_hits(raw_url)

    features = {
        "url_length": url_length,
        "hostname_length": hostname_length,
        "path_length": path_length,
        "query_length": query_length,
        "num_dots": num_dots,
        "num_hyphens": num_hyphens,
        "num_at": num_at,
        "num_digits": num_digits,
        "num_slashes": num_slashes,
        "num_underscores": num_underscores,
        "num_question_marks": num_question_marks,
        "num_equals": num_equals,
        "num_ampersands": num_ampersands,
        "num_percent": num_percent,
        "num_subdomains": _count_subdomains(hostname),
        "num_keywords": num_keywords,
        "has_ip": has_ip,
        "has_https": has_https,
        "has_suspicious_tld": has_suspicious_tld,
        "has_shortener": has_shortener,
        "has_double_slash_path": has_double_slash_path,
        "entropy": _shannon_entropy(raw_url),
    }

    # Defensive cleanup: if any value is missing/invalid, set it to 0.
    clean_features = {}
    for key, value in features.items():
        if value is None:
            clean_features[key] = 0
        else:
            clean_features[key] = value

    return clean_features
