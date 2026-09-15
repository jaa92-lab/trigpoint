"""Mechanical fact checks on analyst rationales. No LLM judging.

Adapted from the Clearing's contradiction checker. Analysts get a fixed evidence
bundle and no tools, so three things are red flags:

- TOOL_CLAIM: the rationale says it searched or browsed. It could not have.
- UNSEEN_SOURCE: it cites a website that was not in its evidence.
- PRICE_MISMATCH: every current price it states is far from the price feed.

Flags reduce an analyst's weight in the pool instead of discarding it, because
false positives are expected and no single analyst should be silenced.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from urllib.parse import urlparse

_TOOL_CLAIM = re.compile(
    r"\b(?:I|we)\s+(?:just\s+)?(?:searched|googled|browsed|queried|visited|"
    r"looked\s+(?:it\s+)?up|pulled\s+up|"
    r"checked\s+(?:online|the\s+(?:web|internet|site|website|page)))\b",
    re.IGNORECASE,
)
_URL = re.compile(r"https?://[^\s)\]>\x22\x27<]+")
_PRICE_CLAIM = re.compile(
    r"\b(?:currently(?:\s+trading)?(?:\s+at)?|trading\s+at|last\s+(?:close|price|traded)"
    r"(?:\s+(?:of|at|was))?|closed\s+at|spot\s+price(?:\s+(?:of|is))?|price\s+(?:is|of|was)|"
    r"priced\s+at|now\s+at)\s*(?:around|about|approximately|roughly|near|~)?\s*\$?\s?"
    r"(\d[\d,]*(?:\.\d+)?)",
    re.IGNORECASE,
)


@dataclass
class GroundingReport:
    flags: list[str] = field(default_factory=list)
    weight: float = 1.0


def domain_of(url: str) -> str:
    host = urlparse(url).netloc.lower().split(":")[0]
    return host[4:] if host.startswith("www.") else host


def _domain_allowed(domain: str, allowed: set[str]) -> bool:
    return any(
        domain == a or domain.endswith("." + a) or a.endswith("." + domain) for a in allowed
    )


def check_rationale(
    rationale: str,
    evidence_urls: Iterable[str],
    reference_price: float | None = None,
    known_values: Iterable[float] = (),
    price_tolerance: float = 0.15,
    min_weight: float = 0.25,
) -> GroundingReport:
    report = GroundingReport()
    allowed = {domain_of(u) for u in evidence_urls if u}

    if _TOOL_CLAIM.search(rationale):
        report.flags.append(
            "TOOL_CLAIM: describes searching or browsing, but analysts have no tools"
        )
        report.weight *= 0.7

    cited = {domain_of(u) for u in _URL.findall(rationale)}
    unseen = sorted(d for d in cited if d and not _domain_allowed(d, allowed))
    if unseen:
        report.flags.append(
            f"UNSEEN_SOURCE: cites {', '.join(unseen)}, which was not in its evidence"
        )
        report.weight *= 0.7

    if reference_price and reference_price > 0:
        knowns = [k for k in known_values if k]
        claims = []
        for match in _PRICE_CLAIM.finditer(rationale):
            value = float(match.group(1).replace(",", ""))
            if any(abs(value - k) <= 0.01 * abs(k) for k in knowns):
                continue
            claims.append(value)
        mismatched = [
            c for c in claims if abs(c - reference_price) / reference_price > price_tolerance
        ]
        if claims and len(mismatched) == len(claims):
            report.flags.append(
                f"PRICE_MISMATCH: states a current price of {mismatched[0]:g}, "
                f"but the price feed shows {reference_price:g}"
            )
            report.weight *= 0.6

    report.weight = max(report.weight, min_weight)
    return report
