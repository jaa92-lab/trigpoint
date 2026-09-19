"""Environment clean-up at startup.

A key pasted into a GitHub secret can carry a trailing newline. OpenRouter then
rejects every model call ("Forbidden control character detected in headers"),
while code that strips the key first keeps working, which hides the cause.
"""

from __future__ import annotations

import os

SECRET_ENV_VARS = (
    "METACULUS_TOKEN",
    "METACULUS_BASELINE_TOKEN",
    "OPENROUTER_API_KEY",
    "ASKNEWS_API_KEY",
    "ASKNEWS_CLIENT_ID",
    "ASKNEWS_SECRET",
)


def clean_secret_env() -> list[str]:
    """Strip stray whitespace from keys in place. Returns the names that needed it, never the values."""
    cleaned = []
    for name in SECRET_ENV_VARS:
        value = os.environ.get(name)
        if value is not None and value != value.strip():
            os.environ[name] = value.strip()
            cleaned.append(name)
    return cleaned
