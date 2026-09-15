"""A small JSON disk cache for pastcasting.

Replaying the same resolved questions many times must not re-spend rate-limited
news calls or re-hit archives, and repeat runs must see identical evidence so
the noise floor measures model variance only.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


class DiskCache:
    def __init__(self, root: Path | str | None) -> None:
        self.root = Path(root) if root else None

    def _path(self, namespace: str, key: str) -> Path:
        assert self.root is not None
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:32]
        return self.root / namespace / f"{digest}.json"

    def get(self, namespace: str, key: str) -> Any | None:
        if self.root is None:
            return None
        path = self._path(namespace, key)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def set(self, namespace: str, key: str, value: Any) -> None:
        if self.root is None:
            return
        path = self._path(namespace, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
