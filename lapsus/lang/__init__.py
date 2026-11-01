# SPDX-License-Identifier: Apache-2.0
"""Language registry. Add a new language by implementing lapsus.lang.base.Language
in a new module and registering it here."""

from __future__ import annotations

from .base import Language
from .en import English
from .it import Italian, find_morphit_file

_REGISTRY: dict[str, type[Language]] = {"en": English, "it": Italian}
_INSTANCES: dict[str, Language] = {}


def get_language(code: str, cfg: dict | None = None) -> Language:
    """Return a (cached) language instance; `cfg` provides paths for lexicons."""
    if code not in _REGISTRY:
        raise KeyError(f"Unsupported language {code!r}; available: {sorted(_REGISTRY)}")
    if code not in _INSTANCES:
        inst = _REGISTRY[code]()
        _INSTANCES[code] = inst
    inst = _INSTANCES[code]
    if code == "it" and cfg is not None and getattr(inst, "_lexicon_path", None) is None:
        path = find_morphit_file(cfg["paths"]["raw"])
        if path:
            inst.set_lexicon_path(path)  # type: ignore[attr-defined]
    return inst


def available_languages() -> list[str]:
    return sorted(_REGISTRY)


__all__ = ["Language", "get_language", "available_languages"]
