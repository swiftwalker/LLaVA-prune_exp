"""Process-local compatibility shims for FastV official code.

The FastV checkout vendors a modified Transformers tree that pins
``tokenizers<0.14``. The local LLaVA environment uses a newer tokenizers build.
For wrapper runs we keep the official source untouched and only relax the
runtime version check inside this process.
"""

from __future__ import annotations

import importlib.metadata
import os


if os.environ.get("FASTV_ALLOW_TOKENIZERS_015", "1") == "1":
    _real_version = importlib.metadata.version

    def _compat_version(distribution_name: str) -> str:
        normalized = distribution_name.lower().replace("_", "-")
        if normalized == "tokenizers":
            return "0.13.3"
        return _real_version(distribution_name)

    importlib.metadata.version = _compat_version
