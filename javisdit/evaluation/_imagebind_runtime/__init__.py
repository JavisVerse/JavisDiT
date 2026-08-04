"""Lightweight namespace over JavisDiT's vendored ImageBind runtime.

Using a private namespace avoids importing ``javisdit.models.__init__``, which
eagerly initializes every generation-model family in the repository.
"""

from pathlib import Path


_SOURCE_DIRECTORY = Path(__file__).resolve().parents[2] / "models" / "prior_encoder" / "ImageBind"
if not _SOURCE_DIRECTORY.is_dir():
    raise ImportError(f"Vendored ImageBind runtime not found: {_SOURCE_DIRECTORY}")

__path__ = [str(_SOURCE_DIRECTORY)]
