"""
gazelle package

This repo uses a `scripts/` entrypoint that imports `gazelle.*`.
Adding this file ensures the directory is recognized as a Python package
when installed (e.g., `pip install -e .`) or when used via PYTHONPATH.
"""

__all__ = [
    "backbone",
    "dataloader",
    "model",
    "utils",
]




