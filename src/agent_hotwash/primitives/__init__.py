"""Reusable, pure detection primitives shared by every detector.

This package is intentionally import-light at the top level: ``events.py``
imports :class:`~agent_hotwash.primitives.filestate.FileState` from a submodule,
so keeping ``__init__`` empty avoids an import cycle.
"""
