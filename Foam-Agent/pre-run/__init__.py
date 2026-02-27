"""
CFDQandA Checkpoint Middleware — Foam-Agent Pre-Run Module.

Provides two-phase simulation support (pre-run + full-run)
for the controlled pipeline mode, without modifying Foam-Agent source code.

Used by worker.py's ``_mcp_stage_pre_run()`` (PreRunExecutor) and
``_mcp_stage_full_run()`` (NormalRunPreparer).
"""

from .controldict_manager import ControlDictManager
from .pre_run_executor import PreRunExecutor
from .normal_run_preparer import NormalRunPreparer

__all__ = ["ControlDictManager", "PreRunExecutor", "NormalRunPreparer"]
