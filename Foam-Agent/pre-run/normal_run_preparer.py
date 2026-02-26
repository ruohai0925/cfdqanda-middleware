"""
Prepare case for normal-run by reverting pre-run modifications.

This module handles the transition from pre-run to normal-run:
  1. Restore original endTime in controlDict
  2. Remove checkpoint function objects
  3. Clean timestep directories (keep 0/)
  4. Clean log files and postProcessing directory

Only uses Python standard library — no external dependencies.
"""

import os
import re
import shutil
import logging

from controldict_manager import ControlDictManager

logger = logging.getLogger(__name__)


class NormalRunPreparer:
    """Prepare case for normal-run by reverting pre-run modifications."""

    def __init__(self, case_dir: str, original_end_time: str):
        """
        Args:
            case_dir: Path to the OpenFOAM case directory.
            original_end_time: The original endTime value to restore.
        """
        self.case_dir = case_dir
        self.original_end_time = original_end_time
        self.controldict_mgr = ControlDictManager(case_dir)

    def prepare(self) -> None:
        """
        Revert all pre-run modifications to prepare for normal-run.

        Steps:
            1. Restore original endTime in controlDict
            2. Remove checkpoint function objects from controlDict
            3. Clean timestep directories (keep only 0/)
            4. Clean log files
            5. Remove postProcessing/ directory
            6. Remove pre-run output files
            7. Remove controlDict backup
        """
        logger.info(f"Preparing normal-run for case: {self.case_dir}")

        # 1. Restore original endTime
        self.controldict_mgr.set_end_time(self.original_end_time)
        logger.info(f"Restored endTime to {self.original_end_time}")

        # 2. Remove checkpoint function objects
        self.controldict_mgr.remove_function_objects()

        # 3. Clean timestep directories
        self.clean_timestep_dirs()

        # 4. Clean log files
        self.clean_log_files()

        # 5. Remove postProcessing directory
        self.clean_post_processing()

        # 6. Remove pre-run output files
        self.clean_pre_run_outputs()

        # 7. Remove backup (no longer needed after restore)
        self.controldict_mgr.remove_backup()

        logger.info("Normal-run preparation complete.")

    def clean_timestep_dirs(self) -> None:
        """
        Remove numeric timestep directories to run from scratch.

        Keeps the 0/ directory (initial conditions) and any non-numeric
        directories (system/, constant/, etc.).

        OpenFOAM timestep directories are named with numbers:
        0/, 0.001/, 0.002/, ..., 1/, 2/, ..., 10/
        We keep ONLY the initial 0/ directory.
        """
        removed_count = 0
        for entry in os.listdir(self.case_dir):
            entry_path = os.path.join(self.case_dir, entry)
            if not os.path.isdir(entry_path):
                continue

            # Check if directory name is a number (integer or float)
            try:
                value = float(entry)
            except ValueError:
                continue  # Not a timestep directory

            # Keep the 0/ directory (initial conditions)
            if value == 0.0:
                continue

            # Remove timestep directory
            shutil.rmtree(entry_path)
            removed_count += 1

        if removed_count > 0:
            logger.info(f"Removed {removed_count} timestep directories.")

    def clean_log_files(self) -> None:
        """
        Remove log files from previous pre-run.

        Targets files matching: log.*, Allrun.out, Allrun.err,
        Allrun.pre-run.out, Allrun.pre-run.err
        """
        removed_count = 0
        for entry in os.listdir(self.case_dir):
            entry_path = os.path.join(self.case_dir, entry)
            if not os.path.isfile(entry_path):
                continue

            # Match log files
            if (entry.startswith("log.")
                    or entry in ("Allrun.out", "Allrun.err",
                                 "Allrun.pre-run.out", "Allrun.pre-run.err")):
                os.remove(entry_path)
                removed_count += 1

        if removed_count > 0:
            logger.info(f"Removed {removed_count} log files.")

    def clean_post_processing(self) -> None:
        """Remove the postProcessing/ directory if it exists."""
        post_dir = os.path.join(self.case_dir, "postProcessing")
        if os.path.isdir(post_dir):
            shutil.rmtree(post_dir)
            logger.info("Removed postProcessing/ directory.")

    def clean_pre_run_outputs(self) -> None:
        """Remove any pre-run specific output files."""
        # Remove processor* directories (from parallel decomposition)
        for entry in os.listdir(self.case_dir):
            entry_path = os.path.join(self.case_dir, entry)
            if os.path.isdir(entry_path) and entry.startswith("processor"):
                shutil.rmtree(entry_path)
                logger.info(f"Removed {entry}/ directory.")
