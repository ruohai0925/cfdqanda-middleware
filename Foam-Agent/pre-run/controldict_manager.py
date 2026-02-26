"""
Manage OpenFOAM controlDict modifications for the checkpoint mechanism.

This module reads, modifies, and restores the system/controlDict file
to support two-phase simulation (pre-run + normal-run).

Only uses Python standard library — no external dependencies.
"""

import os
import re
import shutil
import logging

logger = logging.getLogger(__name__)

# Marker comments used to identify checkpoint-injected function objects
_MARKER_BEGIN = "// --- CFDQANDA_CHECKPOINT_FUNCTIONS_BEGIN ---"
_MARKER_END = "// --- CFDQANDA_CHECKPOINT_FUNCTIONS_END ---"

# Function objects template for pre-run diagnostics (OpenFOAM 10 compatible).
# Uses volFieldValue (cellMax/cellMin) instead of fieldMinMax which doesn't exist in v10.
_FUNCTION_OBJECTS_TEMPLATE = """\
// --- CFDQANDA_CHECKPOINT_FUNCTIONS_BEGIN ---
functions
{
    cellMax
    {
        type            volFieldValue;
        libs            ("libfieldFunctionObjects.so");
        writeControl    timeStep;
        writeInterval   1;
        writeFields     false;
        regionType      all;
        operation       max;
        fields          (p U);
    }

    cellMin
    {
        type            volFieldValue;
        libs            ("libfieldFunctionObjects.so");
        writeControl    timeStep;
        writeInterval   1;
        writeFields     false;
        regionType      all;
        operation       min;
        fields          (p U);
    }

    fieldAverage
    {
        type            fieldAverage;
        libs            ("libfieldFunctionObjects.so");
        fields
        (
            p
            {
                mean        on;
                prime2Mean  off;
                base        time;
            }
            U
            {
                mean        on;
                prime2Mean  off;
                base        time;
            }
        );
        enabled         true;
        log             true;
        writeControl    timeStep;
        writeInterval   1;
    }
}
// --- CFDQANDA_CHECKPOINT_FUNCTIONS_END ---"""


class ControlDictManager:
    """Manage OpenFOAM controlDict modifications for checkpoint mechanism."""

    def __init__(self, case_dir: str):
        """
        Initialize with the OpenFOAM case directory path.

        Args:
            case_dir: Path to the OpenFOAM case directory (containing system/, constant/, 0/).
        """
        self.case_dir = case_dir
        self.controldict_path = os.path.join(case_dir, "system", "controlDict")
        self._backup_path = self.controldict_path + ".pre-run-backup"

    def _read_file(self) -> str:
        """Read the controlDict file content."""
        if not os.path.isfile(self.controldict_path):
            raise FileNotFoundError(
                f"controlDict not found at {self.controldict_path}"
            )
        with open(self.controldict_path, "r") as f:
            return f.read()

    def _write_file(self, content: str) -> None:
        """Write content to the controlDict file."""
        with open(self.controldict_path, "w") as f:
            f.write(content)

    def read_end_time(self) -> str:
        """
        Parse endTime from controlDict.

        Handles formats like:
            endTime     100;
            endTime 0.5;
            endTime     1e-3;

        Returns:
            The endTime value as a string (e.g. "100", "0.5", "1e-3").

        Raises:
            FileNotFoundError: If controlDict does not exist.
            ValueError: If endTime entry cannot be found.
        """
        content = self._read_file()
        match = re.search(r'^\s*endTime\s+([^;]+?)\s*;', content, re.MULTILINE)
        if not match:
            raise ValueError(
                f"Could not find 'endTime' entry in {self.controldict_path}"
            )
        return match.group(1).strip()

    def read_delta_t(self) -> str:
        """
        Parse deltaT from controlDict.

        Returns:
            The deltaT value as a string (e.g. "0.005", "1e-4").

        Raises:
            FileNotFoundError: If controlDict does not exist.
            ValueError: If deltaT entry cannot be found.
        """
        content = self._read_file()
        match = re.search(r'^\s*deltaT\s+([^;]+?)\s*;', content, re.MULTILINE)
        if not match:
            raise ValueError(
                f"Could not find 'deltaT' entry in {self.controldict_path}"
            )
        return match.group(1).strip()

    def set_end_time(self, new_end_time: str) -> str:
        """
        Set endTime in controlDict to a new value.

        Args:
            new_end_time: The new endTime value (e.g. "10", "0.001").

        Returns:
            The original endTime value before modification.

        Raises:
            FileNotFoundError: If controlDict does not exist.
            ValueError: If endTime entry cannot be found.
        """
        content = self._read_file()
        match = re.search(r'^(\s*endTime\s+)([^;]+?)\s*(;)', content, re.MULTILINE)
        if not match:
            raise ValueError(
                f"Could not find 'endTime' entry in {self.controldict_path}"
            )

        original_end_time = match.group(2).strip()
        new_content = (
            content[:match.start()]
            + match.group(1) + new_end_time + match.group(3)
            + content[match.end():]
        )
        self._write_file(new_content)
        logger.info(
            f"controlDict endTime changed: {original_end_time} -> {new_end_time}"
        )
        return original_end_time

    def add_function_objects(self) -> None:
        """
        Add fieldMinMax and fieldAverage function objects to controlDict.

        The function objects are wrapped with marker comments so they can be
        precisely removed later by remove_function_objects().

        If the marker is already present, this is a no-op (idempotent).

        The block is appended at the end of the file.
        """
        content = self._read_file()

        # Check if already added (idempotent)
        if _MARKER_BEGIN in content:
            logger.info("Checkpoint function objects already present, skipping.")
            return

        # Append function objects at the end of the file.
        # This works for both standard OpenFOAM controlDict (top-level braces)
        # and LLM-generated flat format (FoamFile {} + top-level keywords).
        new_content = (
            content.rstrip()
            + "\n\n"
            + _FUNCTION_OBJECTS_TEMPLATE
            + "\n"
        )
        self._write_file(new_content)
        logger.info("Added checkpoint function objects (fieldMinMax + fieldAverage).")

    def remove_function_objects(self) -> None:
        """
        Remove checkpoint function objects that were previously added.

        Uses the marker comments to locate and remove the exact block.
        If no markers are found, this is a no-op.
        """
        content = self._read_file()

        begin_idx = content.find(_MARKER_BEGIN)
        end_idx = content.find(_MARKER_END)

        if begin_idx == -1 or end_idx == -1:
            logger.info("No checkpoint function objects markers found, nothing to remove.")
            return

        # Remove the entire block including markers, plus surrounding blank lines
        end_idx += len(_MARKER_END)
        before = content[:begin_idx].rstrip("\n")
        after = content[end_idx:].lstrip("\n")
        new_content = before + "\n\n" + after
        self._write_file(new_content)
        logger.info("Removed checkpoint function objects.")

    def backup(self) -> str:
        """
        Create a backup copy of controlDict.

        Returns:
            The backup file path.
        """
        if not os.path.isfile(self.controldict_path):
            raise FileNotFoundError(
                f"controlDict not found at {self.controldict_path}"
            )
        shutil.copy2(self.controldict_path, self._backup_path)
        logger.info(f"controlDict backed up to {self._backup_path}")
        return self._backup_path

    def restore_from_backup(self) -> None:
        """
        Restore controlDict from the backup file.

        Raises:
            FileNotFoundError: If the backup file does not exist.
        """
        if not os.path.isfile(self._backup_path):
            raise FileNotFoundError(
                f"Backup file not found at {self._backup_path}"
            )
        shutil.copy2(self._backup_path, self.controldict_path)
        logger.info(f"controlDict restored from {self._backup_path}")

    def remove_backup(self) -> None:
        """Remove the backup file if it exists."""
        if os.path.isfile(self._backup_path):
            os.remove(self._backup_path)
            logger.info(f"Removed backup file {self._backup_path}")
