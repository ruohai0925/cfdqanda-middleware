"""
Execute a short pre-run simulation for checkpoint validation.

This module orchestrates the pre-run phase:
  1. Prepare — backup controlDict, set short endTime, add function objects
  2. Execute — run `bash Allrun` in the case directory
  3. Collect — gather fieldMinMax/fieldAverage outputs for user review

Only uses Python standard library — no external dependencies.
"""

import os
import re
import shutil
import subprocess
import logging
import json

from controldict_manager import ControlDictManager

logger = logging.getLogger(__name__)


class PreRunExecutor:
    """Execute a short pre-run simulation for checkpoint validation."""

    def __init__(self, case_dir: str, pre_run_end_time: int = 10):
        """
        Args:
            case_dir: Path to the OpenFOAM case directory.
            pre_run_end_time: Number of timesteps for pre-run (default 10).
        """
        self.case_dir = case_dir
        self.pre_run_end_time = pre_run_end_time
        self.controldict_mgr = ControlDictManager(case_dir)
        self._original_end_time = None

    def prepare(self) -> dict:
        """
        Prepare case for pre-run.

        Steps:
            1. Backup original controlDict
            2. Read original endTime and deltaT
            3. Compute pre-run endTime = deltaT * pre_run_end_time (timesteps)
            4. Set endTime to the computed value
            5. Add function objects (fieldMinMax + fieldAverage)

        Returns:
            dict with keys: original_end_time, pre_run_end_time,
                            pre_run_physical_time, delta_t
        """
        logger.info(f"Preparing pre-run for case: {self.case_dir}")

        # Backup original controlDict
        self.controldict_mgr.backup()

        # Read original endTime
        self._original_end_time = self.controldict_mgr.read_end_time()
        logger.info(f"Original endTime: {self._original_end_time}")

        # Read deltaT and compute pre-run physical endTime
        # pre_run_end_time is number of timesteps, not physical time
        try:
            delta_t_str = self.controldict_mgr.read_delta_t()
            delta_t = float(delta_t_str)
            pre_run_physical_time = delta_t * self.pre_run_end_time
            # Use a clean string representation
            if pre_run_physical_time == int(pre_run_physical_time):
                physical_time_str = str(int(pre_run_physical_time))
            else:
                physical_time_str = f"{pre_run_physical_time:g}"
            logger.info(
                f"deltaT={delta_t_str}, pre-run timesteps={self.pre_run_end_time}, "
                f"pre-run endTime={physical_time_str}"
            )
        except (ValueError, ZeroDivisionError) as e:
            # Fallback: use pre_run_end_time directly as endTime
            logger.warning(
                f"Could not read deltaT ({e}), using pre_run_end_time={self.pre_run_end_time} "
                "directly as endTime"
            )
            physical_time_str = str(self.pre_run_end_time)
            delta_t_str = "unknown"
            pre_run_physical_time = self.pre_run_end_time

        # Clean up Foam-Agent runner outputs before pre-run
        # Foam-Agent's runner phase already ran the full simulation,
        # so log files and timestep dirs must be removed for pre-run
        # to actually execute with the short endTime.
        self._clean_for_pre_run()

        # Set endTime to computed pre-run value
        self.controldict_mgr.set_end_time(physical_time_str)

        # Add function objects for diagnostics
        self.controldict_mgr.add_function_objects()

        return {
            "original_end_time": self._original_end_time,
            "pre_run_end_time": self.pre_run_end_time,
            "pre_run_physical_time": physical_time_str,
            "delta_t": delta_t_str,
        }

    def _clean_for_pre_run(self) -> None:
        """
        Clean up Foam-Agent runner outputs so the pre-run Allrun
        actually executes instead of being skipped by runApplication.

        Removes:
            - log.* files (runApplication checks these to skip already-run apps)
            - Allrun.out, Allrun.err (Foam-Agent runner output files)
            - Numeric timestep directories except 0/ (previous simulation results)
            - postProcessing/ directory (previous simulation post-processing)
        """
        removed_logs = 0
        removed_dirs = 0

        for entry in os.listdir(self.case_dir):
            entry_path = os.path.join(self.case_dir, entry)

            # Remove log files
            if os.path.isfile(entry_path):
                if (entry.startswith("log.")
                        or entry in ("Allrun.out", "Allrun.err")):
                    os.remove(entry_path)
                    removed_logs += 1
                continue

            # Remove timestep directories (keep 0/)
            if os.path.isdir(entry_path):
                if entry == "postProcessing":
                    shutil.rmtree(entry_path)
                    removed_dirs += 1
                    continue
                try:
                    value = float(entry)
                    if value != 0.0:
                        shutil.rmtree(entry_path)
                        removed_dirs += 1
                except ValueError:
                    pass  # Not a timestep directory

        if removed_logs or removed_dirs:
            logger.info(
                f"Pre-run cleanup: removed {removed_logs} log files, "
                f"{removed_dirs} directories"
            )

    def execute(self, timeout: int = 300) -> dict:
        """
        Execute the pre-run simulation by running `bash Allrun`.

        Args:
            timeout: Maximum seconds to wait for the pre-run (default 300).

        Returns:
            dict with keys: returncode, success, error (if any)
        """
        allrun_path = os.path.join(self.case_dir, "Allrun")
        if not os.path.isfile(allrun_path):
            error_msg = f"Allrun script not found at {allrun_path}"
            logger.error(error_msg)
            return {"returncode": -1, "success": False, "error": error_msg}

        # Ensure Allrun is executable
        os.chmod(allrun_path, 0o755)

        logger.info(f"Executing pre-run: bash Allrun (timeout={timeout}s)")

        # Build command that sources OpenFOAM environment first,
        # matching Foam-Agent's run_command() pattern.
        openfoam_dir = os.environ.get("WM_PROJECT_DIR", "/opt/openfoam10")
        bashrc_path = os.path.join(openfoam_dir, "etc", "bashrc")
        abs_allrun = os.path.abspath(allrun_path)

        if os.path.isfile(bashrc_path):
            command = ["bash", "-c", f"source {bashrc_path} && bash '{abs_allrun}'"]
            logger.info(f"Sourcing OpenFOAM env from {bashrc_path}")
        else:
            # Fallback: run without sourcing (OpenFOAM may already be in PATH)
            command = ["bash", abs_allrun]
            logger.warning(f"OpenFOAM bashrc not found at {bashrc_path}, running without sourcing")

        # Capture output to log files within the case directory
        out_path = os.path.join(self.case_dir, "Allrun.pre-run.out")
        err_path = os.path.join(self.case_dir, "Allrun.pre-run.err")

        try:
            with open(out_path, "w") as out_f, open(err_path, "w") as err_f:
                result = subprocess.run(
                    command,
                    cwd=self.case_dir,
                    stdout=out_f,
                    stderr=err_f,
                    timeout=timeout,
                )

            if result.returncode == 0:
                logger.info("Pre-run completed successfully.")
                return {"returncode": 0, "success": True}
            else:
                logger.warning(f"Pre-run failed with return code {result.returncode}")
                return {
                    "returncode": result.returncode,
                    "success": False,
                    "error": f"Allrun exited with code {result.returncode}",
                }

        except subprocess.TimeoutExpired:
            error_msg = f"Pre-run timed out after {timeout} seconds"
            logger.error(error_msg)
            return {"returncode": -1, "success": False, "error": error_msg}

        except Exception as e:
            error_msg = f"Pre-run execution error: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return {"returncode": -1, "success": False, "error": error_msg}

    def collect_results(self) -> dict:
        """
        Collect pre-run diagnostic outputs.

        Strategy (in order):
            1. If postProcessing/ exists (from function objects), use cellMax/cellMin data
            2. Otherwise, run ``postProcess -func cellMax/cellMin`` to extract field stats
            3. Always parse the solver log for residuals and Courant number

        Returns:
            dict with pre-run diagnostic summary.
        """
        results = {
            "field_min_max": None,
            "field_average": None,
            "solver_log": None,
        }

        # --- Strategy 1: Check existing postProcessing directory ---
        # OpenFOAM 10 uses cellMax/cellMin (volFieldValue), not fieldMinMax
        post_processing_dir = os.path.join(self.case_dir, "postProcessing")
        if os.path.isdir(post_processing_dir):
            # Collect min/max from cellMax and cellMin directories
            cell_max_dir = os.path.join(post_processing_dir, "cellMax")
            cell_min_dir = os.path.join(post_processing_dir, "cellMin")
            min_max_data = {}
            if os.path.isdir(cell_max_dir):
                min_max_data["cellMax"] = self._collect_post_processing_data(cell_max_dir)
                logger.info("Collected cellMax data from postProcessing.")
            if os.path.isdir(cell_min_dir):
                min_max_data["cellMin"] = self._collect_post_processing_data(cell_min_dir)
                logger.info("Collected cellMin data from postProcessing.")
            if min_max_data:
                results["field_min_max"] = min_max_data

            avg_dir = os.path.join(post_processing_dir, "fieldAverage")
            if os.path.isdir(avg_dir):
                results["field_average"] = self._collect_post_processing_data(avg_dir)
                logger.info("Collected fieldAverage data from postProcessing.")
        else:
            logger.warning("No postProcessing directory found after pre-run.")

        # --- Strategy 2: Run postProcess utility if field_min_max is still empty ---
        if results["field_min_max"] is None:
            results["field_min_max"] = self._run_post_process_cell_min_max()

        # --- Strategy 3: Always parse solver log ---
        results["solver_log"] = self._parse_solver_log()

        return results

    def _run_post_process_cell_min_max(self) -> dict:
        """
        Run OpenFOAM ``postProcess -func cellMax`` and ``postProcess -func cellMin``
        to extract field variable min/max values (OpenFOAM 10 compatible).

        This is a fallback used only when the function objects added by
        ``ControlDictManager.add_function_objects()`` did not produce
        a postProcessing/ directory during the pre-run.

        Returns:
            dict with field min/max data, or None on failure.
        """
        openfoam_dir = os.environ.get("WM_PROJECT_DIR", "/opt/openfoam10")
        bashrc_path = os.path.join(openfoam_dir, "etc", "bashrc")

        results = {}
        for func_name in ("cellMax", "cellMin"):
            if os.path.isfile(bashrc_path):
                cmd = f"source {bashrc_path} && postProcess -func {func_name} -latestTime -case '{self.case_dir}'"
            else:
                cmd = f"postProcess -func {func_name} -latestTime -case '{self.case_dir}'"

            try:
                result = subprocess.run(
                    ["bash", "-c", cmd],
                    capture_output=True, text=True, timeout=60,
                )
                if result.returncode != 0:
                    logger.warning(f"postProcess {func_name} failed: {result.stderr[:500]}")
                    continue

                logger.info(f"postProcess {func_name} completed successfully.")

                post_dir = os.path.join(self.case_dir, "postProcessing", func_name)
                if os.path.isdir(post_dir):
                    results[func_name] = self._collect_post_processing_data(post_dir)

            except subprocess.TimeoutExpired:
                logger.warning(f"postProcess {func_name} timed out.")
            except Exception as e:
                logger.warning(f"postProcess {func_name} error: {e}")

        return results if results else None

    def _parse_solver_log(self) -> dict:
        """
        Parse the solver log file to extract residuals, Courant number,
        and continuity errors from the last timestep.

        Returns:
            dict with solver diagnostics, or None if no solver log found.
        """
        # Find the solver log file (log.*, excluding log.blockMesh)
        solver_log = None
        for entry in os.listdir(self.case_dir):
            if entry.startswith("log.") and entry != "log.blockMesh":
                candidate = os.path.join(self.case_dir, entry)
                if os.path.isfile(candidate):
                    solver_log = candidate
                    break

        if not solver_log:
            logger.warning("No solver log file found for diagnostics.")
            return None

        try:
            with open(solver_log, "r") as f:
                content = f.read()
        except Exception as e:
            logger.warning(f"Could not read solver log: {e}")
            return None

        result = {
            "log_file": os.path.basename(solver_log),
            "residuals": {},
            "courant": {},
            "continuity": {},
        }

        # Parse from the end — find the last Time = ... block
        lines = content.splitlines()
        last_time_idx = -1
        for i in range(len(lines) - 1, -1, -1):
            if lines[i].startswith("Time = "):
                last_time_idx = i
                break

        if last_time_idx == -1:
            logger.warning("No 'Time = ...' found in solver log.")
            return result

        # Extract the last timestep value
        time_match = re.match(r'Time = (.+?)s?$', lines[last_time_idx])
        if time_match:
            result["last_time"] = time_match.group(1).strip()

        # Parse lines after the last Time = ... for residuals and Courant
        for line in lines[last_time_idx:]:
            # Residuals: "Solving for Ux, Initial residual = ..., Final residual = ..."
            res_match = re.search(
                r'Solving for (\w+), Initial residual = ([0-9.eE+-]+), '
                r'Final residual = ([0-9.eE+-]+)',
                line
            )
            if res_match:
                field = res_match.group(1)
                result["residuals"][field] = {
                    "initial": float(res_match.group(2)),
                    "final": float(res_match.group(3)),
                }

            # Courant: "Courant Number mean: 0.04 max: 0.18"
            co_match = re.search(
                r'Courant Number mean:\s*([0-9.eE+-]+)\s+max:\s*([0-9.eE+-]+)',
                line
            )
            if co_match:
                result["courant"] = {
                    "mean": float(co_match.group(1)),
                    "max": float(co_match.group(2)),
                }

            # Continuity: "sum local = ..., global = ..., cumulative = ..."
            cont_match = re.search(
                r'continuity errors\s*:\s*sum local = ([0-9.eE+-]+),\s*'
                r'global = ([0-9.eE+-]+),\s*cumulative = ([0-9.eE+-]+)',
                line
            )
            if cont_match:
                result["continuity"] = {
                    "local": float(cont_match.group(1)),
                    "global": float(cont_match.group(2)),
                    "cumulative": float(cont_match.group(3)),
                }

        logger.info(
            f"Parsed solver log: time={result.get('last_time')}, "
            f"{len(result['residuals'])} residual fields, "
            f"Courant max={result['courant'].get('max', 'N/A')}"
        )
        return result

    def _collect_post_processing_data(self, func_dir: str) -> dict:
        """
        Read data files from a postProcessing function directory.

        OpenFOAM writes postProcessing data in the structure:
            postProcessing/<functionName>/<timeDir>/<dataFile>

        Returns:
            dict mapping time directories to their file contents (truncated).
        """
        data = {}
        try:
            for time_dir in sorted(os.listdir(func_dir)):
                time_path = os.path.join(func_dir, time_dir)
                if not os.path.isdir(time_path):
                    continue

                time_data = {}
                for filename in os.listdir(time_path):
                    filepath = os.path.join(time_path, filename)
                    if os.path.isfile(filepath):
                        try:
                            with open(filepath, "r") as f:
                                # Read up to 10KB per file to keep checkpoint_data manageable
                                content = f.read(10240)
                            time_data[filename] = content
                        except Exception as e:
                            time_data[filename] = f"<read error: {e}>"

                if time_data:
                    data[time_dir] = time_data

        except Exception as e:
            logger.warning(f"Error collecting postProcessing data from {func_dir}: {e}")

        return data

    def run(self, timeout: int = 300) -> dict:
        """
        Full pre-run workflow: prepare -> execute -> collect.

        Args:
            timeout: Maximum seconds for the Allrun execution.

        Returns:
            checkpoint_data dict for DB storage, containing:
                - original_end_time
                - pre_run_end_time
                - pre_run_case_dir
                - execution_result
                - diagnostics (fieldMinMax, fieldAverage data)
        """
        # Step 1: Prepare
        prep_data = self.prepare()

        # Step 2: Execute
        exec_result = self.execute(timeout=timeout)

        # Step 3: Collect results (even on failure, there may be partial output)
        diagnostics = self.collect_results()

        checkpoint_data = {
            "original_end_time": prep_data["original_end_time"],
            "pre_run_end_time": prep_data["pre_run_end_time"],
            "pre_run_case_dir": os.path.abspath(self.case_dir),
            "execution_result": exec_result,
            "diagnostics": diagnostics,
        }

        if exec_result["success"]:
            logger.info("Pre-run completed successfully. Checkpoint data ready.")
        else:
            logger.warning(
                f"Pre-run failed: {exec_result.get('error', 'unknown error')}. "
                "Partial diagnostics may be available."
            )

        return checkpoint_data
