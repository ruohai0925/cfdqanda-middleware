"""
Step 020 — Comprehensive unit tests for checkpoint middleware modules.

Tests ControlDictManager, PreRunExecutor, and NormalRunPreparer
using a synthetic OpenFOAM controlDict file in a temp directory.
"""

import os
import sys
import shutil
import tempfile
import unittest

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from controldict_manager import ControlDictManager, _MARKER_BEGIN, _MARKER_END
from pre_run_executor import PreRunExecutor
from normal_run_preparer import NormalRunPreparer

# A minimal but realistic OpenFOAM controlDict for testing
SAMPLE_CONTROLDICT = """\
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    location    "system";
    object      controlDict;
}

application     icoFoam;

startFrom       startTime;

startTime       0;

stopAt          endTime;

endTime         100;

deltaT          0.005;

writeControl    timeStep;

writeInterval   20;

purgeWrite      0;

writeFormat     ascii;

writePrecision  6;

writeCompression off;

timeFormat      general;

timePrecision   6;

runTimeModifiable true;
"""

# controlDict with floating-point endTime
SAMPLE_CONTROLDICT_FLOAT = SAMPLE_CONTROLDICT.replace("endTime         100;", "endTime         0.5;")

# controlDict with scientific notation endTime
SAMPLE_CONTROLDICT_SCI = SAMPLE_CONTROLDICT.replace("endTime         100;", "endTime         1e-3;")


class TestControlDictManager(unittest.TestCase):
    """Tests for ControlDictManager class."""

    def setUp(self):
        """Create a temp case directory with system/controlDict."""
        self.tmp_dir = tempfile.mkdtemp()
        self.case_dir = self.tmp_dir
        self.system_dir = os.path.join(self.case_dir, "system")
        os.makedirs(self.system_dir)
        self.controldict_path = os.path.join(self.system_dir, "controlDict")
        self._write_controldict(SAMPLE_CONTROLDICT)
        self.mgr = ControlDictManager(self.case_dir)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def _write_controldict(self, content):
        with open(self.controldict_path, "w") as f:
            f.write(content)

    def _read_controldict(self):
        with open(self.controldict_path, "r") as f:
            return f.read()

    # --- read_end_time ---

    def test_read_end_time_integer(self):
        result = self.mgr.read_end_time()
        self.assertEqual(result, "100")

    def test_read_end_time_float(self):
        self._write_controldict(SAMPLE_CONTROLDICT_FLOAT)
        result = self.mgr.read_end_time()
        self.assertEqual(result, "0.5")

    def test_read_end_time_scientific(self):
        self._write_controldict(SAMPLE_CONTROLDICT_SCI)
        result = self.mgr.read_end_time()
        self.assertEqual(result, "1e-3")

    def test_read_end_time_missing_file(self):
        os.remove(self.controldict_path)
        with self.assertRaises(FileNotFoundError):
            self.mgr.read_end_time()

    def test_read_end_time_no_entry(self):
        self._write_controldict("FoamFile { }\n// no endTime here\n")
        with self.assertRaises(ValueError):
            self.mgr.read_end_time()

    # --- set_end_time ---

    def test_set_end_time_returns_original(self):
        original = self.mgr.set_end_time("10")
        self.assertEqual(original, "100")

    def test_set_end_time_modifies_file(self):
        self.mgr.set_end_time("10")
        content = self._read_controldict()
        self.assertIn("endTime         10;", content)
        self.assertNotIn("endTime         100;", content)

    def test_set_end_time_roundtrip(self):
        """set → read should return the new value."""
        self.mgr.set_end_time("42")
        result = self.mgr.read_end_time()
        self.assertEqual(result, "42")

    def test_set_end_time_preserves_other_content(self):
        self.mgr.set_end_time("10")
        content = self._read_controldict()
        self.assertIn("application     icoFoam;", content)
        self.assertIn("deltaT          0.005;", content)

    # --- add_function_objects ---

    def test_add_function_objects_inserts_markers(self):
        self.mgr.add_function_objects()
        content = self._read_controldict()
        self.assertIn(_MARKER_BEGIN, content)
        self.assertIn(_MARKER_END, content)

    def test_add_function_objects_includes_fieldMinMax(self):
        self.mgr.add_function_objects()
        content = self._read_controldict()
        self.assertIn("fieldMinMax", content)
        self.assertIn("type            fieldMinMax;", content)

    def test_add_function_objects_includes_fieldAverage(self):
        self.mgr.add_function_objects()
        content = self._read_controldict()
        self.assertIn("fieldAverage", content)
        self.assertIn("type            fieldAverage;", content)

    def test_add_function_objects_idempotent(self):
        """Calling add twice should not duplicate the block."""
        self.mgr.add_function_objects()
        content_first = self._read_controldict()
        self.mgr.add_function_objects()
        content_second = self._read_controldict()
        self.assertEqual(content_first, content_second)

    def test_add_function_objects_before_last_brace(self):
        """The block should appear before the file's final closing brace."""
        self.mgr.add_function_objects()
        content = self._read_controldict()
        marker_pos = content.find(_MARKER_BEGIN)
        last_line_brace = content.rfind("}")
        # The MARKER_END should be before the last closing brace of the original file
        # (the last } belongs to the original FoamFile dict)
        self.assertGreater(marker_pos, 0)

    # --- remove_function_objects ---

    def test_remove_function_objects_cleans_up(self):
        self.mgr.add_function_objects()
        self.mgr.remove_function_objects()
        content = self._read_controldict()
        self.assertNotIn(_MARKER_BEGIN, content)
        self.assertNotIn(_MARKER_END, content)
        self.assertNotIn("fieldMinMax", content)
        self.assertNotIn("fieldAverage", content)

    def test_remove_function_objects_noop_when_absent(self):
        """Should not fail if no markers present."""
        content_before = self._read_controldict()
        self.mgr.remove_function_objects()  # no-op
        content_after = self._read_controldict()
        self.assertEqual(content_before, content_after)

    def test_remove_preserves_original_content(self):
        """After add + remove, key original content should remain."""
        self.mgr.add_function_objects()
        self.mgr.remove_function_objects()
        content = self._read_controldict()
        self.assertIn("application     icoFoam;", content)
        self.assertIn("endTime         100;", content)
        self.assertIn("deltaT          0.005;", content)

    # --- backup / restore ---

    def test_backup_creates_file(self):
        backup_path = self.mgr.backup()
        self.assertTrue(os.path.isfile(backup_path))
        self.assertTrue(backup_path.endswith(".pre-run-backup"))

    def test_restore_from_backup(self):
        self.mgr.backup()
        # Modify the controlDict
        self.mgr.set_end_time("999")
        self.mgr.add_function_objects()
        # Restore
        self.mgr.restore_from_backup()
        content = self._read_controldict()
        self.assertIn("endTime         100;", content)
        self.assertNotIn(_MARKER_BEGIN, content)

    def test_restore_without_backup_raises(self):
        with self.assertRaises(FileNotFoundError):
            self.mgr.restore_from_backup()

    def test_remove_backup(self):
        backup_path = self.mgr.backup()
        self.assertTrue(os.path.isfile(backup_path))
        self.mgr.remove_backup()
        self.assertFalse(os.path.isfile(backup_path))


class TestPreRunExecutor(unittest.TestCase):
    """Tests for PreRunExecutor (prepare phase only, no actual Allrun)."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.case_dir = self.tmp_dir
        self.system_dir = os.path.join(self.case_dir, "system")
        os.makedirs(self.system_dir)
        self.controldict_path = os.path.join(self.system_dir, "controlDict")
        with open(self.controldict_path, "w") as f:
            f.write(SAMPLE_CONTROLDICT)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def test_prepare_returns_correct_data(self):
        executor = PreRunExecutor(self.case_dir, pre_run_end_time=10)
        result = executor.prepare()
        self.assertEqual(result["original_end_time"], "100")
        self.assertEqual(result["pre_run_end_time"], 10)

    def test_prepare_modifies_endtime(self):
        # deltaT=0.005, pre_run_end_time=5 → endTime = 0.005*5 = 0.025
        executor = PreRunExecutor(self.case_dir, pre_run_end_time=5)
        executor.prepare()
        with open(self.controldict_path) as f:
            content = f.read()
        self.assertIn("endTime         0.025;", content)

    def test_prepare_adds_function_objects(self):
        executor = PreRunExecutor(self.case_dir, pre_run_end_time=10)
        executor.prepare()
        with open(self.controldict_path) as f:
            content = f.read()
        self.assertIn(_MARKER_BEGIN, content)
        self.assertIn("fieldMinMax", content)

    def test_prepare_creates_backup(self):
        executor = PreRunExecutor(self.case_dir, pre_run_end_time=10)
        executor.prepare()
        backup = self.controldict_path + ".pre-run-backup"
        self.assertTrue(os.path.isfile(backup))

    def test_execute_without_allrun(self):
        """execute() should return error if Allrun doesn't exist."""
        executor = PreRunExecutor(self.case_dir, pre_run_end_time=10)
        result = executor.execute()
        self.assertFalse(result["success"])
        self.assertIn("Allrun", result["error"])

    def test_collect_results_empty(self):
        """collect_results() with no postProcessing dir should return None values."""
        executor = PreRunExecutor(self.case_dir, pre_run_end_time=10)
        results = executor.collect_results()
        self.assertIsNone(results["field_min_max"])
        self.assertIsNone(results["field_average"])

    def test_collect_results_with_data(self):
        """collect_results() should read postProcessing data if available."""
        executor = PreRunExecutor(self.case_dir, pre_run_end_time=10)

        # Create synthetic postProcessing data
        min_max_dir = os.path.join(self.case_dir, "postProcessing", "fieldMinMax", "0")
        os.makedirs(min_max_dir)
        with open(os.path.join(min_max_dir, "fieldMinMax.dat"), "w") as f:
            f.write("# Time\tp_min\tp_max\n0\t-100\t200\n")

        avg_dir = os.path.join(self.case_dir, "postProcessing", "fieldAverage", "0")
        os.makedirs(avg_dir)
        with open(os.path.join(avg_dir, "fieldAverage.dat"), "w") as f:
            f.write("# Time\tpMean\n0\t50\n")

        results = executor.collect_results()
        self.assertIsNotNone(results["field_min_max"])
        self.assertIn("0", results["field_min_max"])
        self.assertIsNotNone(results["field_average"])
        self.assertIn("0", results["field_average"])


    def test_clean_for_pre_run_removes_runner_outputs(self):
        """_clean_for_pre_run() removes log files, timestep dirs, postProcessing."""
        executor = PreRunExecutor(self.case_dir, pre_run_end_time=10)

        # Create Foam-Agent runner outputs that should be cleaned
        # Log files
        for logname in ("log.blockMesh", "log.icoFoam", "Allrun.out", "Allrun.err"):
            with open(os.path.join(self.case_dir, logname), "w") as f:
                f.write("test log")

        # Timestep directories (0/ should be kept, others removed)
        zero_dir = os.path.join(self.case_dir, "0")
        os.makedirs(zero_dir, exist_ok=True)
        with open(os.path.join(zero_dir, "p"), "w") as f:
            f.write("initial p")
        for ts in ("0.1", "0.2", "0.3", "0.4", "0.5"):
            ts_dir = os.path.join(self.case_dir, ts)
            os.makedirs(ts_dir)
            with open(os.path.join(ts_dir, "p"), "w") as f:
                f.write(f"p at {ts}")

        # postProcessing directory
        pp_dir = os.path.join(self.case_dir, "postProcessing", "fieldMinMax", "0")
        os.makedirs(pp_dir)
        with open(os.path.join(pp_dir, "data.dat"), "w") as f:
            f.write("test data")

        # Run cleanup
        executor._clean_for_pre_run()

        # Verify log files removed
        for logname in ("log.blockMesh", "log.icoFoam", "Allrun.out", "Allrun.err"):
            self.assertFalse(os.path.exists(os.path.join(self.case_dir, logname)),
                             f"{logname} should be removed")

        # Verify timestep dirs removed (except 0/)
        self.assertTrue(os.path.isdir(zero_dir), "0/ should be preserved")
        self.assertTrue(os.path.isfile(os.path.join(zero_dir, "p")),
                        "0/p should be preserved")
        for ts in ("0.1", "0.2", "0.3", "0.4", "0.5"):
            self.assertFalse(os.path.isdir(os.path.join(self.case_dir, ts)),
                             f"{ts}/ should be removed")

        # Verify postProcessing removed
        self.assertFalse(os.path.isdir(os.path.join(self.case_dir, "postProcessing")),
                         "postProcessing/ should be removed")

        # Verify system/ and controlDict preserved
        self.assertTrue(os.path.isfile(self.controldict_path),
                        "controlDict should be preserved")


class TestNormalRunPreparer(unittest.TestCase):
    """Tests for NormalRunPreparer."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.case_dir = self.tmp_dir
        self.system_dir = os.path.join(self.case_dir, "system")
        os.makedirs(self.system_dir)
        self.controldict_path = os.path.join(self.system_dir, "controlDict")

        # Start with a pre-run modified controlDict
        with open(self.controldict_path, "w") as f:
            f.write(SAMPLE_CONTROLDICT)

        # Simulate pre-run state: modify endTime and add function objects
        mgr = ControlDictManager(self.case_dir)
        mgr.set_end_time("10")
        mgr.add_function_objects()

        # Create timestep directories (simulate pre-run output)
        os.makedirs(os.path.join(self.case_dir, "0"))  # initial — keep this
        for t in ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]:
            os.makedirs(os.path.join(self.case_dir, t))

        # Create log files
        for logf in ["log.icoFoam", "log.blockMesh", "Allrun.pre-run.out", "Allrun.pre-run.err"]:
            with open(os.path.join(self.case_dir, logf), "w") as f:
                f.write("sample log content\n")

        # Create postProcessing directory
        pp_dir = os.path.join(self.case_dir, "postProcessing", "fieldMinMax", "0")
        os.makedirs(pp_dir)
        with open(os.path.join(pp_dir, "data.dat"), "w") as f:
            f.write("data\n")

        # Create constant/ and system/ (should not be touched)
        os.makedirs(os.path.join(self.case_dir, "constant"), exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def test_prepare_restores_endtime(self):
        preparer = NormalRunPreparer(self.case_dir, original_end_time="100")
        preparer.prepare()
        mgr = ControlDictManager(self.case_dir)
        self.assertEqual(mgr.read_end_time(), "100")

    def test_prepare_removes_function_objects(self):
        preparer = NormalRunPreparer(self.case_dir, original_end_time="100")
        preparer.prepare()
        with open(self.controldict_path) as f:
            content = f.read()
        self.assertNotIn(_MARKER_BEGIN, content)
        self.assertNotIn("fieldMinMax", content)

    def test_clean_timestep_dirs_keeps_zero(self):
        preparer = NormalRunPreparer(self.case_dir, original_end_time="100")
        preparer.clean_timestep_dirs()
        self.assertTrue(os.path.isdir(os.path.join(self.case_dir, "0")))

    def test_clean_timestep_dirs_removes_nonzero(self):
        preparer = NormalRunPreparer(self.case_dir, original_end_time="100")
        preparer.clean_timestep_dirs()
        for t in ["1", "2", "3", "10"]:
            self.assertFalse(os.path.isdir(os.path.join(self.case_dir, t)))

    def test_clean_timestep_dirs_preserves_named_dirs(self):
        preparer = NormalRunPreparer(self.case_dir, original_end_time="100")
        preparer.clean_timestep_dirs()
        self.assertTrue(os.path.isdir(os.path.join(self.case_dir, "system")))
        self.assertTrue(os.path.isdir(os.path.join(self.case_dir, "constant")))

    def test_clean_log_files(self):
        preparer = NormalRunPreparer(self.case_dir, original_end_time="100")
        preparer.clean_log_files()
        self.assertFalse(os.path.isfile(os.path.join(self.case_dir, "log.icoFoam")))
        self.assertFalse(os.path.isfile(os.path.join(self.case_dir, "Allrun.pre-run.out")))

    def test_clean_post_processing(self):
        preparer = NormalRunPreparer(self.case_dir, original_end_time="100")
        preparer.clean_post_processing()
        self.assertFalse(os.path.isdir(os.path.join(self.case_dir, "postProcessing")))

    def test_full_prepare_workflow(self):
        """Full prepare() should leave the case ready for normal-run."""
        preparer = NormalRunPreparer(self.case_dir, original_end_time="100")
        preparer.prepare()

        # endTime restored
        mgr = ControlDictManager(self.case_dir)
        self.assertEqual(mgr.read_end_time(), "100")

        # No function objects
        with open(self.controldict_path) as f:
            content = f.read()
        self.assertNotIn(_MARKER_BEGIN, content)

        # Only 0/ remains as timestep dir
        self.assertTrue(os.path.isdir(os.path.join(self.case_dir, "0")))
        for t in ["1", "5", "10"]:
            self.assertFalse(os.path.isdir(os.path.join(self.case_dir, t)))

        # No postProcessing
        self.assertFalse(os.path.isdir(os.path.join(self.case_dir, "postProcessing")))

        # system/ and constant/ preserved
        self.assertTrue(os.path.isdir(os.path.join(self.case_dir, "system")))
        self.assertTrue(os.path.isdir(os.path.join(self.case_dir, "constant")))


class TestEndToEndPreRunNormalRun(unittest.TestCase):
    """Integration test: simulate the full pre-run → normal-run prepare cycle."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.case_dir = self.tmp_dir
        self.system_dir = os.path.join(self.case_dir, "system")
        os.makedirs(self.system_dir)
        os.makedirs(os.path.join(self.case_dir, "0"))
        self.controldict_path = os.path.join(self.system_dir, "controlDict")
        with open(self.controldict_path, "w") as f:
            f.write(SAMPLE_CONTROLDICT)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def test_pre_run_prepare_then_normal_run_prepare(self):
        """
        Simulate: PreRunExecutor.prepare() → NormalRunPreparer.prepare()
        controlDict should end up identical to original (minus whitespace).
        """
        original_content = SAMPLE_CONTROLDICT

        # Phase 1: Pre-run prepare
        executor = PreRunExecutor(self.case_dir, pre_run_end_time=10)
        prep_data = executor.prepare()
        self.assertEqual(prep_data["original_end_time"], "100")

        # Verify: endTime changed (deltaT=0.005, 10 steps → 0.05), function objects added
        with open(self.controldict_path) as f:
            modified = f.read()
        self.assertIn("endTime         0.05;", modified)
        self.assertIn(_MARKER_BEGIN, modified)

        # Simulate pre-run producing timestep dirs
        for t in ["1", "2", "3", "4", "5"]:
            os.makedirs(os.path.join(self.case_dir, t))
        os.makedirs(os.path.join(self.case_dir, "postProcessing", "fieldMinMax", "0"), exist_ok=True)

        # Phase 2: Normal-run prepare
        preparer = NormalRunPreparer(self.case_dir, original_end_time="100")
        preparer.prepare()

        # Verify: endTime restored, function objects removed, timesteps cleaned
        with open(self.controldict_path) as f:
            restored = f.read()
        self.assertIn("endTime         100;", restored)
        self.assertNotIn(_MARKER_BEGIN, restored)
        self.assertTrue(os.path.isdir(os.path.join(self.case_dir, "0")))
        self.assertFalse(os.path.isdir(os.path.join(self.case_dir, "5")))
        self.assertFalse(os.path.isdir(os.path.join(self.case_dir, "postProcessing")))


if __name__ == "__main__":
    unittest.main()
