import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pydicom
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

pytest.importorskip("matplotlib")
pytest.importorskip("pandas")
pytest.importorskip("pynetdicom")
pytest.importorskip("SimpleITK")

from usz_artemis_preprocessing.registration import core as registration_core


DATA_DIR = PROJECT_ROOT / "tests" / "data"
TEST_LOG_FILE = PROJECT_ROOT / "tests" / "registration_test_log.csv"
TARGET_SERIES_SUBSTRINGS = (
    "t2_tse_tra",
    "sCT_sp_Pel_T2",
    "t2_trufi3d_tra_p2_ebh",
)


def _discover_series_uid(case_dir: Path, series_substring: str) -> str:
    target = series_substring.lower()
    matches: dict[str, str] = {}

    for dicom_file in case_dir.glob("*.dcm"):
        try:
            ds = pydicom.dcmread(str(dicom_file), stop_before_pixels=True, force=True)
        except Exception:
            continue

        modality = str(getattr(ds, "Modality", "")).strip().upper()
        if modality != "MR":
            continue

        description = str(getattr(ds, "SeriesDescription", "")).strip()
        if target not in description.lower():
            continue

        series_uid = getattr(ds, "SeriesInstanceUID", None)
        if series_uid:
            matches[series_uid] = description

    if not matches:
        pytest.skip(
            f"No MR series containing {series_substring!r} found in {case_dir.name}."
        )

    series_uid, _ = sorted(matches.items(), key=lambda item: item[1])[0]
    return series_uid


def _prepare_case_workspace(tmp_path: Path, case_dir: Path) -> tuple[Path, Path]:
    patient_id = case_dir.name
    rtplan_label = "baseline"
    imaging_source = case_dir / "imaging"
    base_plan_source = case_dir / "base_plan"

    if not imaging_source.is_dir():
        pytest.skip(f"Missing imaging directory in {case_dir.name}.")
    if not base_plan_source.is_dir():
        pytest.skip(f"Missing base_plan directory in {case_dir.name}.")

    current_dir = tmp_path / "current" / patient_id
    baseplan_dir = tmp_path / "baseplans" / patient_id / rtplan_label

    shutil.copytree(imaging_source, current_dir)
    shutil.copytree(base_plan_source, baseplan_dir)

    return current_dir, baseplan_dir


@pytest.mark.registration
@pytest.mark.parametrize(
    "case_dir",
    sorted(path for path in DATA_DIR.iterdir() if path.is_dir()),
    ids=lambda path: path.name,
)
@pytest.mark.parametrize("series_substring", TARGET_SERIES_SUBSTRINGS)
def test_registration_score_for_selected_mr_series(
    tmp_path,
    monkeypatch,
    record_property,
    case_dir: Path,
    series_substring: str,
):
    current_dir, _ = _prepare_case_workspace(tmp_path, case_dir)
    patient_id = case_dir.name
    rtplan_label = "baseline"
    series_uid = _discover_series_uid(current_dir, series_substring)

    monkeypatch.setenv("BASEPLAN_DIR", str(tmp_path / "baseplans"))
    TEST_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(
        registration_core,
        "LOG_FILE",
        TEST_LOG_FILE,
    )

    np.random.seed(0)

    started_at = time.perf_counter()
    transform, score, used_fixed_uid, used_moving_uid, auto_approved = (
        registration_core.perform_registration(
            current_directory=str(current_dir),
            patient_id=patient_id,
            rtplan_label=rtplan_label,
            selected_series_uid=series_uid,
            selected_modality="MR",
            moving_modality="CT",
            confirm_fn=lambda *_: False,
        )
    )
    duration_seconds = round(time.perf_counter() - started_at)

    assert transform is None
    assert used_fixed_uid is None
    assert used_moving_uid is None
    assert auto_approved is False
    assert np.isfinite(score)

    print(
        f"{case_dir.name} | {series_substring} | "
        f"score={score:.4f} | duration={duration_seconds}s"
    )
    record_property("registration_score", float(score))
    record_property("registration_duration_seconds", duration_seconds)
