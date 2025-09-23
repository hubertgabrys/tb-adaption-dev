import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rtstruct_id import FALLBACK_DATE_STR, create_rtstruct_id, datetime2yymmdd


def test_datetime2yymmdd_returns_fallback_for_invalid_values():
    assert datetime2yymmdd("") == FALLBACK_DATE_STR
    assert datetime2yymmdd(None) == FALLBACK_DATE_STR
    assert datetime2yymmdd("notadate") == FALLBACK_DATE_STR


def test_create_rtstruct_id_handles_missing_study_date():
    metadata = SimpleNamespace(
        Modality="MR", SeriesDescription="t1_mprage", StudyDate=""
    )

    assert create_rtstruct_id(metadata) == f"T1M_{FALLBACK_DATE_STR}"
