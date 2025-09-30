import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pydicom

from utils import get_datetime


def get_file_path(directory_path: str, filename: str) -> str:
    """Construct the full file path from directory path and filename."""
    return os.path.join(directory_path, filename)


def remove_file(file_path: str) -> None:
    """Remove the file located at the given file path."""
    os.remove(file_path)


def rename_file_by_modality(directory_path: str, filename: str, modality: str) -> None:
    """
    Rename the file by prepending the modality if the filename does not already start with it.
    """
    if modality == "RTSTRUCT":
        modality = "RS"
    if not filename.startswith(modality):
        new_filename = f"{modality}_{filename}"
        old_filepath = get_file_path(directory_path, filename)
        new_filepath = get_file_path(directory_path, new_filename)
        os.rename(old_filepath, new_filepath)


def _get_referenced_series(ds) -> set[str]:
    """Return a set of SeriesInstanceUIDs referenced by *ds*."""
    referenced = set()
    modality = getattr(ds, "Modality", "")
    if modality == "RTSTRUCT":
        try:
            for fr in ds.ReferencedFrameOfReferenceSequence:
                for st in fr.RTReferencedStudySequence:
                    for se in st.RTReferencedSeriesSequence:
                        uid = getattr(se, "SeriesInstanceUID", None)
                        if uid:
                            referenced.add(uid)
        except Exception:
            pass
    elif modality == "REG":
        try:
            if hasattr(ds, "ReferencedSeriesSequence"):
                for item in ds.ReferencedSeriesSequence:
                    uid = getattr(item, "SeriesInstanceUID", None)
                    if uid:
                        referenced.add(uid)
            if hasattr(ds, "StudiesContainingOtherReferencedInstancesSequence"):
                for study in ds.StudiesContainingOtherReferencedInstancesSequence:
                    if hasattr(study, "ReferencedSeriesSequence"):
                        for item in study.ReferencedSeriesSequence:
                            uid = getattr(item, "SeriesInstanceUID", None)
                            if uid:
                                referenced.add(uid)
        except Exception:
            pass
    return referenced


def process_single_dicom_file(directory_path: str, filename: str) -> None:
    """
    Process a single DICOM file:
    - Remove the file if its series description is not allowed or if the Modality attribute is missing.
    - Otherwise, rename it by adding the modality as a prefix.
    """
    file_path = get_file_path(directory_path, filename)
    try:
        ds = pydicom.dcmread(file_path, stop_before_pixels=True)
        series_description = ds.get("SeriesDescription", "")
        modality = ds.get("Modality")
        # If series is not allowed or modality is missing, remove the file.
        # if not is_series_allowed(series_description) or not modality:
        if not modality:
            remove_file(file_path)
            return
        rename_file_by_modality(directory_path, filename, modality)
    except Exception:
        # Skip files that are not valid DICOM.
        pass


def _extract_series_record(fpath: Path, imaging_only: bool):
    """Read just enough of the header to form one series entry, or None."""
    try:
        ds = pydicom.dcmread(str(fpath), stop_before_pixels=True, force=True)
    except Exception:
        return None

    modality = getattr(ds, "Modality", "").strip() or "<no modality>"
    if imaging_only and modality in ("REG", "RTSTRUCT"):
        return None

    uid = getattr(ds, "SeriesInstanceUID", None)
    if not uid:
        return None

    date = getattr(ds, "SeriesDate", getattr(ds, "StudyDate", ""))
    time = getattr(ds, "SeriesTime", getattr(ds, "StudyTime", ""))
    # normalize time
    if time and len(time) >= 4:
        hh, mm = time[:2], time[2:4]
        ss = f":{time[4:6]}" if len(time) >= 6 else ""
        time = f"{hh}:{mm}{ss}"
    desc = getattr(ds, "SeriesDescription", "").strip() or "<no description>"

    record = {
        "uid": uid,
        "date": date,
        "time": time,
        "modality": modality,
        "description": desc,
        "file": str(fpath)
    }

    if modality in ("RTSTRUCT", "REG"):
        record["references"] = list(_get_referenced_series(ds))

    return record

def list_dicom_series(dir_path: str, imaging_only: bool = False) -> dict:
    print(f"{get_datetime()} Listing "
          f"{'imaging ' if imaging_only else 'all '}DICOM series in: {dir_path}")

    root = Path(dir_path)
    if not root.is_dir() or not any(root.iterdir()):
        print(f"Directory is missing or empty: {dir_path}")
        return {}

    # 1) gather all .dcm files recursively
    dcm_files = list(root.rglob("*.dcm"))

    # 2) read headers in parallel
    records = []
    with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as pool:
        futures = [
            pool.submit(_extract_series_record, p, imaging_only)
            for p in dcm_files
        ]
        for fut in as_completed(futures):
            rec = fut.result()
            if rec:
                records.append(rec)

    # 3) merge into series_info
    series_info = {}
    for rec in records:
        uid = rec["uid"]
        if uid not in series_info:
            series_info[uid] = {
                "date":       rec["date"],
                "time":       rec["time"],
                "modality":   rec["modality"],
                "description":rec["description"],
                "files":      [],
            }
            if "references" in rec:
                series_info[uid]["references"] = set(rec["references"])
        else:
            if "references" in rec:
                series_info[uid].setdefault("references", set()).update(rec["references"])
        series_info[uid]["files"].append(rec["file"])

    if not series_info:
        print(f"No DICOM series found in directory: {dir_path}")
        return {}

    # 4) log summary
    for info in series_info.values():
        cnt = len(info["files"])
        print(f"  {info['date']} {info['time']} – {info['modality']} – "
              f"{info['description']} (files={cnt})")

    # convert any reference sets to lists for easier consumption
    for info in series_info.values():
        if "references" in info:
            info["references"] = list(info["references"])

    return series_info
