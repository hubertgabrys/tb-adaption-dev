import datetime
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pydicom
import pyzipper

from utils import get_datetime

ALLOWED_SERIES_KEYWORDS = ['SyntheticCT HU', 'sCTp1-Dixon-HR_in', 't2_tse_tra']


def is_series_allowed(series_description: str) -> bool:
    """Check if the series description contains any allowed series keyword."""
    return any(keyword in series_description for keyword in ALLOWED_SERIES_KEYWORDS)


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


def copy_directory(src, dst):
    """
    Copies the entire directory tree from src to dst.
    If the destination directory exists, it will be overwritten in Python 3.8+.
    """
    try:
        # Using dirs_exist_ok to allow copying into an existing directory
        shutil.copytree(src, dst, dirs_exist_ok=True)
        # print(f"Directory copied from {src} to {dst}")
    except Exception as e:
        print(f"Error: {e}")


def zip_directory(directory_path, output_zip_path, password=None):
    if password:
        password_bytes = password.encode('utf-8')
        zipf = pyzipper.AESZipFile(output_zip_path, 'w', compression=pyzipper.ZIP_DEFLATED, encryption=pyzipper.WZ_AES)
        zipf.setpassword(password_bytes)
    else:
        zipf = pyzipper.AESZipFile(output_zip_path, 'w', compression=pyzipper.ZIP_DEFLATED)

    # Use a try-finally block to ensure the file is properly closed after writing
    try:
        base_dir_name = os.path.basename(os.path.normpath(directory_path))
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                file_path = os.path.join(root, file)
                # Calculate relative path within the directory_path
                rel_path = os.path.relpath(file_path, start=directory_path)
                # Prepend the base directory name so the structure in the archive is:
                # XXX/file1, XXX/file2, etc.
                arcname = os.path.join(base_dir_name, rel_path)
                zipf.write(file_path, arcname)
    finally:
        zipf.close()

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

    return {
        "uid": uid,
        "date": date,
        "time": time,
        "modality": modality,
        "description": desc,
        "file": str(fpath)
    }

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
                "files":      []
            }
        series_info[uid]["files"].append(rec["file"])

    if not series_info:
        print(f"No DICOM series found in directory: {dir_path}")
        return {}

    # 4) log summary
    for info in series_info.values():
        cnt = len(info["files"])
        print(f"  {info['date']} {info['time']} – {info['modality']} – "
              f"{info['description']} (files={cnt})")

    return series_info

def confirm_and_delete_old_series(series_info: dict):
    """
    Given the dict from ``list_dicom_series``, finds all files whose
    SeriesDate is not today, prompts the user, and deletes them if
    the user confirms.
    """
    if not series_info:
        # nothing to do
        return

    today_str = datetime.datetime.now().strftime("%Y%m%d")
    files_to_delete = []

    for info in series_info.values():
        if info["date"] != today_str:
            files_to_delete.extend(info["files"])

    if not files_to_delete:
        print("All DICOM files have today's SeriesDate; nothing to delete.")
        return

    print(f"Found {len(files_to_delete)} DICOM file(s) with SeriesDate ≠ {today_str}.")
    resp = input("Delete these file(s)? (y/n): ").strip().lower()
    if resp == "y":
        deleted = 0
        for fpath in files_to_delete:
            try:
                os.remove(fpath)
                deleted += 1
            except Exception as e:
                print(f"  Failed to delete {fpath}: {e}")
        print(f"Deleted {deleted} file(s).")
    else:
        print("No files were deleted.")


def check_input_directory(directory_path: str) -> None:
    series = list_dicom_series(directory_path, imaging_only=False)
    confirm_and_delete_old_series(series)

def preprocess_input_directory(directory_path: str) -> None:
    """
    Process each file in the directory by renaming based on the DICOM modality
    tag if applicable, or removing the file if it does not meet criteria.
    """
    print(f"{get_datetime()} Backing up images into a zip file")
    # directory_path_new = directory_path+"_processed"
    # copy_directory(directory_path, directory_path_new)
    zip_directory(directory_path, directory_path+".zip")
    print(f"{get_datetime()} Renaming images (if neccessary)")
    for filename in os.listdir(directory_path):
        file_path = get_file_path(directory_path, filename)
        if os.path.isfile(file_path):
            process_single_dicom_file(directory_path, filename)
