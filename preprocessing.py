import datetime
import os
import shutil

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

def list_imaging_series(dir_path: str) -> dict:
    """
    Scan dir_path for DICOM files, group them by SeriesInstanceUID,
    print each series (date, time, description, count),
    and return a dict:
      { uid: { date, time, description, files: [file1, file2, …] } }
    """
    if not os.path.exists(dir_path):
        print(f"Directory does not exist: {dir_path}")
        return {}
    if not os.path.isdir(dir_path):
        print(f"Path is not a directory: {dir_path}")
        return {}
    if not os.listdir(dir_path):
        print(f"Directory is empty: {dir_path}")
        return {}

    series_info = {}

    for root, _, files in os.walk(dir_path):
        for fname in files:
            fpath = os.path.join(root, fname)
            try:
                ds = pydicom.dcmread(fpath, stop_before_pixels=True, force=True)
            except Exception:
                continue

            uid = getattr(ds, "SeriesInstanceUID", None)
            if not uid:
                continue

            if uid not in series_info:
                date = getattr(ds, "SeriesDate",
                               getattr(ds, "StudyDate", ""))
                time = getattr(ds, "SeriesTime",
                               getattr(ds, "StudyTime", ""))
                if time and len(time) >= 4:
                    hh, mm = time[:2], time[2:4]
                    ss = f":{time[4:6]}" if len(time) >= 6 else ""
                    time = f"{hh}:{mm}{ss}"
                desc = getattr(ds, "SeriesDescription", "").strip() or "<no description>"
                modality = getattr(ds, "Modality", "").strip() or "<no modality>"

                series_info[uid] = {
                    "date": date,
                    "time": time,
                    "modality": modality,
                    "description": desc,
                    "files": []
                }

            series_info[uid]["files"].append(fpath)

    if not series_info:
        print(f"No DICOM series found in directory: {dir_path}")
        return {}

    for info in series_info.values():
        count = len(info["files"])
        print(f"{info['date']} {info['time']} – {info['modality']} - {info['description']} "
              f"(number of files = {count})")

    return series_info

def confirm_and_delete_old_series(series_info: dict):
    """
    Given the dict from list_imaging_series, finds all files whose
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
    series = list_imaging_series(directory_path)
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
