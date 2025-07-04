import datetime
import sys

import pydicom
from dotenv import load_dotenv


ALLOWED_SERIES = ['SyntheticCT HU', 'sCTp1-Dixon-HR_in', 't2_tse_tra']


def get_datetime():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")



def find_series_uids(dir_path):
    """
    Get a dictionary of unique SeriesInstanceUIDs from DICOM files in a directory.
    Each key in the dictionary maps to a list of file paths that share that SeriesInstanceUID.

    Args:
        dir_path (str): Path to the directory containing DICOM files.

    Returns:
        Dict[str, List[str]]: Dictionary where each key is a SeriesInstanceUID and
                              each value is a list of filepaths corresponding to that UID.
    """
    series_instance_uids = {}

    # Loop through all files in the directory
    for filename in os.listdir(dir_path):
        filepath = os.path.join(dir_path, filename)

        # Try reading the file as a DICOM file
        try:
            # Only read SeriesInstanceUID (avoid reading unnecessary data/pixels)
            dataset = pydicom.dcmread(
                filepath,
                specific_tags=['SeriesInstanceUID', 'SeriesDescription'],
                stop_before_pixels=True
            )

            # series_desc = dataset.get("SeriesDescription", "")
            # if not any(allowed in series_desc for allowed in ALLOWED_SERIES):
            #     continue

            # If the SeriesInstanceUID attribute exists in the dataset
            if 'SeriesInstanceUID' in dataset:
                uid = dataset.SeriesInstanceUID

                # Initialize the list if this UID is not yet in the dictionary
                if uid not in series_instance_uids:
                    series_instance_uids[uid] = []

                # Append this file's path under the appropriate UID
                series_instance_uids[uid].append(filepath)

        except Exception as e:
            # Handle non-DICOM files or errors in reading
            print(f"Skipping {filename}: {e}")

    return series_instance_uids


import os


def check_if_ct_present(directory):
    """
    Check if at least one filename in the directory starts with 'CT'.

    Args:
        directory (str): The path to the directory.

    Returns:
        bool: True if at least one file starts with 'CT', False otherwise.
    """
    for filename in os.listdir(directory):
        # Ensure it's a file, not a directory
        if os.path.isfile(os.path.join(directory, filename)):
            if filename.startswith("CT"):
                return True
    return False


def load_environment(env_file_path: str = ".env"):
    if getattr(sys, "frozen", False):
        # running as bundled exe
        base_dir = os.path.dirname(sys.executable)
    else:
        # running as normal script
        base_dir = os.path.dirname(os.path.abspath(__file__))
    dotenv_path = os.path.join(base_dir, env_file_path)
    load_dotenv(dotenv_path)