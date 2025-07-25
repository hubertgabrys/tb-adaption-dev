import os
import socket

import pydicom
from pynetdicom import AE
from pynetdicom.sop_class import CTImageStorage, MRImageStorage, RTStructureSetStorage, Verification, SpatialRegistrationStorage
from tqdm import tqdm

from utils import load_environment
from concurrent.futures import ThreadPoolExecutor
import math
import threading

# Load the .env
load_environment(".env")

# DICOM server configuration
SERVER_AE_TITLE = os.environ.get('SERVER_AE_TITLE')
SERVER_IP = os.environ.get('SERVER_IP')
SERVER_PORT = int(os.environ.get('SERVER_PORT'))

ALLOWED_SERIES = [
    'SyntheticCT HU',
    'sCTp1-Dixon-HR_in',
    't2_tse_tra',
    'Spatial Registration (Rigid)',
]


def get_dicom_files(dir_path):
    """
    Retrieve and categorize DICOM files from a directory.
    Returns lists of tuples (filepath, series_desc) for image and RTStruct files.
    """
    dicom_files = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith(".dcm")]
    image_files = []
    rtstruct_files = []
    reg_files = []
    for filepath in tqdm(dicom_files, desc="Identifying files to send"):
        dataset = pydicom.dcmread(filepath, stop_before_pixels=True)
        sop_class = dataset.SOPClassUID
        series_desc = dataset.get("SeriesDescription", "")

        if not any(allowed in series_desc for allowed in ALLOWED_SERIES):
            continue

        if sop_class in (CTImageStorage, MRImageStorage):
            image_files.append((filepath, series_desc))
        elif sop_class == RTStructureSetStorage:
            rtstruct_files.append((filepath, series_desc))
        elif sop_class == SpatialRegistrationStorage:
            reg_files.append((filepath, series_desc))

    return image_files, rtstruct_files, reg_files


def send_file(assoc, ds):
    status = assoc.send_c_store(ds)
    return status


def _send_subset(file_subset, lock, progress, total, progress_callback):
    ae = AE(socket.gethostname())
    ae.add_requested_context(CTImageStorage)
    ae.add_requested_context(MRImageStorage)
    ae.add_requested_context(RTStructureSetStorage)
    ae.add_requested_context(Verification)
    ae.add_requested_context(SpatialRegistrationStorage)
    ae.acse_timeout = 1200
    ae.dimse_timeout = 1200
    ae.network_timeout = 1200

    assoc = ae.associate(SERVER_IP, SERVER_PORT, ae_title=SERVER_AE_TITLE)
    if not assoc.is_established:
        print("Failed to establish association.")
        return False

    success = True
    for fpath in file_subset:
        ds = pydicom.dcmread(fpath)
        status = send_file(assoc, ds)
        if status and status.Status == 0x0000:
            try:
                os.remove(fpath)
            except Exception:
                pass
        else:
            print(f"Failed to send file: {fpath}")
            success = False
        with lock:
            progress[0] += 1
            if progress_callback:
                progress_callback(progress[0], total)

    assoc.release()
    return success


def send_files_to_aria(filepaths, progress_callback=None, num_workers=4):
    """Send the DICOM *filepaths* to the ARIA server using multiple associations."""

    if num_workers < 1:
        num_workers = 1

    total = len(filepaths)
    if total == 0:
        return True

    lock = threading.Lock()
    progress = [0]
    chunk_size = int(math.ceil(total / float(num_workers)))
    subsets = [filepaths[i:i + chunk_size] for i in range(0, total, chunk_size)]

    results = []
    with ThreadPoolExecutor(max_workers=len(subsets)) as executor:
        futures = [executor.submit(_send_subset, subset, lock, progress, total, progress_callback)
                   for subset in subsets]
        for fut in futures:
            results.append(fut.result())

    return all(results)


def send2aria(dir_path):
    """Send all allowed series in *dir_path* to the ARIA server."""
    image_files, rtstruct_files, reg_files = get_dicom_files(dir_path)

    filepaths = [
        *(f for f, _ in image_files),
        *(f for f, _ in rtstruct_files),
        *(f for f, _ in reg_files),
    ]

    return send_files_to_aria(filepaths)
