import os
import socket

import pydicom
from pynetdicom import AE
from pynetdicom.sop_class import CTImageStorage, MRImageStorage, RTStructureSetStorage, Verification, SpatialRegistrationStorage
from tqdm import tqdm

from utils import load_environment

# Load the .env
load_environment(".env")

# DICOM server configuration
SERVER_AE_TITLE = os.environ.get('SERVER_AE_TITLE')
SERVER_IP = os.environ.get('SERVER_IP')
SERVER_PORT = int(os.environ.get('SERVER_PORT'))

ALLOWED_SERIES = ['SyntheticCT HU', 'sCTp1-Dixon-HR_in', 't2_tse_tra', "Spatial Registration (Rigid)"]


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


def send2aria(dir_path):
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
        print("Failed to establish association for a thread.")
        return 0

    image_files, rtstruct_files, reg_files = get_dicom_files(dir_path)
    # print()
    print("Image files")
    for series in ALLOWED_SERIES:
        # Filter image files that include the allowed series string in their series description.
        series_files = [f for f in image_files if series in f[1]]  # f[1] is series_desc
        if series_files:
            for f, _ in tqdm(series_files, desc=series):
                ds = pydicom.dcmread(f)
                status = send_file(assoc, ds)
                if status and status.Status == 0x0000:
                    os.remove(f)
                else:
                    print(f"Failed to send file: {f}")

    # print()
    print("RTStruct files")
    for series in ALLOWED_SERIES:
        # Filter RTStruct files that include the allowed series string in their series description.
        series_files = [f for f in rtstruct_files if series in f[1]]
        if series_files:
            for f, _ in tqdm(series_files, desc=series):
                ds = pydicom.dcmread(f)
                status = send_file(assoc, ds)
                if status and status.Status == 0x0000:
                    os.remove(f)
                else:
                    print(f"Failed to send file: {f}")

    print("REG files")
    for series in ALLOWED_SERIES:
        # Filter RTStruct files that include the allowed series string in their series description.
        series_files = [f for f in reg_files if series in f[1]]
        if series_files:
            for f, _ in tqdm(series_files, desc=series):
                ds = pydicom.dcmread(f)
                status = send_file(assoc, ds)
                if status and status.Status == 0x0000:
                    os.remove(f)
                else:
                    print(f"Failed to send file: {f}")

    assoc.release()
