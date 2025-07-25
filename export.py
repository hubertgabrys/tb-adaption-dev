import os
import socket

import pydicom
from pydicom.uid import ImplicitVRLittleEndian
from pynetdicom import AE
from pynetdicom.sop_class import CTImageStorage, MRImageStorage, RTStructureSetStorage, Verification, SpatialRegistrationStorage

from utils import load_environment

# Load the .env
load_environment(".env")

# DICOM server configuration
SERVER_AE_TITLE = os.environ.get('SERVER_AE_TITLE')
SERVER_IP = os.environ.get('SERVER_IP')
SERVER_PORT = int(os.environ.get('SERVER_PORT'))

TXS = [ImplicitVRLittleEndian]

SOPS = [
    CTImageStorage,
    MRImageStorage,
    RTStructureSetStorage,
    SpatialRegistrationStorage
]


def send_file(assoc, ds):
    status = assoc.send_c_store(ds)
    return status


def send_files_to_aria(filepaths, progress_callback=None):
    """Send the DICOM *filepaths* to the ARIA server."""
    ae = AE(socket.gethostname())

    for sop in SOPS:
        ae.add_requested_context(sop, TXS)
    ae.add_requested_context(Verification)

    ae.maximum_pdu_size = 16 * 1024 * 1024
    ae.acse_timeout = 120
    ae.dimse_timeout = 120
    ae.network_timeout = 120

    assoc = ae.associate(SERVER_IP, SERVER_PORT, ae_title=SERVER_AE_TITLE)
    if not assoc.is_established:
        print("Failed to establish association.")
        return False

    success = True

    imaging = []
    rtstruct = []
    registration = []

    for fpath in filepaths:
        try:
            ds = pydicom.dcmread(fpath)
        except Exception as exc:
            print(f"Failed to read file {fpath}: {exc}")
            success = False
            continue

        modality = getattr(ds, "Modality", "")
        if modality == "REG":
            registration.append((ds, fpath))
        elif modality == "RTSTRUCT":
            rtstruct.append((ds, fpath))
        else:
            imaging.append((ds, fpath))

    ordered = imaging + rtstruct + registration
    total = len(ordered)

    for idx, (ds, fpath) in enumerate(ordered, 1):
        status = send_file(assoc, ds)
        if status and status.Status == 0x0000:
            try:
                pass
                # os.remove(fpath)
            except Exception:
                pass
        else:
            print(f"Failed to send file: {fpath}")
            success = False
        if progress_callback:
            progress_callback(idx, total)

    assoc.release()
    return success
