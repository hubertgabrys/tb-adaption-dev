import os
import socket

import pydicom
from pydicom.uid import ImplicitVRLittleEndian
from pynetdicom import AE
from pynetdicom.sop_class import CTImageStorage, MRImageStorage, RTStructureSetStorage, Verification, SpatialRegistrationStorage

from utils import load_environment, require_env

# Load the .env
load_environment(".env")

# DICOM server configuration cache
_SERVER_CONFIG = None

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


def _get_server_config():
    global _SERVER_CONFIG
    if _SERVER_CONFIG is None:
        try:
            ae_title = require_env('SERVER_AE_TITLE')
            server_ip = require_env('SERVER_IP')
            server_port = int(require_env('SERVER_PORT'))
        except (EnvironmentError, ValueError) as exc:
            raise RuntimeError(str(exc)) from exc
        _SERVER_CONFIG = (ae_title, server_ip, server_port)
    return _SERVER_CONFIG


def send_files_to_aria(filepaths, progress_callback=None):
    """Send the DICOM *filepaths* to the ARIA server."""
    try:
        ae_title, server_ip, server_port = _get_server_config()
    except RuntimeError as exc:
        print(f"Configuration error: {exc}")
        return False

    ae = AE(socket.gethostname())

    for sop in SOPS:
        ae.add_requested_context(sop, TXS)
    ae.add_requested_context(Verification)

    ae.maximum_pdu_size = 16 * 1024 * 1024
    ae.acse_timeout = 120
    ae.dimse_timeout = 120
    ae.network_timeout = 120

    assoc = ae.associate(server_ip, server_port, ae_title=ae_title)
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
                os.remove(fpath)
            except Exception as exc:
                print(f"Failed to delete file {fpath}: {exc}")
        else:
            print(f"Failed to send file: {fpath}")
            success = False
        if progress_callback:
            progress_callback(idx, total)

    assoc.release()
    return success
