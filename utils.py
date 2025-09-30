import datetime
import functools
import os
import sys
import warnings

from dotenv import load_dotenv
from pydicom.valuerep import DS


def float_to_ds_string(x: float, precision: int = 8) -> DS:
    """Return *x* formatted for the DICOM DS VR."""
    s = f"{x:.{precision}f}".rstrip('0').rstrip('.')
    if len(s) > 16:
        raise ValueError(f"Value '{s}' exceeds 16 characters for DICOM DS")
    return DS(s)


def get_datetime():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

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


def configure_sitk_threads():
    """Configure SimpleITK to use all CPU cores."""
    try:
        import multiprocessing
        import SimpleITK as sitk
        n_threads = multiprocessing.cpu_count()
        sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(n_threads)
        print(f"Using {n_threads} SimpleITK threads")
    except Exception as exc:
        print(f"Could not configure SimpleITK threads: {exc}")

def deprecated(reason):
    def decorator(func):
        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            warnings.warn(
                f"{func.__name__}() is deprecated: {reason}",
                category=DeprecationWarning,
                stacklevel=2
            )
            return func(*args, **kwargs)
        return wrapped
    return decorator

def count_files(path):
    return sum(1 for _ in os.scandir(path))
