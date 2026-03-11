import datetime
import functools
import os
import sys
import warnings
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - depends on local environment
    def load_dotenv(*args, **kwargs):
        return False

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
    env_path = Path(env_file_path)
    if getattr(sys, "frozen", False):
        # running as bundled exe
        candidates = [Path(sys.executable).resolve().parent / env_path]
    else:
        project_root = Path(__file__).resolve().parents[1]
        candidates = [
            Path.cwd() / env_path,
            project_root / env_path,
        ]

    for candidate in candidates:
        if candidate.exists():
            load_dotenv(candidate)
            return

    # Fall back to the project-root candidate so callers keep consistent behavior.
    load_dotenv(candidates[-1])


def require_env(var_name: str) -> str:
    """Return the value of *var_name* or raise if the environment is missing it."""

    value = os.environ.get(var_name)
    if value is None or value == "":
        raise EnvironmentError(f"Environment variable '{var_name}' must be set")
    return value


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
