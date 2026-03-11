import os
import sys
from pathlib import Path

import PyInstaller.__main__
import pydicom

# Detect platform
is_windows = sys.platform.startswith('win')
is_mac = sys.platform.startswith('darwin')

# Choose add-data separator based on OS
add_data_sep = ';' if is_windows else ':'

# Get the path to the pydicom data directory
pydicom_data_dir = os.path.join(os.path.dirname(pydicom.__file__), 'data')

# Print pydicom data directory for verification
print(f"Pydicom data directory: {pydicom_data_dir}")

project_root = Path(__file__).resolve().parents[1]

PyInstaller.__main__.run([
    str(project_root / 'run_gui.py'),
    '--onefile',
    f'--add-data={project_root / ".env"}{add_data_sep}.',
    f'--paths={project_root / "src"}',
    '--hidden-import=pandas',
    '--hidden-import=pydicom',
    '--hidden-import=pynetdicom',
    '--clean'
])
