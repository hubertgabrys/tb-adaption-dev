import os
import sys

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

PyInstaller.__main__.run([
    'run_gui.py',
    '--onefile',
    f'--add-data=.env{add_data_sep}.',
    '--hidden-import=pandas',
    '--hidden-import=pydicom',
    '--hidden-import=pynetdicom',
    '--clean'
])
