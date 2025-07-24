import os
import shutil
import time

import SimpleITK as sitk
import pydicom
from pydicom.uid import generate_uid
from pydicom.tag import Tag

from utils import get_datetime


def get_dicom_value(ds, tag, default=""):
    """Return the value of a DICOM tag, or a default if missing/empty."""
    element = ds.get(tag)
    if element is None:
        return default
    value = element.value
    if value in (None, "", b""):
        return default
    return value


def move_original_ct(folder_path):
    """
    Move all DICOM files starting with 'CT' and ending with '.dcm' from
    the given folder into a new folder named <folder_name>_original_CT
    in the parent directory.

    Returns:
        str: The path to the newly created folder containing the original CT files.
    """
    parent_dir = os.path.abspath(os.path.join(folder_path, os.pardir))
    original_folder_name = os.path.basename(folder_path) + "_original_CT"
    target_dir = os.path.join(parent_dir, original_folder_name)

    # Recreate the target directory if it exists
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    os.makedirs(target_dir, exist_ok=True)

    # Move all "CT...dcm" files
    for file_name in os.listdir(folder_path):
        if file_name.startswith("CT") and file_name.endswith(".dcm"):
            source_path = os.path.join(folder_path, file_name)
            destination_path = os.path.join(target_dir, file_name)
            shutil.move(source_path, destination_path)

    return target_dir


def load_dicom_series(folder_path):
    """
    Load a DICOM series from the given folder using SimpleITK.
    The DICOM files are sorted based on slice positions extracted from DICOM tags.

    Returns:
        SimpleITK.Image: The loaded 3D image.
    """
    # Get list of files in the directory
    dicom_files = [f for f in os.listdir(folder_path) if f.startswith("CT") and f.endswith(".dcm")]

    # Ensure there are matching DICOM files
    if not dicom_files:
        raise FileNotFoundError("No DICOM files starting with 'CT' and ending with '.dcm' found in the folder.")

    # Create a list of tuples: (slice_position, full_file_path)
    dicom_slices = []
    for file_name in dicom_files:
        file_path = os.path.join(folder_path, file_name)
        ds = pydicom.dcmread(file_path, stop_before_pixels=True)

        # Use Image Position (Patient) for sorting if available
        if hasattr(ds, "ImagePositionPatient"):
            slice_position = ds.ImagePositionPatient[2]  # Z-coordinate
        else:
            raise ValueError(f"Cannot determine slice position for file: {file_name}")

        dicom_slices.append((slice_position, file_path))

    # Sort DICOM files based on slice position
    dicom_slices.sort(key=lambda x: x[0])
    sorted_file_paths = [file_path for _, file_path in dicom_slices]

    # Load the sorted series using SimpleITK
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(sorted_file_paths)
    image = reader.Execute()

    return image


def delete_folder(folder_path):
    """
    Deletes the specified folder and all its contents if it exists.
    """
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)


def resample_image_to_resolution(image, new_spacing):
    """
    Resamples a 3D SimpleITK image to the given user-defined resolution (voxel dimensions).

    Parameters:
    - image: SimpleITK image (3D)
    - new_spacing: List or tuple with the new voxel dimensions (resolution) in mm [x, y, z]

    Returns:
    - Resampled 3D image with the new voxel spacing.
    """
    # Get original spacing and size
    original_spacing = image.GetSpacing()  # Current voxel dimensions
    original_size = image.GetSize()  # Current image size (number of voxels)
    print(f"  Original spacing: {[round(e,1) for e in original_spacing]}")

    # Calculate the new size based on the new spacing
    # New size (number of voxels) is calculated to preserve physical image dimensions
    new_size = [
        int(round(osz * ospc / nspc)) for osz, ospc, nspc in zip(original_size, original_spacing, new_spacing)
    ]

    # Define the resampler
    resampler = sitk.ResampleImageFilter()

    # Set the new spacing (voxel dimensions)
    resampler.SetOutputSpacing(new_spacing)

    # Set the new size (number of voxels) based on the new spacing
    resampler.SetSize(new_size)

    # Set the interpolator (use linear interpolation for continuous image values)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(-1024)

    # Set the output origin and direction the same as the original image
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputDirection(image.GetDirection())

    # Resample the image
    resampled_image = resampler.Execute(image)

    # Preserve the original pixel type when casting to avoid HU wrap-around
    # issues when saving the resampled slices as DICOM.
    resampled_image = sitk.Cast(resampled_image, image.GetPixelID())
    print(f"  Resampled image spacing: {[round(e, 1) for e in resampled_image.GetSpacing()]}")

    return resampled_image


def save_resampled_image_as_dicom(resampled_CT, input_folder, output_folder):
    """
    Save the resampled 3D image as a series of DICOM files, with most DICOM tags copied from the original image.

    Parameters:
    - resampled_CT: The resampled 3D SimpleITK image to save.
    - original_CT: The original 3D SimpleITK image to copy DICOM tags from.
    - output_folder: The folder to save the series of DICOM slices.
    """

    # Load the original CT to get the DICOM tags
    dicom_files = [f for f in os.listdir(input_folder) if f.startswith("CT") and f.endswith(".dcm")]

    # Ensure there are matching DICOM files
    if not dicom_files:
        raise FileNotFoundError("No DICOM files starting with 'CT' and ending with '.dcm' found in the folder.")

    # Convert relative file paths to absolute paths
    dicom_file_paths = [os.path.join(input_folder, f) for f in dicom_files]

    # Load the first slice with pydicom to retreive tags
    original_CT_pydicom = pydicom.dcmread(dicom_file_paths[0])

    writer = sitk.ImageFileWriter()
    # Use the UID values explicitly stored in the metadata and do not let
    # SimpleITK generate new identifiers automatically.
    writer.KeepOriginalImageUIDOn()

    # Copy relevant tags from the original meta-data dictionary (private tags are
    # also accessible).
    # Generate a new Series Instance UID for the resampled output
    new_series_uid = generate_uid()
    # Series DICOM tags to copy
    series_tag_values = [
        ("0008|0005", get_dicom_value(original_CT_pydicom, Tag(0x00080005))),  # Specific Character Set
        ("0018|0060", get_dicom_value(original_CT_pydicom, Tag(0x00180060))),  # kVp
        ("0010|0010", get_dicom_value(original_CT_pydicom, Tag(0x00100010))),  # Patient Name
        ("0010|0020", get_dicom_value(original_CT_pydicom, Tag(0x00100020))),  # Patient ID
        ("0010|0030", get_dicom_value(original_CT_pydicom, Tag(0x00100030))),  # Patient Birth Date
        ("0010|0040", get_dicom_value(original_CT_pydicom, Tag(0x00100040))),  # Patient Sex
        ("0020|000d", get_dicom_value(original_CT_pydicom, Tag(0x0020000d))),  # Study Instance UID, for machine consumption
        ("0020|0010", get_dicom_value(original_CT_pydicom, Tag(0x00200010))),  # Study ID, for human consumption
        ("0008|0020", get_dicom_value(original_CT_pydicom, Tag(0x00080020))),  # Study Date
        ("0008|0021", get_dicom_value(original_CT_pydicom, Tag(0x00080021))),  # Series Date
        ("0008|0022", get_dicom_value(original_CT_pydicom, Tag(0x00080022))),  # Acquisition Date
        ("0008|0030", get_dicom_value(original_CT_pydicom, Tag(0x00080030))),  # Study Time
        ("0008|0031", get_dicom_value(original_CT_pydicom, Tag(0x00080031))),  # Series time
        ("0008|0032", get_dicom_value(original_CT_pydicom, Tag(0x00080032))),  # Acquisition time
        ("0008|0050", get_dicom_value(original_CT_pydicom, Tag(0x00080050))),  # Accession Number
        ("0008|0060", get_dicom_value(original_CT_pydicom, Tag(0x00080060))),  # Modality
        ("0008|0064", get_dicom_value(original_CT_pydicom, Tag(0x00080064))),  # Conversion Type
        ("0028|1050", get_dicom_value(original_CT_pydicom, Tag(0x00281050))),  # Window center
        ("0028|1051", get_dicom_value(original_CT_pydicom, Tag(0x00281051))),  # Window width
        ("0028|1052", get_dicom_value(original_CT_pydicom, Tag(0x00281052))),  # Rescale Intercept
        ("0028|1053", get_dicom_value(original_CT_pydicom, Tag(0x00281053))),  # Rescale Slope
        ("0028|1054", get_dicom_value(original_CT_pydicom, Tag(0x00281054))),  # Rescale Type
        ("0018|5100", get_dicom_value(original_CT_pydicom, Tag(0x00185100))),  # Patient Position
        ("0020|0052", get_dicom_value(original_CT_pydicom, Tag(0x00200052))),  # Frame of Reference UID
        ("0008|0080", get_dicom_value(original_CT_pydicom, Tag(0x00080080))),  # Institution
        ("0008|1030", get_dicom_value(original_CT_pydicom, Tag(0x00081030))),  # Study description
        ("0020|0011", get_dicom_value(original_CT_pydicom, Tag(0x00200011))),  # Series number
        ("0020|0012", get_dicom_value(original_CT_pydicom, Tag(0x00200012))),  # Acquisition number
        ("0020|1040", get_dicom_value(original_CT_pydicom, Tag(0x00201040))),  # Position reference indicator
        ("0020|000e", new_series_uid),  # Series Instance UID (new)
        ("0008|103e", f"{get_dicom_value(original_CT_pydicom, Tag(0x0008103e))} Resampled"),  # Series description
        # ("0008|0070", get_dicom_value(original_CT_pydicom, Tag(0x00080070))),  # Manufacturer
        ("0008|1090", get_dicom_value(original_CT_pydicom, Tag(0x00081090))),  # Manufacturer model name
        ("0018|1000", get_dicom_value(original_CT_pydicom, Tag(0x00181000))),  # Device Serial Number
        ("0008|0070", "Spectronic Medical AB / MIM Software"),  # Manufacturer
        # ("0008|1090", "Freemax"),  # Manufacturer model name
        # ("0018|1000", "206207"),  # Device Serial Number

    ]

    spacing_resampled = resampled_CT.GetSpacing()  # (spacing_x, spacing_y, spacing_z)

    for i in range(resampled_CT.GetDepth()):
        image_slice = resampled_CT[:, :, i]
        # Tags shared by the series.
        for tag, value in series_tag_values:
            image_slice.SetMetaData(tag, str(value))
        # Slice specific tags.
        #   Instance Creation Date
        image_slice.SetMetaData("0008|0012", time.strftime("%Y%m%d"))
        #   Instance Creation Time
        image_slice.SetMetaData("0008|0013", time.strftime("%H%M%S"))
        #   Image Position (Patient)
        image_slice.SetMetaData(
            "0020|0032",
            "\\".join(map(str, resampled_CT.TransformIndexToPhysicalPoint((0, 0, i)))),
        )
        #   Instance Number - DICOM uses 1-based numbering
        image_slice.SetMetaData("0020|0013", str(i + 1))
        #   SOP Instance UID - unique per slice
        image_slice.SetMetaData("0008|0018", generate_uid())
        # Set slice thickness (assuming uniform thickness)
        image_slice.SetMetaData("0018|0050", f"{spacing_resampled[2]:.6f}")  # Slice Thickness

        # Image type
        image_slice.SetMetaData("0008|0008", "DERIVED\\SECONDARY\\AXIAL")

        # Write to the output directory and add the extension dcm, to force writing
        # in DICOM format.
        filename_save = os.path.join(output_folder, f"CT_Resampled_Slice_{i:04d}.dcm")
        writer.SetFileName(filename_save)
        writer.Execute(image_slice)

        # This will fix issues with German special characters in patient name
        ds = pydicom.dcmread(filename_save)
        ds.PatientName = get_dicom_value(original_CT_pydicom, Tag(0x00100010))
        ds.save_as(filename_save)


def resample_ct(current_folder):
    """
    Main workflow:
      1. Check the current image resolution. If the spacing is already
         1.5x1.5x1.5 mm, don't perform the next steps.
      2. Move DICOM files to a new folder named <folder>_original_CT.
      3. Load original CT (SimpleITK).
      4. Resample to new_spacing.
      5. Save resampled CT as DICOM.
      6. Delete the folder with the original CT.
    """

    # Step 1: Check current spacing using the header of the first slice
    print(f"{get_datetime()} Checking current CT resolution")
    dicom_files = [f for f in os.listdir(current_folder) if f.startswith("CT") and f.endswith(".dcm")]
    if not dicom_files:
        print(f"{get_datetime()} No CT series found -> skipping resampling")
        return "aborted"

    header = pydicom.dcmread(os.path.join(current_folder, dicom_files[0]), stop_before_pixels=True)
    try:
        spacing_x, spacing_y = map(float, header.PixelSpacing)
        spacing_z = float(getattr(header, "SliceThickness", 0))
        current_spacing = (spacing_x, spacing_y, spacing_z)
    except Exception:
        current_CT = load_dicom_series(current_folder)
        current_spacing = current_CT.GetSpacing()

    if all(abs(sp - 1.5) < 1e-3 for sp in current_spacing):
        print(
            f"{get_datetime()} CT already at 1.5x1.5x1.5 mm resolution -> no resampling"
        )
        return "aborted"

    # Step 2: Move the original CT to another folder
    print(f"{get_datetime()} Move the original CT to another folder")
    moved_original_CT_folder = move_original_ct(current_folder)

    # Step 3: Load the original CT from the moved folder
    print(f"{get_datetime()} Reading the original sCT")
    original_CT = load_dicom_series(moved_original_CT_folder)

    # Step 4: Resample to a user-defined resolution
    print(f"{get_datetime()} Resampling to the 1.5x1.5x1.5 mm resolution")
    new_spacing = [1.5, 1.5, 1.5]  # in mm
    resampled_CT = resample_image_to_resolution(original_CT, new_spacing)

    # Step 5: Write the resampled CT to the subfolder
    print(f"{get_datetime()} Saving the resampled sCT")
    save_resampled_image_as_dicom(resampled_CT, moved_original_CT_folder, current_folder)

    # Step 6: Delete the original CT to avoid problems
    print(f"{get_datetime()} Deleting the original CT")
    delete_folder(moved_original_CT_folder)

    return "success"
