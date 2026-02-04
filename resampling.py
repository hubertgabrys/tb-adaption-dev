import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor

import SimpleITK as sitk
import pydicom
from pydicom.multival import MultiValue
from pydicom.tag import Tag
from pydicom.uid import generate_uid

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


def _format_cs_value(value):
    """Format CS (Code String) values without Python list/quote artifacts."""
    if value in (None, "", b""):
        return ""
    if isinstance(value, (list, tuple)):
        return "\\".join(str(item) for item in value if item not in (None, "", b""))
    return str(value)


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


def load_dicom_series(folder_path: str) -> sitk.Image:
    """
    Load a DICOM series in one shot using SimpleITK’s GDCM helper.
    """
    reader = sitk.ImageSeriesReader()
    # get all series IDs in the folder
    series_IDs = reader.GetGDCMSeriesIDs(folder_path)
    if not series_IDs:
        raise FileNotFoundError(f"No DICOM series found in {folder_path}")

    # pick the first (or choose by modality, UID, etc.)
    series_id = series_IDs[0]

    # GetGDCMSeriesFileNames will both list *and* sort the files correctly
    file_names = reader.GetGDCMSeriesFileNames(folder_path, series_id)
    reader.SetFileNames(file_names)

    # Optionally turn on metadata dictionary update if you need to inspect tags later
    reader.MetaDataDictionaryArrayUpdateOn()

    # This Execute() call is entirely in C++ and multi‑threaded under the hood
    image = reader.Execute()
    return image


def _sanitize_folder_component(value: str) -> str:
    """Make a safe folder name component from a DICOM string."""
    value = (value or "").strip()
    if not value:
        return "unknown"
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value)


def _find_series_by_description(folder_path: str, modality: str, series_description: str):
    """Return matching GDCM series entries (series_id, files, header)."""
    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(folder_path) or []
    matches = []
    for series_id in series_ids:
        file_names = reader.GetGDCMSeriesFileNames(folder_path, series_id)
        if not file_names:
            continue
        try:
            header = pydicom.dcmread(file_names[0], stop_before_pixels=True)
        except Exception:
            continue
        if getattr(header, "Modality", "").strip() != modality:
            continue
        desc = getattr(header, "SeriesDescription", "").strip()
        if desc != series_description:
            continue
        matches.append((series_id, file_names, header))
    return matches


def delete_folder(folder_path):
    """
    Deletes the specified folder and all its contents if it exists.
    """
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)


def resample_image_to_resolution(image, new_spacing, default_pixel_value=-1024):
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
    resampler.SetDefaultPixelValue(default_pixel_value)

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
    resampled_CT = resample_image_to_resolution(
        original_CT,
        new_spacing,
        default_pixel_value=-1024,
    )

    # Step 5: Write the resampled CT to the subfolder
    print(f"{get_datetime()} Saving the resampled sCT")
    save_resampled_ct_as_dicom(resampled_CT, moved_original_CT_folder, current_folder)

    # Step 6: Delete the original CT to avoid problems
    print(f"{get_datetime()} Deleting the original CT")
    delete_folder(moved_original_CT_folder)

    return "success"


def resample_mr_series_by_description(
    current_folder,
    series_description="sCT_sp_Pel_T2",
    target_slice_thickness=3.0,
):
    """
    Resample MR series matching *series_description* to target slice thickness,
    keeping in-plane resolution unchanged.
    """
    print(
        f"{get_datetime()} Checking MR series for resampling: '{series_description}'"
    )
    matches = _find_series_by_description(
        current_folder,
        modality="MR",
        series_description=series_description,
    )
    if not matches:
        print(
            f"{get_datetime()} No matching MR series found -> skipping resampling"
        )
        return "aborted"

    result_state = "aborted"
    for _, file_names, header in matches:
        try:
            spacing_x, spacing_y = map(float, header.PixelSpacing)
        except Exception:
            spacing_x = spacing_y = None
        try:
            spacing_z = float(getattr(header, "SliceThickness", 0) or 0)
        except Exception:
            spacing_z = 0.0

        if spacing_z and abs(spacing_z - target_slice_thickness) < 1e-3:
            print(
                f"{get_datetime()} MR series already at {target_slice_thickness:.1f} mm -> skipping"
            )
            continue

        series_uid = getattr(header, "SeriesInstanceUID", "")
        series_uid_suffix = series_uid[-8:] if series_uid else "series"
        safe_desc = _sanitize_folder_component(series_description)
        parent_dir = os.path.abspath(os.path.join(current_folder, os.pardir))
        original_folder_name = (
            f"{os.path.basename(current_folder)}_original_MR_{safe_desc}_{series_uid_suffix}"
        )
        moved_series_folder = os.path.join(parent_dir, original_folder_name)

        if os.path.exists(moved_series_folder):
            shutil.rmtree(moved_series_folder)
        os.makedirs(moved_series_folder, exist_ok=True)

        for src_path in file_names:
            dst_path = os.path.join(moved_series_folder, os.path.basename(src_path))
            shutil.move(src_path, dst_path)

        print(
            f"{get_datetime()} Resampling MR series '{series_description}'"
        )
        original_series = load_dicom_series(moved_series_folder)
        if spacing_x is None or spacing_y is None:
            spacing_x, spacing_y, _ = original_series.GetSpacing()

        new_spacing = [spacing_x, spacing_y, target_slice_thickness]
        resampled_series = resample_image_to_resolution(
            original_series,
            new_spacing,
            default_pixel_value=0,
        )

        save_resampled_mr_as_dicom(
            resampled_series,
            moved_series_folder,
            current_folder,
            source_prefix=None,
            output_prefix="MR_Resampled_Slice_",
            series_description_suffix=" Resampled",
        )

        delete_folder(moved_series_folder)
        result_state = "success"

    return result_state


def _format_meta_value(value) -> str:
    """
    Convert pydicom values (including MultiValue / list) into DICOM-style strings.
    Ensures multi-valued attributes become backslash-separated (e.g. "SP\\SK"),
    not Python list repr (e.g. "['SP', 'SK']").
    """
    if value in (None, "", b""):
        return ""

    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except Exception:
            return value.decode("latin-1", errors="replace")

    # Multi-valued: join with backslash
    if isinstance(value, (list, tuple, MultiValue)):
        items = [v for v in value if v not in (None, "", b"")]
        if len(items) > 1:
            return "\\".join(str(v) for v in items)
        if len(items) == 1:
            return str(items[0])
        return ""

    s = str(value)

    # Optional: fix already-stringified python-list artifacts like "['SP', 'SK']"
    if s.startswith("[") and s.endswith("]") and ("'" in s or '"' in s) and "," in s:
        inner = s[1:-1]
        parts = [p.strip().strip("'").strip('"') for p in inner.split(",")]
        parts = [p for p in parts if p]
        if len(parts) > 1:
            return "\\".join(parts)
        if len(parts) == 1:
            return parts[0]
        return ""

    return s


def _get_dicom_value(ds, tag: Tag, default=""):
    elem = ds.get(tag)
    if elem is None:
        return default
    v = elem.value
    if v in (None, "", b""):
        return default
    return v


def _infer_specific_charset(src_ds: pydicom.Dataset) -> str:
    """
    Writing via SimpleITK/GDCM: if any written text is non-ASCII, force UTF-8 (ISO_IR 192).
    Otherwise keep source charset if present, else empty.
    """
    src_cs = _format_meta_value(_get_dicom_value(src_ds, Tag(0x0008, 0x0005), ""))

    # Check a few high-impact text fields that you write
    text_tags = [
        Tag(0x0010, 0x0010),  # PatientName
        Tag(0x0008, 0x0080),  # InstitutionName
        Tag(0x0008, 0x1030),  # StudyDescription
        Tag(0x0008, 0x103E),  # SeriesDescription
    ]
    for t in text_tags:
        v = _get_dicom_value(src_ds, t, "")
        if not v:
            continue
        s = _format_meta_value(v)
        try:
            s.encode("ascii")
        except Exception:
            return "ISO_IR 192"  # force UTF-8 for output

    # ASCII-only: keep original (if any), else empty
    return src_cs


def _read_source_header(input_folder: str, *, source_prefix: str | None) -> pydicom.Dataset:
    dicom_files = [
        f for f in os.listdir(input_folder)
        if f.lower().endswith(".dcm") and (source_prefix is None or f.startswith(source_prefix))
    ]
    if not dicom_files:
        raise FileNotFoundError("No DICOM files found in input_folder matching the expected prefix.")
    dicom_files.sort()
    first_path = os.path.join(input_folder, dicom_files[0])
    return pydicom.dcmread(first_path, stop_before_pixels=True)


def _sitk_direction_to_iop(direction_3x3) -> str:
    """
    SimpleITK direction is a 3x3 matrix flattened row-major:
      (d00,d01,d02, d10,d11,d12, d20,d21,d22)
    ITK convention: columns are the direction cosines of image axes (i, j, k).
    DICOM IOP expects: row_cosines (i axis) then col_cosines (j axis):
      [d00,d10,d20, d01,d11,d21]
    """
    d = direction_3x3
    return "\\".join(map(str, [d[0], d[3], d[6], d[1], d[4], d[7]]))


def _build_common_series_tags(
    src_ds: pydicom.Dataset,
    *,
    new_series_uid: str,
    series_description_suffix: str,
) -> list[tuple[str, str]]:
    specific_charset = _infer_specific_charset(src_ds)

    # Manufacturer quirk preserved from your original code
    manufacturer = _format_meta_value(_get_dicom_value(src_ds, Tag(0x0008, 0x0070), ""))
    if manufacturer == "Spectronic Medical AB":
        manufacturer = "Spectronic Medical AB / MIM Software"

    base = [
        ("0008|0005", specific_charset),  # Specific Character Set

        # Patient
        ("0010|0010", _format_meta_value(_get_dicom_value(src_ds, Tag(0x0010, 0x0010), ""))),  # Patient Name
        ("0010|0020", _format_meta_value(_get_dicom_value(src_ds, Tag(0x0010, 0x0020), ""))),  # Patient ID
        ("0010|0030", _format_meta_value(_get_dicom_value(src_ds, Tag(0x0010, 0x0030), ""))),  # Patient Birth Date
        ("0010|0040", _format_meta_value(_get_dicom_value(src_ds, Tag(0x0010, 0x0040), ""))),  # Patient Sex

        # Study
        ("0020|000d", _format_meta_value(_get_dicom_value(src_ds, Tag(0x0020, 0x000D), ""))),  # Study Instance UID
        ("0020|0010", _format_meta_value(_get_dicom_value(src_ds, Tag(0x0020, 0x0010), ""))),  # Study ID
        ("0008|0020", _format_meta_value(_get_dicom_value(src_ds, Tag(0x0008, 0x0020), ""))),  # Study Date
        ("0008|0030", _format_meta_value(_get_dicom_value(src_ds, Tag(0x0008, 0x0030), ""))),  # Study Time
        ("0008|0050", _format_meta_value(_get_dicom_value(src_ds, Tag(0x0008, 0x0050), ""))),  # Accession Number
        ("0008|0080", _format_meta_value(_get_dicom_value(src_ds, Tag(0x0008, 0x0080), ""))),  # Institution
        ("0008|1030", _format_meta_value(_get_dicom_value(src_ds, Tag(0x0008, 0x1030), ""))),  # Study description

        # Series
        ("0020|000e", new_series_uid),  # Series Instance UID (new)
        ("0008|0060", _format_meta_value(_get_dicom_value(src_ds, Tag(0x0008, 0x0060), ""))),  # Modality
        ("0008|0070", manufacturer),  # Manufacturer
        ("0008|1090", _format_meta_value(_get_dicom_value(src_ds, Tag(0x0008, 0x1090), ""))),  # Model
        ("0018|1000", _format_meta_value(_get_dicom_value(src_ds, Tag(0x0018, 0x1000), ""))),  # Device serial
        ("0020|0011", _format_meta_value(_get_dicom_value(src_ds, Tag(0x0020, 0x0011), ""))),  # Series number
        ("0020|0012", _format_meta_value(_get_dicom_value(src_ds, Tag(0x0020, 0x0012), ""))),  # Acquisition number
        ("0020|0052", _format_meta_value(_get_dicom_value(src_ds, Tag(0x0020, 0x0052), ""))),  # Frame of Reference UID
        ("0020|1040", _format_meta_value(_get_dicom_value(src_ds, Tag(0x0020, 0x1040), ""))),  # Position ref indicator
        ("0008|103e", _format_meta_value(_get_dicom_value(src_ds, Tag(0x0008, 0x103E), "")) + series_description_suffix),
    ]

    # Avoid writing empty strings as metadata if possible
    return [(t, v) for (t, v) in base if v != ""]


def _build_ct_series_tags(src_ds: pydicom.Dataset) -> list[tuple[str, str]]:
    # CT-specific tags (keep your current set; expand if needed)
    tags = [
        ("0018|0060", _format_meta_value(_get_dicom_value(src_ds, Tag(0x0018, 0x0060), ""))),  # kVp
        ("0018|5100", _format_meta_value(_get_dicom_value(src_ds, Tag(0x0018, 0x5100), ""))),  # Patient Position

        ("0028|1050", _format_meta_value(_get_dicom_value(src_ds, Tag(0x0028, 0x1050), ""))),  # Window center
        ("0028|1051", _format_meta_value(_get_dicom_value(src_ds, Tag(0x0028, 0x1051), ""))),  # Window width
        ("0028|1052", _format_meta_value(_get_dicom_value(src_ds, Tag(0x0028, 0x1052), ""))),  # Rescale Intercept
        ("0028|1053", _format_meta_value(_get_dicom_value(src_ds, Tag(0x0028, 0x1053), ""))),  # Rescale Slope
        ("0028|1054", _format_meta_value(_get_dicom_value(src_ds, Tag(0x0028, 0x1054), ""))),  # Rescale Type

        ("0008|0064", _format_meta_value(_get_dicom_value(src_ds, Tag(0x0008, 0x0064), ""))),  # Conversion Type
        ("0018|0020", _format_meta_value(_get_dicom_value(src_ds, Tag(0x0018, 0x0020), ""))),  # Scanning Sequence
        ("0018|0021", _format_meta_value(_get_dicom_value(src_ds, Tag(0x0018, 0x0021), ""))),  # Sequence Variant
    ]
    return [(t, v) for (t, v) in tags if v != ""]


def _build_mr_series_tags(src_ds: pydicom.Dataset) -> list[tuple[str, str]]:
    # MR-specific tags (minimal/safe set; extend if your consumers rely on more)
    tags = [
        ("0018|0020", _format_meta_value(_get_dicom_value(src_ds, Tag(0x0018, 0x0020), ""))),  # Scanning Sequence
        ("0018|0021", _format_meta_value(_get_dicom_value(src_ds, Tag(0x0018, 0x0021), ""))),  # Sequence Variant
        ("0018|0022", _format_meta_value(_get_dicom_value(src_ds, Tag(0x0018, 0x0022), ""))),  # Scan Options
        ("0018|0023", _format_meta_value(_get_dicom_value(src_ds, Tag(0x0018, 0x0023), ""))),  # MR Acquisition Type
        ("0018|0080", _format_meta_value(_get_dicom_value(src_ds, Tag(0x0018, 0x0080), ""))),  # TR
        ("0018|0081", _format_meta_value(_get_dicom_value(src_ds, Tag(0x0018, 0x0081), ""))),  # TE
        ("0018|1314", _format_meta_value(_get_dicom_value(src_ds, Tag(0x0018, 0x1314), ""))),  # Flip Angle
        ("0018|1312", _format_meta_value(_get_dicom_value(src_ds, Tag(0x0018, 0x1312), ""))),  # Phase Encoding Dir
        ("0020|1040", _format_meta_value(_get_dicom_value(src_ds, Tag(0x0020, 0x1040), ""))),  # Position Reference Indicator
    ]
    return [(t, v) for (t, v) in tags if v != ""]


def _write_sitk_volume_as_dicom_series(
    volume: sitk.Image,
    *,
    output_folder: str,
    output_prefix: str,
    series_tags: list[tuple[str, str]],
    default_image_type: str,
    use_threads: bool = True,
    max_workers: int | None = None,
) -> None:
    os.makedirs(output_folder, exist_ok=True)

    spacing = volume.GetSpacing()          # (sx, sy, sz)
    direction = volume.GetDirection()
    iop = _sitk_direction_to_iop(direction)

    # Keep timestamps consistent across slices
    creation_date = time.strftime("%Y%m%d")
    creation_time = time.strftime("%H%M%S")

    pixel_spacing = f"{spacing[0]:.6f}\\{spacing[1]:.6f}"
    slice_thickness = f"{spacing[2]:.6f}"
    spacing_between_slices = f"{spacing[2]:.6f}"

    def write_slice(k: int) -> None:
        img2d = volume[:, :, k]

        # Series-level tags
        for tag, val in series_tags:
            img2d.SetMetaData(tag, val)

        # Geometry (explicit)
        img2d.SetMetaData("0028|0030", pixel_spacing)            # Pixel Spacing
        img2d.SetMetaData("0020|0037", iop)                      # Image Orientation (Patient)
        img2d.SetMetaData("0018|0050", slice_thickness)          # Slice Thickness
        img2d.SetMetaData("0018|0088", spacing_between_slices)   # Spacing Between Slices (if used)

        # Slice-specific tags
        img2d.SetMetaData("0008|0012", creation_date)            # Instance Creation Date
        img2d.SetMetaData("0008|0013", creation_time)            # Instance Creation Time
        img2d.SetMetaData("0020|0032", "\\".join(map(str, volume.TransformIndexToPhysicalPoint((0, 0, k)))))  # IPP
        img2d.SetMetaData("0020|0013", str(k + 1))               # Instance Number
        img2d.SetMetaData("0008|0018", generate_uid())           # SOP Instance UID
        img2d.SetMetaData("0008|0008", default_image_type)       # Image Type

        out_path = os.path.join(output_folder, f"{output_prefix}{k:04d}.dcm")
        writer = sitk.ImageFileWriter()
        writer.KeepOriginalImageUIDOn()
        writer.SetFileName(out_path)
        writer.Execute(img2d)

    depth = volume.GetDepth()
    if use_threads and depth > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            list(ex.map(write_slice, range(depth)))
    else:
        for k in range(depth):
            write_slice(k)


def save_resampled_ct_as_dicom(
    resampled_ct: sitk.Image,
    input_folder: str,
    output_folder: str,
    *,
    source_prefix: str = "CT",
    output_prefix: str = "CT_Resampled_Slice_",
    series_description_suffix: str = " Resampled",
    use_threads: bool = True,
    max_workers: int | None = None,
) -> None:
    src_ds = _read_source_header(input_folder, source_prefix=source_prefix)

    new_series_uid = generate_uid()
    tags = _build_common_series_tags(
        src_ds,
        new_series_uid=new_series_uid,
        series_description_suffix=series_description_suffix,
    )
    tags += _build_ct_series_tags(src_ds)

    _write_sitk_volume_as_dicom_series(
        resampled_ct,
        output_folder=output_folder,
        output_prefix=output_prefix,
        series_tags=tags,
        default_image_type="DERIVED\\SECONDARY\\AXIAL",
        use_threads=use_threads,
        max_workers=max_workers,
    )


def save_resampled_mr_as_dicom(
    resampled_mr: sitk.Image,
    input_folder: str,
    output_folder: str,
    *,
    source_prefix: str | None = None,  # allow arbitrary filenames for MR
    output_prefix: str = "MR_Resampled_Slice_",
    series_description_suffix: str = " Resampled",
    use_threads: bool = True,
    max_workers: int | None = None,
) -> None:
    src_ds = _read_source_header(input_folder, source_prefix=source_prefix)

    new_series_uid = generate_uid()
    tags = _build_common_series_tags(
        src_ds,
        new_series_uid=new_series_uid,
        series_description_suffix=series_description_suffix,
    )
    tags += _build_mr_series_tags(src_ds)

    _write_sitk_volume_as_dicom_series(
        resampled_mr,
        output_folder=output_folder,
        output_prefix=output_prefix,
        series_tags=tags,
        default_image_type="DERIVED\\SECONDARY",
        use_threads=use_threads,
        max_workers=max_workers,
    )
