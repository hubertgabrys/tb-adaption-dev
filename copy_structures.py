import copy
import os
import re
from pathlib import Path

import pydicom
from tqdm import tqdm

from utils import float_to_ds_string


def transform_contour_points(transform, contour_data, precision: int = 8):
    """
    Given:
      - transform: a SimpleITK transform (with .TransformPoint)
      - contour_data: flat sequence of floats or numeric strings [x1,y1,z1, x2,y2,z2, ...]
    Returns:
      - a list of pydicom.valuerep.DS strings suitable for placing back into ContourData,
        with each coordinate transformed and formatted per DICOM DS VR.
    """
    if len(contour_data) % 3 != 0:
        raise ValueError("contour_data length must be a multiple of 3")

    out_ds = []
    for i in range(0, len(contour_data), 3):
        # build the input point as floats
        x, y, z = (float(contour_data[i]),
                   float(contour_data[i + 1]),
                   float(contour_data[i + 2]))
        # transform
        x_t, y_t, z_t = transform.TransformPoint((x, y, z))
        # convert each to DS
        out_ds.append(float_to_ds_string(x_t, precision))
        out_ds.append(float_to_ds_string(y_t, precision))
        out_ds.append(float_to_ds_string(z_t, precision))

    return out_ds


def _rtstruct_references_series(ds, series_uid):
    """Return True if *ds* references *series_uid*."""
    try:
        for fr in ds.ReferencedFrameOfReferenceSequence:
            for st in fr.RTReferencedStudySequence:
                for se in st.RTReferencedSeriesSequence:
                    if getattr(se, "SeriesInstanceUID", None) == series_uid:
                        return True
    except Exception:
        pass
    return False


def find_rtstruct(directory, description_prefix=None, series_uid=None):
    """Locate an RTSTRUCT in *directory* optionally filtered by description prefix
    or referenced SeriesInstanceUID."""

    for file_name in os.listdir(directory):
        rtstruct_path = os.path.join(directory, file_name)
        try:
            ds = pydicom.dcmread(rtstruct_path, stop_before_pixels=True)
        except Exception:
            continue
        if ds.Modality != "RTSTRUCT":
            continue

        if series_uid and not _rtstruct_references_series(ds, series_uid):
            continue

        if description_prefix is None:
            return ds, file_name
        elif hasattr(ds, "SeriesDescription"):
            if ds.SeriesDescription.lower().startswith(description_prefix.lower()):
                return ds, file_name
    return None, None


def read_base_rtstruct(patient_id, rtplan_label, series_uid=None):
    base_dir = Path(os.environ.get("BASEPLAN_DIR")) / patient_id / rtplan_label
    rtstruct, _ = find_rtstruct(str(base_dir), series_uid=series_uid)
    return rtstruct


def read_new_rtstruct(current_directory, series_uid=None):
    """Return RTSTRUCT in *current_directory* matching *series_uid* if provided."""

    if series_uid:
        filename = os.path.join(current_directory, f"RS_{series_uid}.dcm")
        if os.path.exists(filename):
            ds = pydicom.dcmread(filename)
            return ds, os.path.basename(filename)
        rtstruct, rtstruct_filename = find_rtstruct(current_directory, series_uid=series_uid)
        if rtstruct is not None:
            return rtstruct, rtstruct_filename

    # Fallback to keyword-based search
    rtstruct, rtstruct_filename = find_rtstruct(current_directory, "syntheticcthu")
    if rtstruct is not None:
        return rtstruct, rtstruct_filename
    rtstruct, rtstruct_filename = find_rtstruct(current_directory, "synthetic ct")
    if rtstruct is not None:
        return rtstruct, rtstruct_filename
    rtstruct, rtstruct_filename = find_rtstruct(current_directory, "t2_tse_tra")
    return rtstruct, rtstruct_filename


def copy_structures(current_directory, patient_id, rtplan_label, rigid_transform,
                    series_uid=None, base_series_uid=None, progress_callback=None):
    """Copy structures from the base plan RTSTRUCT to the daily RTSTRUCT."""

    # Read the base and new RTSTRUCT files referencing the chosen series.
    rtstruct_base = read_base_rtstruct(patient_id, rtplan_label, series_uid=base_series_uid)
    rtstruct_new, rtstruct_new_filename = read_new_rtstruct(current_directory, series_uid)

    # Extract plan suffix (e.g. "_1a") from the plan label if present
    match = re.search(r"_(\d[a-z])$", rtplan_label.lower())
    plan_suffix = match.group(0) if match else None

    # Get the inverse of the transform
    rigid_transform = rigid_transform.GetInverse()

    # Reset (or initialize) the new RTSTRUCT sequences.
    # We assume these sequences exist so we replace them with new, filtered sequences.
    rtstruct_new.StructureSetROISequence = pydicom.sequence.Sequence()
    rtstruct_new.ROIContourSequence = pydicom.sequence.Sequence()
    rtstruct_new.RTROIObservationsSequence = pydicom.sequence.Sequence()

    # Determine which ROI numbers contain contour data
    roi_number_has_contour = set()
    if hasattr(rtstruct_base, "ROIContourSequence"):
        for roi_contour in rtstruct_base.ROIContourSequence:
            num = getattr(roi_contour, "ReferencedROINumber", None)
            if num is None:
                continue
            has_data = False
            if hasattr(roi_contour, "ContourSequence"):
                for contour in roi_contour.ContourSequence:
                    data = getattr(contour, "ContourData", [])
                    if data:
                        has_data = True
                        break
            if has_data:
                roi_number_has_contour.add(num)

    # Helper function: determine if an ROI name should be skipped.
    def skip_roi(name, number):
        name_lower = name.lower()

        # Skip if there are no contours associated with this ROI
        if number not in roi_number_has_contour:
            print(f"Skipping ROI {number} ({name}) because it has no contour data")
            return True

        # Check for ROI names ending with pattern like "_1a" and ensure it matches the plan label
        if plan_suffix:
            match_suffix = re.search(r"_(\d[a-z])$", name_lower)
            if match_suffix and match_suffix.group(0) != plan_suffix:
                print(f"Skipping ROI {number} ({name}) because it does not match the plan suffix")
                return True

        # Special handling for PTV ROIs ending with "+2cm_ph" (should not be skipped due to "_ph")
        if name_lower.startswith("ptv") and name_lower.endswith("+2cm_ph"):
            pass  # allowed
        else:
            if name_lower.startswith("zzz") or name_lower.endswith("_ph"):
                print(f"Skipping ROI {number} ({name})")
                return True

        if name_lower in {"body", "couchsurface", "couchinterior"}:
            print(f"Skipping ROI {number} ({name})")
            return True
        if "ring" in name_lower or "body" in name_lower:
            print(f"Skipping ROI {number} ({name})")
            return True

        return False

    # Helper function: determine if an ROI's Contour Sequence should be removed.
    def skip_contour(name):
        name_lower = name.lower()
        # Keep contours only for PTVs ending with "+2cm_ph"
        if name_lower.startswith("ptv") and not name_lower.endswith("+2cm_ph"):
            return True
        return False

    # --- Step 1: Filter Structure Set ROI Sequence ---
    # Process each ROI item based on its ROI Name (tag 3006,0026).
    # Record its associated ROI Number (tag 3006,0022) and copy the ROI item into the new StructureSetROISequence.
    approved_roi_numbers = set()
    # Build a lookup from ROI Number to ROI Name (for later use in Step 2)
    roi_lookup = {}
    for roi in rtstruct_base.StructureSetROISequence:
        roi_name = getattr(roi, "ROIName", "")
        roi_number = getattr(roi, "ROINumber", None)
        if skip_roi(roi_name, roi_number):
            continue
        if roi_number is not None:
            approved_roi_numbers.add(roi_number)
            roi_lookup[roi_number] = roi_name  # store the name
            new_roi = copy.deepcopy(roi)
            rtstruct_new.StructureSetROISequence.append(new_roi)

    # --- Step 2: Copy and Transform ROI Contour Sequence ---
    # Process ROI contour items only if their Referenced ROI Number (tag 3006,0084)
    # is part of the approved ROI numbers.
    sequence = rtstruct_base.ROIContourSequence
    iterator = tqdm(sequence, desc="ROIContourSequence") if progress_callback is None else sequence
    total = len(sequence)
    for idx, roi_contour in enumerate(iterator, 1):
        # Retrieve the Referenced ROI Number (tag 3006,0084)
        ref_roi_num = getattr(roi_contour, "ReferencedROINumber", None)
        if ref_roi_num not in approved_roi_numbers:
            continue

        # Create a deep copy of the ROI contour to avoid modifying the base file.
        new_roi_contour = copy.deepcopy(roi_contour)

        # Look up the ROI name that corresponds to this contour using the ROI number.
        roi_name = roi_lookup.get(ref_roi_num, "")
        if skip_contour(roi_name):
            # For ROIs starting with "PTV", remove the Contour Sequence.
            print(f"Copying ROI {ref_roi_num} ({roi_name}) without contour sequence")
            new_roi_contour.ContourSequence = pydicom.sequence.Sequence()
        else:
            # Transform the coordinates for each contour within this ROI contour item.
            print(f"Copying ROI {ref_roi_num} ({roi_name})")
            if hasattr(new_roi_contour, "ContourSequence"):
                for contour in new_roi_contour.ContourSequence:
                    current_data = contour.ContourData
                    new_data = transform_contour_points(rigid_transform, current_data)
                    # Convert the transformed coordinates to strings as required by DICOM.
                    contour.ContourData = [str(v) for v in new_data]

        rtstruct_new.ROIContourSequence.append(new_roi_contour)
        if progress_callback:
            progress_callback(idx, total)

    # --- Step 3: Copy RT ROI Observations Sequence ---
    # Each observation item is included only if its Referenced ROI Number (tag 3006,0084)
    # matches one of the approved ROI numbers.
    for obs in rtstruct_base.RTROIObservationsSequence:
        ref_roi_num = getattr(obs, "ReferencedROINumber", None)
        if ref_roi_num not in approved_roi_numbers:
            continue
        new_obs = copy.deepcopy(obs)
        rtstruct_new.RTROIObservationsSequence.append(new_obs)

    # --- Save the Updated RTSTRUCT ---
    output_filename = os.path.join(current_directory, rtstruct_new_filename)
    pydicom.dcmwrite(output_filename, rtstruct_new, write_like_original=False)
    # print(f"Updated RTSTRUCT saved to {output_filename}")
