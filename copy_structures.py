import copy
import os

import pydicom
from pydicom.valuerep import DS
from tqdm import tqdm


def float_to_ds_string(x: float, precision: int = 8) -> DS:
    """
    Convert a float to a DICOM DS (Decimal String) compliant value:
      - Fixed‐point notation (no exponent)
      - Up to `precision` decimal places
      - Trim trailing zeros and decimal point
      - Ensure max length 16 chars (raises if exceeded)
    """
    s = f"{x:.{precision}f}".rstrip('0').rstrip('.')
    if len(s) > 16:
        raise ValueError(f"Value '{s}' exceeds 16 characters for DICOM DS")
    return DS(s)


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


def find_rtstruct(directory, description_prefix=None):
    for file_name in os.listdir(directory):
        rtstruct_path = os.path.join(directory, file_name)
        ds = pydicom.dcmread(rtstruct_path, stop_before_pixels=True)
        if ds.Modality != "RTSTRUCT":
            continue
        if description_prefix is None:
            return ds, file_name
        elif hasattr(ds, "SeriesDescription"):
            if ds.SeriesDescription.lower().startswith(description_prefix.lower()):
                return ds, file_name
    return None, None


def read_base_rtstruct(patient_id, rtplan_label):
    base_dir = f"\\\\raoariaapps\\raoariaapps$\\Utilities\\tb_adaption\\base_plans\\{patient_id}\\{rtplan_label}"
    rtstruct, _ = find_rtstruct(base_dir)
    return rtstruct


def read_new_rtstruct(current_directory):
    # Prioritize Synthetic CT HU RTSTRUCT files.
    rtstruct, rtstruct_filename = find_rtstruct(current_directory, "syntheticct hu")
    if rtstruct is not None:
        return rtstruct, rtstruct_filename
    # Fall back to T2_TSE_TRA RTSTRUCT files.
    rtstruct, rtstruct_filename = find_rtstruct(current_directory, "t2_tse_tra")
    return rtstruct, rtstruct_filename


def copy_structures(current_directory, patient_id, rtplan_label, rigid_transform):
    # Read the base and new RTSTRUCT files.
    rtstruct_base = read_base_rtstruct(patient_id, rtplan_label)
    rtstruct_new, rtstruct_new_filename = read_new_rtstruct(current_directory)

    # Get the inverse of the transform
    rigid_transform = rigid_transform.GetInverse()

    # Reset (or initialize) the new RTSTRUCT sequences.
    # We assume these sequences exist so we replace them with new, filtered sequences.
    rtstruct_new.StructureSetROISequence = pydicom.sequence.Sequence()
    rtstruct_new.ROIContourSequence = pydicom.sequence.Sequence()
    rtstruct_new.RTROIObservationsSequence = pydicom.sequence.Sequence()

    # Helper function: determine if an ROI name should be skipped.
    def skip_roi(name):
        name_lower = name.lower()
        # Do not skip if this is the special ROI that always starts with "ptv" and ends with "+2cm_ph"
        if name_lower.startswith("ptv") and name_lower.endswith("+2cm_ph"):
            return False
        # For all other cases, skip if it starts with "zzz_", or ends with "_ph", or is in the specified list
        return name_lower.startswith("zzz_") or name_lower.endswith("_ph") or (name_lower in ["body", "couchsurface", "couchinterior"])

    # Helper function: determine if an ROI's Contour Sequence should be skipped (i.e., left empty).
    def skip_contour(name):
        name_lower = name.lower()
        # If the ROI is the special case (starts with "ptv" and ends with "+2cm_ph"), do not skip it.
        if name_lower.startswith("ptv") and name_lower.endswith("+2cm_ph"):
            return False
        # Otherwise, skip the contour if it starts with "ptv"
        return name_lower.startswith("ptv")

    # --- Step 1: Filter Structure Set ROI Sequence ---
    # Process each ROI item based on its ROI Name (tag 3006,0026).
    # Record its associated ROI Number (tag 3006,0022) and copy the ROI item into the new StructureSetROISequence.
    approved_roi_numbers = set()
    # Build a lookup from ROI Number to ROI Name (for later use in Step 2)
    roi_lookup = {}
    for roi in rtstruct_base.StructureSetROISequence:
        roi_name = getattr(roi, "ROIName", "")
        if skip_roi(roi_name):
            continue
        # Get the ROI Number (tag 3006,0022)
        roi_number = getattr(roi, "ROINumber", None)
        if roi_number is not None:
            approved_roi_numbers.add(roi_number)
            roi_lookup[roi_number] = roi_name  # store the name
            new_roi = copy.deepcopy(roi)
            rtstruct_new.StructureSetROISequence.append(new_roi)

    # --- Step 2: Copy and Transform ROI Contour Sequence ---
    # Process ROI contour items only if their Referenced ROI Number (tag 3006,0084)
    # is part of the approved ROI numbers.
    for roi_contour in tqdm(rtstruct_base.ROIContourSequence, desc="ROIContourSequence"):
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
            new_roi_contour.ContourSequence = pydicom.sequence.Sequence()
        else:
            # Transform the coordinates for each contour within this ROI contour item.
            if hasattr(new_roi_contour, "ContourSequence"):
                for contour in new_roi_contour.ContourSequence:
                    current_data = contour.ContourData
                    new_data = transform_contour_points(rigid_transform, current_data)
                    # Convert the transformed coordinates to strings as required by DICOM.
                    contour.ContourData = [str(v) for v in new_data]

        rtstruct_new.ROIContourSequence.append(new_roi_contour)

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
