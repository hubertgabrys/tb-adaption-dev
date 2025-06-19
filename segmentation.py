import datetime
import os

import pydicom
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import generate_uid

from rtstruct_id import create_rtstruct_id


def create_empty_rtstruct(dir_path, series_uid, filepaths):
    """
    Create an empty but DICOM-valid RTSTRUCT file for the specified SeriesInstanceUID.
    """

    first_dataset = pydicom.dcmread(filepaths[0], stop_before_pixels=True)

    rtstruct_uid = generate_uid()
    rtstruct = FileDataset(
        f"RS_{rtstruct_uid}.dcm",
        {},
        file_meta=pydicom.dataset.FileMetaDataset(),
        preamble=b"\0" * 128
    )

    # -- File Meta info
    rtstruct.file_meta.MediaStorageSOPClassUID = pydicom.uid.RTStructureSetStorage
    rtstruct.file_meta.MediaStorageSOPInstanceUID = rtstruct_uid
    rtstruct.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
    rtstruct.file_meta.ImplementationClassUID = "2.16.840.1.114362.1"

    # -- Main SOP common module IDs
    rtstruct.SpecificCharacterSet = "ISO_IR 192"
    rtstruct.InstanceCreationDate = datetime.datetime.now().strftime("%Y%m%d")
    rtstruct.InstanceCreationTime = datetime.datetime.now().strftime("%H%M%S")
    rtstruct.SOPClassUID = rtstruct.file_meta.MediaStorageSOPClassUID
    rtstruct.SOPInstanceUID = rtstruct_uid

    # Patient/Study/Series attributes
    rtstruct.PatientName = getattr(first_dataset, "PatientName", "")
    rtstruct.PatientID = getattr(first_dataset, "PatientID", "")
    rtstruct.StudyInstanceUID = first_dataset.StudyInstanceUID
    rtstruct.SeriesInstanceUID = rtstruct_uid  # new Series for the RTSTRUCT
    rtstruct.Modality = "RTSTRUCT"
    rtstruct.PatientBirthDate = getattr(first_dataset, "PatientBirthDate", "")
    rtstruct.Sex = getattr(first_dataset, "Sex", "")
    rtstruct.AccessionNumber = getattr(first_dataset, "AccessionNumber", "")
    rtstruct.StudyID = getattr(first_dataset, "StudyID", "")
    rtstruct.ReferringPhysicianName = getattr(first_dataset, "ReferringPhysicianName", "")
    rtstruct.PositionReferenceIndicator = ""  # what's this?
    rtstruct.Manufacturer = ""
    rtstruct.InstitutionName = ""

    rtstruct.StudyDate = getattr(first_dataset, "StudyDate", "")
    rtstruct.StudyTime = getattr(first_dataset, "StudyTime", "")
    rtstruct.SeriesDate = getattr(first_dataset, "SeriesDate", "")
    rtstruct.SeriesTime = getattr(first_dataset, "SeriesTime", "")
    rtstruct.SeriesNumber = getattr(first_dataset, "SeriesNumber", 999)
    rtstruct.InstanceNumber = 1

    rtstruct.SeriesDescription = getattr(first_dataset, "SeriesDescription", "")

    rtstruct.StructureSetLabel = create_rtstruct_id(first_dataset)
    # print(rtstruct.StructureSetLabel)
    # rtstruct.StructureSetName = "Empty"
    rtstruct.StructureSetDate = datetime.datetime.now().strftime("%Y%m%d")
    rtstruct.StructureSetTime = datetime.datetime.now().strftime("%H%M%S")

    rtstruct.FrameOfReferenceUID = getattr(first_dataset, "FrameOfReferenceUID", generate_uid())

    # --- ReferencedFrameOfReferenceSequence
    rtstruct.ReferencedFrameOfReferenceSequence = [Dataset()]
    rrfr = rtstruct.ReferencedFrameOfReferenceSequence[0]
    rrfr.FrameOfReferenceUID = rtstruct.FrameOfReferenceUID
    rrfr.RTReferencedStudySequence = [Dataset()]
    rrfr.RTReferencedStudySequence[0].ReferencedSOPInstanceUID = first_dataset.StudyInstanceUID
    rrfr.RTReferencedStudySequence[0].RTReferencedSeriesSequence = [Dataset()]
    rrfr_series = rrfr.RTReferencedStudySequence[0].RTReferencedSeriesSequence[0]
    rrfr_series.SeriesInstanceUID = series_uid

    # Add all referenced images in the ContourImageSequence
    contour_image_sequence = []
    for filepath in filepaths:
        ds = pydicom.dcmread(filepath, stop_before_pixels=True)
        contour_image = Dataset()
        contour_image.ReferencedSOPClassUID = ds.SOPClassUID
        contour_image.ReferencedSOPInstanceUID = ds.SOPInstanceUID
        contour_image_sequence.append(contour_image)
    rrfr_series.ContourImageSequence = contour_image_sequence

    # [Optional but sometimes helpful] Minimal ReferencedStudySequence at top-level
    # This is Type 2 in some modules, so some TPS require it:
    rtstruct.ReferencedStudySequence = [Dataset()]
    rtstruct.ReferencedStudySequence[0].ReferencedSOPClassUID = first_dataset.SOPClassUID
    rtstruct.ReferencedStudySequence[0].ReferencedSOPInstanceUID = first_dataset.StudyInstanceUID

    # --- MANDATORY RTSTRUCT sequences
    # 1) StructureSetROISequence
    rtstruct.StructureSetROISequence = [Dataset()]
    rtstruct.StructureSetROISequence[0].ROINumber = 1
    rtstruct.StructureSetROISequence[0].ReferencedFrameOfReferenceUID = rtstruct.FrameOfReferenceUID
    rtstruct.StructureSetROISequence[0].ROIName = "Dummy_PH"
    rtstruct.StructureSetROISequence[0].ROIGenerationAlgorithm = "MANUAL"

    # 2) ROIContourSequence: must have one item for each ROI
    rtstruct.ROIContourSequence = [Dataset()]
    rtstruct.ROIContourSequence[0].ReferencedROINumber = 1
    rtstruct.ROIContourSequence[0].ROIDisplayColor = [128, 128, 128]  # Type 3
    rtstruct.ROIContourSequence[0].ContourSequence = []  # Type 2, can be empty

    # 3) RTROIObservationsSequence
    rtstruct.RTROIObservationsSequence = [Dataset()]
    rtstruct.RTROIObservationsSequence[0].ObservationNumber = 1
    rtstruct.RTROIObservationsSequence[0].ReferencedROINumber = 1
    rtstruct.RTROIObservationsSequence[0].RTROIInterpretedType = "ORGAN"
    rtstruct.RTROIObservationsSequence[0].ROIInterpreter = ""

    # Save the file
    output_path = os.path.join(dir_path, f"RS_{series_uid}.dcm")
    pydicom.dcmwrite(output_path, rtstruct, write_like_original=False)
    # print(f"Empty RTSTRUCT saved to {output_path}")
