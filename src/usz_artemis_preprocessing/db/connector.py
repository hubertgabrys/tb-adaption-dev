import logging
import os
import warnings
from dataclasses import dataclass

import pandas as pd
from pydicom import dcmread
from pydicom.dataset import Dataset
from pynetdicom import AE, evt, AllStoragePresentationContexts
from pynetdicom.sop_class import (
    PatientRootQueryRetrieveInformationModelFind,
    PatientRootQueryRetrieveInformationModelMove,
    RTPlanStorage,
    RTDoseStorage,
)

# Set up logging
logger = logging.getLogger(__name__)


def flatten_list(list_of_lists):
    """Flatten a list of lists into a single list."""
    return [item for sublist in list_of_lists for item in sublist]


def rename_dicom_files_by_modality(output_dir):
    """
    Renames DICOM files in the output directory by adding a modality-based prefix.

    Mapping:
        RTPLAN  -> RP_<original_filename>.dcm
        RTDOSE  -> RDPLAN_<original_filename>.dcm (if DoseSummationType == "PLAN")
                -> RDBEAM_<original_filename>.dcm (if DoseSummationType == "BEAM")
                -> RD_<original_filename>.dcm (otherwise)
        RTSTRUCT -> RS_<original_filename>.dcm
        Any other modality -> <Modality>_<original_filename>.dcm
    """
    modality_prefix_map = {
        "RTPLAN": "RP",
        "RTSTRUCT": "RS",
    }

    for filename in os.listdir(output_dir):
        filepath = os.path.join(output_dir, filename)

        # Ensure the file is a DICOM file
        if not filename.endswith(".dcm"):
            continue

        try:
            # Read DICOM metadata
            dicom_data = dcmread(filepath, stop_before_pixels=True)
            modality = getattr(dicom_data, "Modality", "UNKNOWN")

            # Special handling for RTDOSE
            if modality == "RTDOSE":
                dose_summation_type = getattr(dicom_data, "DoseSummationType", None)
                if dose_summation_type == "PLAN":
                    prefix = "RDPLAN"
                elif dose_summation_type == "BEAM":
                    prefix = "RDBEAM"
                else:
                    prefix = "RD"
            else:
                # Default mapping or fallback to the Modality tag itself
                prefix = modality_prefix_map.get(modality, modality)

            new_filename = f"{prefix}_{filename}"
            new_filepath = os.path.join(output_dir, new_filename)

            # Rename the file
            os.rename(filepath, new_filepath)
            logger.info(f"Renamed {filename} -> {new_filename}")

        except Exception as e:
            logger.warning(f"Failed to rename {filename}: {e}")


@dataclass
class RTPlan:
    patient_id: str
    rtplan_label: str
    rtplan_uid: str
    study_uid: str
    series_uid: str
    rtstruct_uid: str = None


class DBHandler:
    """Handler for managing DICOM database operations using pynetdicom."""

    def __init__(self, server_ip, server_port, called_ae_title, calling_ae_title, scp_port):
        self.server_ip = server_ip
        self.server_port = server_port
        self.called_ae_title = called_ae_title
        self.calling_ae_title = calling_ae_title
        self.scp_port = scp_port

        self.ae = None
        self._initialize_ae()

    def _initialize_ae(self):
        """Initialize the Application Entity (AE) with requested and supported presentation contexts."""
        self.ae = AE(ae_title=self.calling_ae_title)
        # Setting timeouts to None uses the default settings (or you can adjust them)
        self.ae.acse_timeout = None
        self.ae.dimse_timeout = None
        self.ae.network_timeout = None

        # Add Query/Retrieve presentation contexts
        self.ae.add_requested_context(PatientRootQueryRetrieveInformationModelFind)
        self.ae.add_requested_context(PatientRootQueryRetrieveInformationModelMove)

        # Add all Storage presentation contexts so the SCP can receive instances.
        for context in AllStoragePresentationContexts:
            self.ae.add_supported_context(context.abstract_syntax)

    def _acquire_association(self):
        """Establish an association with the remote PACS server."""
        assoc = self.ae.associate(self.server_ip, self.server_port, ae_title=self.called_ae_title)
        if not assoc.is_established:
            raise ConnectionError("Association with the PACS server was rejected or aborted.")
        return assoc

    def _get_plan_information(self, assoc, patient_id_pattern):
        """
        Send a C-FIND request to retrieve RT Plan information based on a patient ID pattern.
        Returns a tuple of (list_of_plans, error_status).
        """
        query_ds = Dataset()
        query_ds.QueryRetrieveLevel = "IMAGE"
        query_ds.PatientID = patient_id_pattern
        query_ds.SOPClassUID = RTPlanStorage

        # Requested attributes – empty strings indicate that we want the value returned.
        query_ds.ApprovalStatus = ""
        query_ds.PlanIntent = ""
        query_ds.SOPInstanceUID = ""
        query_ds.StudyInstanceUID = ""
        query_ds.SeriesInstanceUID = ""
        query_ds.RTPlanLabel = ""
        query_ds.RTPlanDate = ""

        plans = []
        error_status = None

        responses = assoc.send_c_find(query_ds, PatientRootQueryRetrieveInformationModelFind)
        for status, identifier in responses:
            if status is None:
                logger.error("C-FIND: No response status received.")
                continue

            if status.Status in (0xFF00, 0xFF01):
                plans.append({
                    "patient_id": getattr(identifier, "PatientID", "N/A"),
                    "rtplan_label": getattr(identifier, "RTPlanLabel", "N/A"),
                    "rtplan_uid": getattr(identifier, "SOPInstanceUID", "N/A"),
                    "rtplan_date": getattr(identifier, "RTPlanDate", "N/A"),
                    "rtplan_approval": getattr(identifier, "ApprovalStatus", "N/A"),
                    "rtplan_intent": getattr(identifier, "PlanIntent", "N/A"),
                    "study_uid": getattr(identifier, "StudyInstanceUID", "N/A"),
                    "series_uid": getattr(identifier, "SeriesInstanceUID", "N/A"),
                })
            elif status.Status == 0x0000:
                # Completed successfully.
                pass
            else:
                logger.warning(f"C-FIND failed with status: 0x{status.Status:04X}")
                error_status = status.Status

        return plans, error_status

    def _search_for_plans(self, assoc, pattern, max_depth=8):
        """
        Recursively search for RT Plans matching a given patient ID pattern.
        Returns a tuple: (list_of_rtplans, list_of_errors)
        """
        plans, status = self._get_plan_information(assoc, pattern)
        if status != 0xC001:
            return plans, []

        errors = []
        results = []
        # Use a stack for depth-first search
        stack = [(pattern, 0)]
        while stack:
            current_pattern, depth = stack.pop()
            if depth >= max_depth:
                errors.append(current_pattern[1:])
                continue

            for digit in "0123456789":
                new_pattern = f"*{digit}{current_pattern[1:]}"
                plan_list, status = self._get_plan_information(assoc, new_pattern)
                if status == 0xC001:
                    stack.append((new_pattern, depth + 1))
                else:
                    results.append(plan_list)
        return flatten_list(results), errors

    def read_patient(self, pid="*"):
        """
        Query the PACS for patient data matching the given Patient ID pattern.
        Returns a pandas DataFrame containing the RT Plan information.
        """
        assoc = self._acquire_association()
        try:
            rtplans, errors = self._search_for_plans(assoc, pid)
        finally:
            assoc.release()

        if errors:
            warnings.warn(
                "Patient data couldn't be fully extracted. See 'patients_with_errors.txt' for details."
            )
            with open("patients_with_errors.txt", "w") as error_file:
                error_file.write(", ".join(errors))

        data = pd.DataFrame(rtplans)
        if not data.empty:
            # Convert RT Plan date strings to datetime objects (invalid values become NaT)
            data["rtplan_date"] = pd.to_datetime(data["rtplan_date"], errors="coerce")
        return data

    def export_dicom(self, patient_id, rtplan_label, output_dir,
                     to_export=("rtplan", "rtdose", "ct", "rtstruct"), rtplan_uid=None):
        """
        Export DICOM files for a given patient and RT Plan label.
        Depending on the 'to_export' tuple, additional objects (RTSTRUCT, CT, RTDOSE)
        will be retrieved.
        """
        data = self.read_patient(patient_id)
        if data.empty:
            raise ValueError(f"No data found for patient_id {patient_id}")

        patient_data = data[data["rtplan_label"] == rtplan_label]
        if patient_data.empty:
            raise ValueError(
                f"No RT Plan found with label {rtplan_label} for patient_id {patient_id}"
            )

        if len(patient_data) > 1 and rtplan_uid is None:
            raise Exception(
                "More than one plan found and no RTPlan UID was provided. Export manually."
            )

        # Use the provided UID if available, otherwise take the first match.
        plan_uid = rtplan_uid if rtplan_uid else patient_data["rtplan_uid"].values[0]
        rtplan = RTPlan(
            patient_id=patient_id,
            rtplan_label=rtplan_label,
            rtplan_uid=plan_uid,
            study_uid=patient_data["study_uid"].values[0],
            series_uid=patient_data["series_uid"].values[0],
        )

        # Ensure output directory exists.
        os.makedirs(output_dir, exist_ok=True)

        # Start the Storage SCP to receive incoming DICOM files.
        def handle_store(event, store_dir):
            ds = event.dataset
            ds.file_meta = event.file_meta
            filename = os.path.join(store_dir, f"{ds.SOPInstanceUID}.dcm")
            ds.save_as(filename, write_like_original=False)
            logger.info(f"Stored {ds.SOPClassUID.name} file: {filename}")
            return 0x0000  # Success

        handlers = [(evt.EVT_C_STORE, handle_store, [output_dir])]
        scp = self.ae.start_server(("", self.scp_port), block=False, evt_handlers=handlers)
        logger.info(f"Storage SCP started on port {self.scp_port}")

        # Establish an association for C-MOVE operations.
        assoc = self._acquire_association()
        try:
            logger.info(f"Retrieving RT Plan: Label={rtplan_label}, UID={rtplan.rtplan_uid}")
            self._export_rtplan(assoc, rtplan)

            if "rtstruct" in to_export:
                self._export_rtstruct(assoc, rtplan, output_dir)

            if "ct" in to_export:
                self._export_ct(assoc, rtplan, output_dir)

            if "rtdose" in to_export:
                self._export_rtdose(assoc, rtplan, output_dir, dose_summation_type="PLAN")

            if "rtdose_beam" in to_export:
                self._export_rtdose(assoc, rtplan, output_dir, dose_summation_type="BEAM")
        except Exception as e:
            logger.error(f"Error during export: {e}")
            raise
        finally:
            assoc.release()
            scp.shutdown()
            logger.info("Associations closed and SCP shut down.")

    def _send_cmove(self, assoc, query_ds, description):
        """
        Helper method to send a C-MOVE request and log responses.
        """
        responses = assoc.send_c_move(
            query_ds,
            move_aet=self.calling_ae_title,
            query_model=PatientRootQueryRetrieveInformationModelMove,
        )
        for status, identifier in responses:
            if status is None:
                logger.error(f"{description}: No status returned.")
                continue
            if status.Status in (0xFF00, 0xFF01):
                # Pending responses.
                continue
            elif status.Status == 0x0000:
                logger.info(f"{description} retrieval completed successfully.")
            else:
                logger.warning(f"{description} retrieval failed with status 0x{status.Status:04X}")
        return responses

    def _export_rtplan(self, assoc, rtplan):
        """Retrieve the RT Plan from the PACS using C-MOVE."""
        move_ds = Dataset()
        move_ds.QueryRetrieveLevel = "IMAGE"
        move_ds.PatientID = rtplan.patient_id
        move_ds.SOPClassUID = RTPlanStorage

        # Specify the RT Plan to retrieve.
        move_ds.SOPInstanceUID = rtplan.rtplan_uid
        # Other attributes can be left empty.
        move_ds.StudyInstanceUID = ""
        move_ds.SeriesInstanceUID = ""
        move_ds.RTPlanLabel = ""
        move_ds.RTPlanDate = ""

        self._send_cmove(assoc, move_ds, f"RT Plan (UID: {rtplan.rtplan_uid})")

    def _export_rtstruct(self, assoc, rtplan, output_dir):
        """
        Retrieve the RTSTRUCT corresponding to the RT Plan.
        Assumes the RT Plan file has been retrieved and stored.
        """
        rtplan_path = os.path.join(output_dir, f"{rtplan.rtplan_uid}.dcm")
        if not os.path.exists(rtplan_path):
            raise FileNotFoundError(f"RT Plan file {rtplan_path} not found.")

        rtplan_dcm = dcmread(rtplan_path)
        try:
            rtplan.rtstruct_uid = rtplan_dcm.ReferencedStructureSetSequence[0].ReferencedSOPInstanceUID
        except (AttributeError, IndexError) as e:
            raise ValueError("RT Plan does not contain a valid ReferencedStructureSetSequence.") from e

        move_ds = Dataset()
        move_ds.QueryRetrieveLevel = "IMAGE"
        move_ds.PatientID = rtplan.patient_id
        move_ds.SOPInstanceUID = rtplan.rtstruct_uid

        self._send_cmove(assoc, move_ds, f"RTSTRUCT (UID: {rtplan.rtstruct_uid})")

    def _export_ct(self, assoc, rtplan, output_dir):
        """
        Retrieve CT images referenced in the RTSTRUCT.
        """
        rtstruct_path = os.path.join(output_dir, f"{rtplan.rtstruct_uid}.dcm")
        if not os.path.exists(rtstruct_path):
            raise FileNotFoundError(f"RTSTRUCT file {rtstruct_path} not found.")

        rtstruct = dcmread(rtstruct_path)
        ct_series_uids = set()
        if hasattr(rtstruct, "ReferencedFrameOfReferenceSequence"):
            for frame_ref in rtstruct.ReferencedFrameOfReferenceSequence:
                if hasattr(frame_ref, "RTReferencedStudySequence"):
                    for study_ref in frame_ref.RTReferencedStudySequence:
                        if hasattr(study_ref, "RTReferencedSeriesSequence"):
                            for series_ref in study_ref.RTReferencedSeriesSequence:
                                series_uid = getattr(series_ref, "SeriesInstanceUID", None)
                                if series_uid:
                                    ct_series_uids.add(series_uid)
                                    logger.info(f"Found referenced CT Series UID: {series_uid}")
        else:
            logger.info("No ReferencedFrameOfReferenceSequence found in RTSTRUCT.")

        for series_uid in ct_series_uids:
            move_ds = Dataset()
            move_ds.QueryRetrieveLevel = "SERIES"
            move_ds.PatientID = rtplan.patient_id
            move_ds.SeriesInstanceUID = series_uid

            self._send_cmove(assoc, move_ds, f"CT Series (UID: {series_uid})")

    def _export_rtdose(self, assoc, rtplan, output_dir, dose_summation_type="PLAN"):
        """
        Retrieve RTDOSE objects for the given RT Plan.
        First, attempt to retrieve any referenced dose via the RT Plan’s ReferencedDoseSequence.
        If not found, perform a C-FIND and then a C-MOVE for candidate RTDOSE objects.
        """
        rtplan_path = os.path.join(output_dir, f"{rtplan.rtplan_uid}.dcm")
        if not os.path.exists(rtplan_path):
            raise FileNotFoundError(f"RT Plan file {rtplan_path} not found.")

        rtplan_dcm = dcmread(rtplan_path)
        dose_uids = []

        def retrieve_dose(sop_uid):
            logger.info(f"Retrieving RTDOSE (UID: {sop_uid})")
            move_ds = Dataset()
            move_ds.QueryRetrieveLevel = "IMAGE"
            move_ds.PatientID = rtplan.patient_id
            move_ds.SOPInstanceUID = sop_uid

            self._send_cmove(assoc, move_ds, f"RTDOSE (UID: {sop_uid})")

        # First, try to get dose UIDs from ReferencedDoseSequence.
        if hasattr(rtplan_dcm, "ReferencedDoseSequence"):
            for ref in rtplan_dcm.ReferencedDoseSequence:
                dose_uid = getattr(ref, "ReferencedSOPInstanceUID", None)
                if dose_uid:
                    logger.info(f"Found RTDOSE UID in RT Plan: {dose_uid}")
                    dose_uids.append(dose_uid)
                    retrieve_dose(dose_uid)
            if not dose_uids:
                logger.info("RT Plan has ReferencedDoseSequence but no valid dose UIDs were found.")
        else:
            logger.info("RT Plan does not have a ReferencedDoseSequence.")

        # If no dose UIDs were found, perform a C-FIND query.
        if not dose_uids:
            logger.info("Querying for RTDOSE objects referencing the RT Plan.")
            assoc_find = self.ae.associate(self.server_ip, self.server_port, ae_title=self.called_ae_title)
            try:
                query_ds = Dataset()
                query_ds.QueryRetrieveLevel = "IMAGE"
                query_ds.DoseSummationType = dose_summation_type
                query_ds.SOPClassUID = RTDoseStorage
                query_ds.PatientID = rtplan.patient_id
                query_ds.StudyInstanceUID = rtplan.study_uid
                query_ds.SOPInstanceUID = ""

                responses = assoc_find.send_c_find(query_ds, PatientRootQueryRetrieveInformationModelFind)
                candidate_uids = set()
                for status, identifier in responses:
                    if status is None:
                        logger.error("C-FIND for RTDOSE: No status returned.")
                        continue
                    if status.Status in (0xFF00, 0xFF01) and identifier:
                        found_uid = getattr(identifier, "SOPInstanceUID", None)
                        if found_uid:
                            candidate_uids.add(found_uid)
                            logger.info(f"Candidate RTDOSE UID: {found_uid}")
                    elif status.Status == 0x0000:
                        logger.info("C-FIND for RTDOSE completed successfully.")
                    else:
                        logger.warning(f"C-FIND for RTDOSE failed with status 0x{status.Status:04X}")
            finally:
                assoc_find.release()

            # Retrieve each candidate and verify if it references the RT Plan.
            for dose_uid in candidate_uids:
                retrieve_dose(dose_uid)
                dose_path = os.path.join(output_dir, f"{dose_uid}.dcm")
                if os.path.exists(dose_path):
                    rtdose = dcmread(dose_path)
                    if hasattr(rtdose, "ReferencedRTPlanSequence"):
                        for ref_plan in rtdose.ReferencedRTPlanSequence:
                            if getattr(ref_plan, "ReferencedSOPInstanceUID", None) == rtplan.rtplan_uid:
                                logger.info(f"RTDOSE {dose_uid} correctly references RT Plan {rtplan.rtplan_uid}.")
                                dose_uids.append(dose_uid)
                                break
                        else:
                            logger.info(f"RTDOSE {dose_uid} does not reference the RT Plan; removing file.")
                            os.remove(dose_path)
                    else:
                        logger.info(f"RTDOSE {dose_uid} has no ReferencedRTPlanSequence; file ignored.")
                else:
                    logger.info(f"RTDOSE file {dose_path} not found.")
