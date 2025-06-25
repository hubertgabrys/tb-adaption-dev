import datetime
import os
import socket
from pathlib import Path

import SimpleITK as sitk
import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileDataset
from pydicom.sequence import Sequence
from pydicom.uid import generate_uid, ExplicitVRLittleEndian
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from utils import get_datetime, load_environment

from dbconnector import DBHandler

# --------------------------------------------------------------------
# Helper functions
# --------------------------------------------------------------------

load_environment(".env")

def get_base_plan(patient_id, rtplan_label, rtplan_uid):
    baseplan_dir = Path(os.environ.get('BASEPLAN_DIR'))
    moving_dir = baseplan_dir / patient_id / rtplan_label
    # moving_dir = f"\\\\raoariaapps\\raoariaapps$\\Utilities\\tb_adaption\\base_plans\\{patient_id}\\{rtplan_label}"
    if os.path.isdir(moving_dir):
        print(f"{get_datetime()} Base plan exists")

    else:
        print(f"{get_datetime()} Downloading the base plan")
        dbh = DBHandler(
            server_ip=os.environ.get('SERVER_IP'),
            server_port=int(os.environ.get('SERVER_PORT')),
            called_ae_title=os.environ.get('SERVER_AE_TITLE'),
            calling_ae_title=socket.gethostname(),
            scp_port=int(os.environ.get('SCP_PORT')))
        dbh.export_dicom(patient_id, rtplan_label, moving_dir, to_export=("rtplan", "ct", "rtstruct"),
                         rtplan_uid=rtplan_uid)


def read_dicom_series(directory, modality="CT", series_uid=None):
    reader = sitk.ImageSeriesReader()
    series_IDs = reader.GetGDCMSeriesIDs(directory)
    if not series_IDs:
        raise ValueError(f"No DICOM series found in directory: {directory}")

    if series_uid:
        if series_uid not in series_IDs:
            raise ValueError(
                f"Series UID {series_uid} not found in directory: {directory}"
            )
        file_names = reader.GetGDCMSeriesFileNames(directory, series_uid)
        reader.SetFileNames(file_names)
        image = reader.Execute()
        return image, file_names, series_uid

    # Set the search patterns based on modality.
    if modality == "CT":
        patterns = ["SyntheticCT HU", "Synthetic CT", ""]
        print("Using CT for registration.")
    elif modality == "MR":
        patterns = ["t2_tse_tra_warp", "t2_tse_tra"]
        print("Using MR for registration.")
    else:
        raise ValueError("Unsupported modality. Please choose 'CT' or 'MR'.")

    selected_series = None
    selected_series_file_names = None

    # Try each pattern in priority order.
    for pattern in patterns:
        for series_id in series_IDs:
            file_names = reader.GetGDCMSeriesFileNames(directory, series_id)
            if not file_names:
                continue

            # Use a lightweight image file reader to extract metadata from the first file.
            meta_reader = sitk.ImageFileReader()
            meta_reader.SetFileName(file_names[0])
            meta_reader.LoadPrivateTagsOn()
            meta_reader.ReadImageInformation()
            series_description = ""
            if meta_reader.HasMetaDataKey("0008|103e"):
                series_description = meta_reader.GetMetaData("0008|103e")

            if series_description.startswith(pattern):
                selected_series = series_id
                selected_series_file_names = file_names
                break  # Break from inner loop if match is found.

        if selected_series is not None:
            # A matching series was found for the current pattern.
            break

    if selected_series is None:
        raise ValueError(
            f"No series with a description starting with one of {patterns} found "
            f"in directory: {directory}. Series found: {series_IDs}"
        )

    # Read the selected series.
    reader.SetFileNames(selected_series_file_names)
    image = reader.Execute()
    return image, selected_series_file_names, selected_series


def extract_metadata(dicom_file):
    ds = pydicom.dcmread(dicom_file, stop_before_pixels=True)
    meta = {}
    meta['PatientName'] = ds.get("PatientName", "Anonymous")
    meta['PatientID'] = ds.get("PatientID", "Anonymous")
    meta['StudyDescription'] = ds.get("StudyDescription", "Anonymous")
    meta['StudyDate'] = ds.get("StudyDate", "Anonymous")
    meta['StudyTime'] = ds.get("StudyTime", "Anonymous")
    meta['StudyInstanceUID'] = ds.get("StudyInstanceUID", generate_uid())
    meta['SeriesInstanceUID'] = ds.get("SeriesInstanceUID", generate_uid())
    meta['FrameOfReferenceUID'] = ds.get("FrameOfReferenceUID", generate_uid())
    meta['AccessionNumber'] = ds.get("AccessionNumber", "Anonymous")
    meta['PatientBirthDate'] = ds.get("PatientBirthDate", "Anonymous")
    meta['PatientSex'] = ds.get("PatientSex", "Anonymous")
    meta['PatientAge'] = ds.get("PatientAge", "Anonymous")
    meta['StudyID'] = ds.get("StudyID", "Anonymous")
    return meta

def perform_rigid_registration(fixed_image, moving_image, initial_transform):
    print(f"{get_datetime()} Initializing registration...")

    # Clip intensities to [-160, 240] to reduce outliers
    clip = False
    if clip:
        fixed_processed = sitk.Clamp(fixed_image, lowerBound=-160, upperBound=240)
        moving_processed = sitk.Clamp(moving_image, lowerBound=-160, upperBound=240)
    else:
        fixed_processed = fixed_image
        moving_processed = moving_image

    # # Geometry-based initializer using a VersorRigid3DTransform
    # initial_transform = sitk.CenteredTransformInitializer(
    #     fixed_processed,
    #     moving_processed,
    #     sitk.VersorRigid3DTransform(),
    #     sitk.CenteredTransformInitializerFilter.MOMENTS
    # )

    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetMetricAsCorrelation()
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.1, seed=42)
    registration_method.SetInterpolator(sitk.sitkLinear)
    registration_method.SetOptimizerAsRegularStepGradientDescent(
        learningRate=1.0,
        minStep=1e-6,
        numberOfIterations=200,
        gradientMagnitudeTolerance=1e-6
    )
    registration_method.SetOptimizerScalesFromPhysicalShift()
    registration_method.SetInitialTransform(initial_transform, inPlace=False)
    registration_method.SetShrinkFactorsPerLevel([4, 2, 1])
    registration_method.SetSmoothingSigmasPerLevel([2, 1, 0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    print(f"{get_datetime()} Performing registration...")
    final_transform = registration_method.Execute(fixed_processed, moving_processed)
    print(f"{get_datetime()} Registration completed.")

    return final_transform

# --------------------------------------------------------------------
# Transform extraction & DICOM REG creation
# --------------------------------------------------------------------
def get_final_rigid_transform(transform):
    """
    If the transform is composite, return its last sub-transform.
    Otherwise, return the transform itself.
    """
    if isinstance(transform, sitk.CompositeTransform):
        n = transform.GetNumberOfTransforms()
        if n == 0:
            raise ValueError("CompositeTransform is empty!")
        return transform.GetNthTransform(n - 1)
    else:
        return transform

def create_registration_file(output_reg_file, final_transform, fixed_meta, moving_meta,
                             fixed_files, moving_files):
    """
    Create a DICOM Spatial Registration file containing the transform.
    """
    def transformation_matrix():
        # Extract the rigid transform from the (possibly composite) transform
        rigid_transform = get_final_rigid_transform(final_transform)

        # Invert the rigid transform so that the matrix maps from moving -> fixed
        rigid_transform = rigid_transform.GetInverse()

        # Extract rotation (3x3) and translation (3-element)
        rot = np.array(rigid_transform.GetMatrix()).reshape(3, 3)
        trans = np.array(rigid_transform.GetTranslation())

        # Build a 4x4 homogeneous transformation matrix in row-major order
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = rot
        T[:3, 3] = trans

        # Flatten in row-major (C-order)
        mat_list = T.flatten(order='C').tolist()
        return mat_list

    # Helper: create referenced image sequence
    def create_referenced_image_sequence(dicom_files):
        seq = Sequence()
        for dcm_file in dicom_files[::-1]:
            dcm_temp = pydicom.dcmread(dcm_file, stop_before_pixels=True)
            ref_instance = Dataset()
            ref_instance.ReferencedSOPClassUID = dcm_temp.SOPClassUID
            ref_instance.ReferencedSOPInstanceUID = dcm_temp.SOPInstanceUID
            seq.append(ref_instance)
        return seq

    def generate_referenced_series_sequence():
        seq = Sequence()
        item = Dataset()
        item.ReferencedInstanceSequence = create_referenced_image_sequence(fixed_files)
        item.SeriesInstanceUID = fixed_meta['SeriesInstanceUID']
        seq.append(item)
        return seq

    def generate_scoris():
        seq = Sequence()
        item = Dataset()
        item.ReferencedSeriesSequence = Sequence()
        item.StudyInstanceUID = moving_meta['StudyInstanceUID']
        seq.append(item)
        ref_series_item = Dataset()
        ref_series_item.ReferencedInstanceSequence = create_referenced_image_sequence(moving_files)
        ref_series_item.SeriesInstanceUID = moving_meta['SeriesInstanceUID']
        item.ReferencedSeriesSequence.append(ref_series_item)
        return seq

    def generate_registration_sequence():
        registration_sequence = Sequence()

        # (A) Fixed reference item with identity transform
        registration_item_fixed = Dataset()
        registration_item_fixed.ReferencedImageSequence = create_referenced_image_sequence(fixed_files)
        registration_item_fixed.FrameOfReferenceUID = fixed_meta['FrameOfReferenceUID']
        registration_item_fixed.MatrixRegistrationSequence = Sequence()

        matrix_reg_item_fixed = Dataset()
        matrix_reg_item_fixed.MatrixSequence = Sequence()
        matrix_reg_item_fixed.RegistrationTypeCodeSequence = Sequence()

        matrix_seq_fixed = Dataset()
        matrix_seq_fixed.FrameOfReferenceTransformationMatrixType = "RIGID"
        matrix_seq_fixed.FrameOfReferenceTransformationMatrix = [
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0
        ]

        reg_type_fixed = Dataset()
        reg_type_fixed.CodeValue = "125025"
        reg_type_fixed.CodingSchemeDesignator = "DCM"
        reg_type_fixed.CodeMeaning = "Visual Alignment"

        matrix_reg_item_fixed.MatrixSequence.append(matrix_seq_fixed)
        matrix_reg_item_fixed.RegistrationTypeCodeSequence.append(reg_type_fixed)
        registration_item_fixed.MatrixRegistrationSequence.append(matrix_reg_item_fixed)
        registration_sequence.append(registration_item_fixed)

        # (B) Moving reference item with actual transform
        registration_item_moving = Dataset()
        registration_item_moving.ReferencedImageSequence = create_referenced_image_sequence(moving_files)
        registration_item_moving.FrameOfReferenceUID = moving_meta['FrameOfReferenceUID']
        registration_item_moving.MatrixRegistrationSequence = Sequence()

        matrix_reg_item_moving = Dataset()
        matrix_reg_item_moving.MatrixSequence = Sequence()
        matrix_reg_item_moving.RegistrationTypeCodeSequence = Sequence()

        matrix_seq_moving = Dataset()
        matrix_seq_moving.FrameOfReferenceTransformationMatrixType = "RIGID"
        mat_list = transformation_matrix()  # 4x4 in row-major order
        matrix_seq_moving.FrameOfReferenceTransformationMatrix = mat_list

        reg_type_moving = Dataset()
        reg_type_moving.CodeValue = "125025"
        reg_type_moving.CodingSchemeDesignator = "DCM"
        reg_type_moving.CodeMeaning = "Visual Alignment"

        matrix_reg_item_moving.MatrixSequence.append(matrix_seq_moving)
        matrix_reg_item_moving.RegistrationTypeCodeSequence.append(reg_type_moving)
        registration_item_moving.MatrixRegistrationSequence.append(matrix_reg_item_moving)
        registration_sequence.append(registration_item_moving)

        return registration_sequence

    # Create the REG dataset
    dt = datetime.datetime.now()
    date_str = dt.strftime("%Y%m%d")
    time_str = dt.strftime("%H%M%S")
    sop_instance_uid = generate_uid()

    file_meta = Dataset()
    file_meta.FileMetaInformationVersion = b"\x00\x01"
    file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.66.1'
    file_meta.MediaStorageSOPInstanceUID = sop_instance_uid
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    # file_meta.TransferSyntaxUID = ImplicitVRLittleEndian
    file_meta.ImplementationClassUID = generate_uid()

    ds = FileDataset(output_reg_file, {}, file_meta=file_meta, preamble=b"\0" * 128)

    ds.SpecificCharacterSet = "ISO_IR 192"
    ds.InstanceCreationDate = date_str
    ds.InstanceCreationTime = time_str
    ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = sop_instance_uid
    ds.StudyDate = fixed_meta.get('StudyDate', date_str)
    ds.SeriesDate = date_str
    ds.ContentDate = date_str
    ds.StudyTime = fixed_meta.get('StudyTime', time_str)
    ds.SeriesTime = time_str
    ds.ContentTime = time_str
    ds.AccessionNumber = fixed_meta.get('AccessionNumber', "")
    ds.Modality = "REG"
    ds.Manufacturer = ""
    ds.InstitutionName = "USZ"
    ds.StudyDescription = fixed_meta.get('StudyDescription', "")
    ds.SeriesDescription = "Spatial Registration (Rigid)"

    ds.PatientName = fixed_meta.get('PatientName', "Anonymous")
    ds.PatientID = fixed_meta.get('PatientID', "UnknownID")
    ds.PatientBirthDate = fixed_meta.get('PatientBirthDate', "")
    ds.PatientSex = fixed_meta.get('PatientSex', "")
    ds.PatientAge = ""

    ds.SoftwareVersions = "0.1.0-dev"
    ds.StudyInstanceUID = fixed_meta['StudyInstanceUID']
    ds.SeriesInstanceUID = generate_uid()
    ds.StudyID = fixed_meta['StudyID']
    ds.SeriesNumber = "1"
    ds.InstanceNumber = "1"
    ds.FrameOfReferenceUID = fixed_meta['FrameOfReferenceUID']
    ds.PositionReferenceIndicator = ""

    ds.ContentLabel = "REGISTRATION"
    ds.ContentDescription = "Rigid registration of CT scans"
    ds.ContentCreatorName = ""

    ds.ReferencedSeriesSequence = generate_referenced_series_sequence()
    ds.StudiesContainingOtherReferencedInstancesSequence = generate_scoris()
    ds.RegistrationSequence = generate_registration_sequence()

    ds.is_little_endian = True
    ds.is_implicit_VR = False
    ds.save_as(output_reg_file)
    print(f"{get_datetime()} DICOM registration file saved to:", output_reg_file)

# --------------------------------------------------------------------
# Visualization
# --------------------------------------------------------------------
class MultiViewOverlay:
    def __init__(self, fixed_array, moving_array, spacing,
                 fixed_modality="CT", moving_modality="CT"):
        self.fixed = fixed_array
        self.moving = moving_array
        # spacing comes from SimpleITK in (x, y, z) order
        self.spacing = spacing
        self.alpha = 0.5

        self.fixed_modality = fixed_modality
        self.moving_modality = moving_modality

        self.fixed_vmin, self.fixed_vmax = self._compute_range(self.fixed,
                                                                self.fixed_modality)
        self.moving_vmin, self.moving_vmax = self._compute_range(self.moving,
                                                                  self.moving_modality)

        self.cmap_fixed = plt.get_cmap("gray")
        self.cmap_moving = plt.get_cmap("hot")

        # initial slice indices
        self.slice_z = self.fixed.shape[0] // 2
        self.slice_y = self.fixed.shape[1] // 2
        self.slice_x = self.fixed.shape[2] // 2

        # manual shifts (in slices)
        self.shift_z = 0
        self.shift_y = 0
        self.shift_x = 0
        max_shift_z = max(self.fixed.shape[0], self.moving.shape[0]) // 2
        max_shift_y = max(self.fixed.shape[1], self.moving.shape[1]) // 2
        max_shift_x = max(self.fixed.shape[2], self.moving.shape[2]) // 2

        # compute extents for aspect-correct display based on the
        # combined range of both images
        range_x = max(self.fixed.shape[2], self.moving.shape[2]) * self.spacing[0]
        range_y = max(self.fixed.shape[1], self.moving.shape[1]) * self.spacing[1]
        range_z = max(self.fixed.shape[0], self.moving.shape[0]) * self.spacing[2]

        self.extent_transverse = [0, range_x, 0, range_y]
        self.extent_coronal = [0, range_x, 0, range_z]
        self.extent_sagittal = [0, range_y, 0, range_z]

        # show larger views for easier inspection
        self.fig, self.axes = plt.subplots(1, 3, figsize=(20, 7))
        self.ax_transverse, self.ax_coronal, self.ax_sagittal = self.axes
        # step size for scroll interactions
        self.scroll_speed = 2

        for ax in self.axes:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect('equal')

        # initial images
        self.im_transverse = self.ax_transverse.imshow(
            self.get_transverse_slice(),
            origin='upper',
            extent=self.extent_transverse,
        )
        self.ax_transverse.set_title('Transverse (Axial)')
        self.im_coronal = self.ax_coronal.imshow(
            self.get_coronal_slice(),
            origin='lower',
            extent=self.extent_coronal,
        )
        self.ax_coronal   .set_title('Coronal')
        self.im_sagittal = self.ax_sagittal.imshow(
            self.get_sagittal_slice(),
            origin='lower',
            extent=self.extent_sagittal,
        )
        self.ax_sagittal .set_title('Sagittal')

        # slice text
        self.text_transverse = self.ax_transverse.text(
            0.05, 0.95,
            f"Slice {self.slice_z + 1} / {self.fixed.shape[0]}",
            transform=self.ax_transverse.transAxes, color='yellow', fontsize=10, verticalalignment='top'
        )
        self.text_coronal = self.ax_coronal.text(
            0.05, 0.95,
            f"Slice {self.slice_y + 1} / {self.fixed.shape[1]}",
            transform=self.ax_coronal.transAxes, color='yellow', fontsize=10, verticalalignment='top'
        )
        self.text_sagittal = self.ax_sagittal.text(
            0.05, 0.95,
            f"Slice {self.slice_x + 1} / {self.fixed.shape[2]}",
            transform=self.ax_sagittal.transAxes, color='yellow', fontsize=10, verticalalignment='top'
        )

        # alpha slider (overlay blend)
        slider_ax = self.fig.add_axes([0.25, 0.11, 0.5, 0.03])
        self.slider_alpha = Slider(slider_ax, 'Overlay', 0.0, 1.0, valinit=self.alpha)
        self.slider_alpha.on_changed(self.update_alpha)

        # shift sliders (X above Y above Z)
        shift_ax_x = self.fig.add_axes([0.25, 0.07, 0.5, 0.03])
        self.slider_shift_x = Slider(
            shift_ax_x,
            'X Shift (slices)',
            -max_shift_x, max_shift_x,
            valinit=self.shift_x,
            valstep=1
        )
        self.slider_shift_x.on_changed(self.update_shift_x)

        shift_ax_y = self.fig.add_axes([0.25, 0.04, 0.5, 0.03])
        self.slider_shift_y = Slider(
            shift_ax_y,
            'Y Shift (slices)',
            -max_shift_y, max_shift_y,
            valinit=self.shift_y,
            valstep=1
        )
        self.slider_shift_y.on_changed(self.update_shift_y)

        shift_ax_z = self.fig.add_axes([0.25, 0.01, 0.5, 0.03])
        self.slider_shift_z = Slider(
            shift_ax_z,
            'Z Shift (slices)',
            -max_shift_z, max_shift_z,
            valinit=self.shift_z,
            valstep=1
        )
        self.slider_shift_z.on_changed(self.update_shift_z)

        self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)
        plt.show()

    def _compute_range(self, array, modality):
        if modality == "CT":
            return -160.0, 240.0
        lo = np.percentile(array, 5)
        hi = np.percentile(array, 95)
        if lo == hi:
            lo = float(np.min(array))
            hi = float(np.max(array))
        return float(lo), float(hi)

    def apply_colormap(self, image_slice, cmap, vmin, vmax):
        normed = (image_slice - vmin) / (vmax - vmin)
        normed = np.clip(normed, 0, 1)
        rgba = cmap(normed)
        return rgba[..., :3]

    def blend_slices(self, fixed_slice, moving_slice):
        fixed_rgb  = self.apply_colormap(fixed_slice,  self.cmap_fixed,
                                         self.fixed_vmin, self.fixed_vmax)
        moving_rgb = self.apply_colormap(moving_slice, self.cmap_moving,
                                         self.moving_vmin, self.moving_vmax)
        return (1 - self.alpha) * fixed_rgb + self.alpha * moving_rgb

    def get_transverse_slice(self):
        z = int(self.slice_z)
        f_slc = self.fixed[z, :, :]
        m_vol = np.roll(self.moving, int(self.shift_z), axis=0)
        m_slc = m_vol[z, :, :]
        m_slc = np.roll(m_slc, int(self.shift_y), axis=0)
        m_slc = np.roll(m_slc, int(self.shift_x), axis=1)
        return self.blend_slices(f_slc, m_slc)

    def get_coronal_slice(self):
        # shift affects Z (axis 0) and Y (axis 1)
        f_slc = self.fixed[:, self.slice_y, :]
        m_vol = np.roll(self.moving, int(self.shift_z), axis=0)
        m_vol = np.roll(m_vol, int(self.shift_y), axis=1)
        m_vol = np.roll(m_vol, int(self.shift_x), axis=2)
        m_slc = m_vol[:, self.slice_y, :]
        return self.blend_slices(f_slc, m_slc)

    def get_sagittal_slice(self):
        f_slc = self.fixed[:, :, self.slice_x]
        m_vol = np.roll(self.moving, int(self.shift_z), axis=0)
        m_vol = np.roll(m_vol, int(self.shift_y), axis=1)
        m_vol = np.roll(m_vol, int(self.shift_x), axis=2)
        m_slc = m_vol[:, :, self.slice_x]
        return self.blend_slices(f_slc, m_slc)

    def update_alpha(self, val):
        self.alpha = val
        self.update_display()

    def update_shift_z(self, val):
        self.shift_z = val
        self.update_display()

    def update_shift_y(self, val):
        self.shift_y = val
        self.update_display()

    def update_shift_x(self, val):
        self.shift_x = val
        self.update_display()

    def update_display(self):
        self.im_transverse.set_data(self.get_transverse_slice())
        self.im_coronal.set_data(self.get_coronal_slice())
        self.im_sagittal.set_data(self.get_sagittal_slice())
        # Ensure extents are applied when the image updates
        self.im_transverse.set_extent(self.extent_transverse)
        self.im_coronal.set_extent(self.extent_coronal)
        self.im_sagittal.set_extent(self.extent_sagittal)

        self.text_transverse.set_text(f"Slice {self.slice_z + 1} / {self.fixed.shape[0]}")
        self.text_coronal   .set_text(f"Slice {self.slice_y + 1} / {self.fixed.shape[1]}")
        self.text_sagittal  .set_text(f"Slice {self.slice_x + 1} / {self.fixed.shape[2]}")

        self.fig.canvas.draw_idle()

    def on_scroll(self, event):
        delta = self.scroll_speed if event.button == 'up' else -self.scroll_speed
        if event.inaxes == self.ax_transverse:
            self.slice_z = np.clip(self.slice_z + delta, 0, self.fixed.shape[0] - 1)
        elif event.inaxes == self.ax_coronal:
            self.slice_y = np.clip(self.slice_y + delta, 0, self.fixed.shape[1] - 1)
        elif event.inaxes == self.ax_sagittal:
            self.slice_x = np.clip(self.slice_x + delta, 0, self.fixed.shape[2] - 1)
        self.update_display()

# --------------------------------------------------------------------
# Main
# --------------------------------------------------------------------
def run_viewer(fixed_image, moving_image,
               fixed_modality="CT", moving_modality="CT"):
    fixed_array = sitk.GetArrayFromImage(fixed_image)
    moving_array = sitk.GetArrayFromImage(moving_image)
    spacing = fixed_image.GetSpacing()  # (x, y, z)
    overlay = MultiViewOverlay(
        fixed_array,
        moving_array,
        spacing,
        fixed_modality=fixed_modality,
        moving_modality=moving_modality,
    )
    return overlay.shift_z, overlay.shift_y, overlay.shift_x

def perform_registration(current_directory, patient_id, rtplan_label,
                         selected_series_uid=None, selected_modality=None,
                         moving_series_uid=None, moving_modality=None,
                         confirm_fn=None):
    fixed_dir = current_directory
    baseplan_dir = Path(os.environ.get('BASEPLAN_DIR'))
    moving_dir = baseplan_dir / patient_id / rtplan_label
    # moving_dir = f"\\\\raoariaapps\\raoariaapps$\\Utilities\\tb_adaption\\base_plans\\{patient_id}\\{rtplan_label}"
    output_reg_file = current_directory + "\\REG.dcm"

    print(f"{get_datetime()} Reading fixed image from:", fixed_dir)
    if selected_series_uid:
        fixed_modality = selected_modality or "CT"
        fixed_image, fixed_files, used_fixed_uid = read_dicom_series(
            fixed_dir,
            modality=fixed_modality,
            series_uid=selected_series_uid,
        )
    else:
        try:
            fixed_modality = "CT"
            fixed_image, fixed_files, used_fixed_uid = read_dicom_series(fixed_dir, "CT")
        except ValueError:
            fixed_modality = "MR"
            fixed_image, fixed_files, used_fixed_uid = read_dicom_series(fixed_dir, "MR")
    print(f"{get_datetime()} Reading moving image from:", moving_dir)
    if moving_series_uid:
        moving_modality = moving_modality or "CT"
        moving_image, moving_files, used_moving_uid = read_dicom_series(
            moving_dir,
            modality=moving_modality,
            series_uid=moving_series_uid,
        )
    else:
        moving_modality = moving_modality or "CT"
        moving_image, moving_files, used_moving_uid = read_dicom_series(moving_dir, moving_modality)

    # Cast & orient
    fixed_image = sitk.Cast(sitk.DICOMOrient(fixed_image, 'LPS'), sitk.sitkFloat32)
    moving_image = sitk.Cast(sitk.DICOMOrient(moving_image, 'LPS'), sitk.sitkFloat32)

    fixed_meta = extract_metadata(os.path.join(fixed_dir, os.path.basename(fixed_files[0])))
    moving_meta = extract_metadata(os.path.join(moving_dir, os.path.basename(moving_files[0])))

    # Compute the min value of the fixed image
    stats = sitk.StatisticsImageFilter()
    stats.Execute(fixed_image)
    min_val_fixed = stats.GetMinimum()

    # Compute the min value of the moving image
    stats = sitk.StatisticsImageFilter()
    stats.Execute(fixed_image)
    min_val_moving = stats.GetMinimum()

    # Pad the fixed image by 20 slices on both cranial and caudal ends:
    pad_lower = (0, 0, 20)   # (pad_x_before, pad_y_before, pad_z_before)
    pad_upper = (0, 0, 20)   # (pad_x_after,  pad_y_after,  pad_z_after)
    padded_fixed = sitk.ConstantPad(
        fixed_image,
        pad_lower,
        pad_upper,
        constant=min_val_fixed
    )

    # Let user pick Z- and Y-shifts manually (in slices)
    moving_resized = sitk.Resample(
        moving_image,
        padded_fixed,
        sitk.Transform(),
        sitk.sitkLinear,
        min_val_moving,
        moving_image.GetPixelIDValue()
    )
    shift_z_slices, shift_y_slices, shift_x_slices = run_viewer(
        padded_fixed,
        moving_resized,
        fixed_modality=fixed_modality,
        moving_modality=moving_modality,
    )

    # Convert slice shift → mm
    spacing = fixed_image.GetSpacing()  # (x, y, z)
    shift_z_mm = shift_z_slices * spacing[2] * (-1)
    shift_y_mm = shift_y_slices * spacing[1] * (-1)
    shift_x_mm = shift_x_slices * spacing[0] * (-1)
    # print(f"Moving image spacing: {moving_image.GetSpacing()} mm")
    # print(f"Fixed image spacing: {fixed_image.GetSpacing()} mm")
    print(f"{get_datetime()} User‐defined Z-shift: {shift_z_slices} slices = {shift_z_mm:.2f} mm")
    print(f"{get_datetime()} User‐defined Y-shift: {shift_y_slices} slices = {shift_y_mm:.2f} mm")
    print(f"{get_datetime()} User‐defined X-shift: {shift_x_slices} slices = {shift_x_mm:.2f} mm")

    # Build the initial transform from the manual offsets
    initial_transform = sitk.VersorRigid3DTransform()
    initial_transform.SetTranslation((shift_x_mm, shift_y_mm, shift_z_mm))
    translation = initial_transform.GetTranslation()
    print(f"Initial translation: {translation}")
    # moving_reg = sitk.Resample(moving_image, fixed_image, initial_transform,
    #                            sitk.sitkLinear, min_val_moving, fixed_image.GetPixelIDValue())
    # run_viewer(fixed_image, moving_reg)

    # Rigid registration
    rigid_transform = perform_rigid_registration(fixed_image, moving_image, initial_transform)

    # Resample for visual check
    moving_reg = sitk.Resample(moving_image, fixed_image, rigid_transform,
                               sitk.sitkLinear, min_val_moving, fixed_image.GetPixelIDValue())

    # Show images after registration
    translation = rigid_transform.GetNthTransform(0).GetTranslation()
    print(f"Rigid translation: {translation}")
    run_viewer(
        fixed_image,
        moving_reg,
        fixed_modality=fixed_modality,
        moving_modality=moving_modality,
    )

    if confirm_fn is None:
        registration_accepted = input("Registration accepted? (y/n): ") == "y"
    else:
        registration_accepted = confirm_fn()

    if registration_accepted:
        print(f"{get_datetime()} Registration accepted")
        # In this implementation the transform is inverted inside transformation_matrix()
        # so we pass rigid_transform as is.
        create_registration_file(output_reg_file, rigid_transform, fixed_meta, moving_meta,
                                 fixed_files, moving_files)
        return rigid_transform, used_fixed_uid, used_moving_uid
    else:
        print(f"{get_datetime()} Registration rejected")
        return None, None, None
