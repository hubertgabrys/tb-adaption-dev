import datetime
import multiprocessing
import os
import sys
import tempfile
import time
import traceback
import tkinter as tk
from tkinter.scrolledtext import ScrolledText
from pathlib import Path

import pydicom
import SimpleITK as sitk

from usz_artemis_preprocessing.dicom.preprocessing import (
    list_dicom_series,
    process_single_dicom_file,
    format_dicom_time,
)
from usz_artemis_preprocessing.registration.core import (
    get_base_plan,
    perform_registration,
    run_viewer,
)
from tkinter import messagebox
from tkinter import ttk
from usz_artemis_preprocessing.imaging.resampling import (
    resample_ct,
    resample_mr_series_by_description,
)
from usz_artemis_preprocessing.io.export import send_files_to_aria
from usz_artemis_preprocessing.utils import (
    load_environment,
    check_if_ct_present,
    configure_sitk_threads,
    count_files,
    get_datetime,
    require_env,
)
from usz_artemis_preprocessing.dicom.segmentation import create_empty_rtstruct
from usz_artemis_preprocessing.dicom.copy_structures import (
    copy_structures,
    _rtstruct_references_series,
)
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import queue
import gc


class ConsoleRedirector:
    """Redirect writes to a Tkinter text widget from any thread."""

    def __init__(self, widget, max_queue: int = 1000):
        self.widget = widget
        self.queue: queue.Queue[str] = queue.Queue(maxsize=max_queue)
        self._dropped = 0
        self.widget.after(100, self._poll_queue)

    def write(self, text: str) -> None:
        """Thread-safe write that schedules GUI updates on the main thread."""

        try:
            self.queue.put_nowait(text)
        except queue.Full:
            self._dropped += 1

    def _append_text(self, text: str) -> None:
        self.widget.configure(state="normal")
        self.widget.insert("end", text)
        self.widget.see("end")
        self.widget.configure(state="disabled")

    def _poll_queue(self) -> None:
        try:
            while True:
                text = self.queue.get_nowait()
                self._append_text(text)
        except queue.Empty:
            pass

        if self._dropped:
            dropped = self._dropped
            self._dropped = 0
            self._append_text(
                f"\n… {dropped} console messages dropped (queue full) …\n"
            )

        self.widget.after(100, self._poll_queue)

    def flush(self) -> None:
        pass


def _is_placeholder_rtstruct(path: str) -> bool:
    """Return True if *path* points to a synthetic empty RTSTRUCT we created."""

    try:
        ds = pydicom.dcmread(path, stop_before_pixels=True, force=True)
    except Exception:
        return False

    try:
        roi_seq = getattr(ds, "StructureSetROISequence", None)
        if not roi_seq:
            return False
        first_roi = roi_seq[0]
        if getattr(first_roi, "ROIName", "") != "Dummy_PH":
            return False
        contour_seq = getattr(ds, "ROIContourSequence", None)
        if not contour_seq:
            return True
        for item in contour_seq:
            seq = getattr(item, "ContourSequence", None)
            if seq and len(seq):
                return False
        return True
    except Exception:
        return False


def rename_all_dicom_files(directory_path: str) -> None:
    """Ensure all DICOM files in *directory_path* have consistent names."""

    print(f"{get_datetime()} Renaming DICOM files…")
    with os.scandir(directory_path) as it:
        files = [
            entry.name for entry in it
            if entry.is_file() and entry.name.lower().endswith('.dcm')
        ]

    errors: list[Exception] = []
    with ThreadPoolExecutor(max_workers=os.cpu_count() or 1) as pool:
        futures = [
            pool.submit(process_single_dicom_file, directory_path, fname)
            for fname in files
        ]
        for fut in as_completed(futures):
            try:
                fut.result()
            except Exception as exc:  # pragma: no cover - surfaced to caller
                errors.append(exc)

    if errors:
        # Raise the first exception to fail the caller while still logging all
        # issues for debugging.
        for err in errors:
            print(f"{get_datetime()} Failed to rename DICOM: {err}")
        raise errors[0]


def wait_for_stable_imaging(directory: str, interval: float = 2.0,
                            stable_checks: int = 3) -> dict:
    """Wait until file count stabilizes before continuing downstream processing."""
    previous_total: int | None = None
    consecutive = 0
    result = {}
    while consecutive < stable_checks:
        # result = list_dicom_series(directory, imaging_only=True)
        # total = sum(len(info["files"]) for info in result.values())
        total = count_files(directory)
        if total == previous_total:
            consecutive += 1
        else:
            consecutive = 0
            previous_total = total
        if consecutive < stable_checks:
            time.sleep(interval)
    return result


def ct_already_resampled(directory: str) -> bool:
    """Return True if the directory already contains resampled CT DICOM slices."""

    for fname in os.listdir(directory):
        if fname.startswith("CT_Resampled") and fname.lower().endswith(".dcm"):
            return True
    return False


def _copy_structures_process(
    transform_path: str,
    current_directory: str,
    patient_id: str,
    rtplan_label: str,
    series_uid: str | None,
    base_series_uid: str | None,
    progress_q: multiprocessing.Queue,
) -> None:
    """Run copy_structures in a separate process to keep the UI responsive."""

    try:
        class _QueueWriter:
            def __init__(self, queue_obj):
                self.queue = queue_obj
                self._buffer = ""

            def write(self, text: str) -> None:
                if not text:
                    return
                self._buffer += text
                while "\n" in self._buffer:
                    line, self._buffer = self._buffer.split("\n", 1)
                    self.queue.put(("log", line + "\n"))

            def flush(self) -> None:
                if self._buffer:
                    self.queue.put(("log", self._buffer))
                    self._buffer = ""

        sys.stdout = _QueueWriter(progress_q)
        sys.stderr = _QueueWriter(progress_q)

        rigid_transform = sitk.ReadTransform(transform_path)

        def progress_cb(idx, total):
            progress_q.put((idx, total))

        copy_structures(
            current_directory,
            patient_id,
            rtplan_label,
            rigid_transform,
            series_uid=series_uid,
            base_series_uid=base_series_uid,
            progress_callback=progress_cb,
        )
    except Exception as exc:
        progress_q.put(("error", str(exc)))
    finally:
        progress_q.put(None)


def find_rtstructs_for_series(directory: str, series_uid: str) -> list[str]:
    """Return paths to RTSTRUCT files in *directory* referencing *series_uid*."""
    matches: list[str] = []
    for fname in os.listdir(directory):
        fpath = os.path.join(directory, fname)
        if not fname.lower().endswith(".dcm"):
            continue
        try:
            ds = pydicom.dcmread(fpath, stop_before_pixels=True)
        except Exception:
            continue
        if ds.Modality != "RTSTRUCT":
            continue
        if _rtstruct_references_series(ds, series_uid):
            matches.append(fpath)
    return matches


def remove_orphan_rt_files(directory: str, valid_series: set[str]) -> None:
    """Delete RTSTRUCT or REG files referencing series not present in *valid_series*."""
    print(f"{get_datetime()} Removing orphan RTSTRUCT/REG files...")
    for fname in os.listdir(directory):
        fpath = os.path.join(directory, fname)
        if not fname.lower().endswith(".dcm"):
            continue
        try:
            ds = pydicom.dcmread(fpath, stop_before_pixels=True)
        except Exception:
            continue
        modality = getattr(ds, "Modality", "")
        if modality not in ("RTSTRUCT", "REG"):
            continue

        referenced = set()
        if modality == "RTSTRUCT":
            try:
                for fr in ds.ReferencedFrameOfReferenceSequence:
                    for st in fr.RTReferencedStudySequence:
                        for se in st.RTReferencedSeriesSequence:
                            uid = getattr(se, "SeriesInstanceUID", None)
                            if uid:
                                referenced.add(uid)
            except Exception:
                pass
        else:  # REG
            try:
                if hasattr(ds, "ReferencedSeriesSequence"):
                    for item in ds.ReferencedSeriesSequence:
                        uid = getattr(item, "SeriesInstanceUID", None)
                        if uid:
                            referenced.add(uid)
                if hasattr(ds, "StudiesContainingOtherReferencedInstancesSequence"):
                    for study in ds.StudiesContainingOtherReferencedInstancesSequence:
                        if hasattr(study, "ReferencedSeriesSequence"):
                            for item in study.ReferencedSeriesSequence:
                                uid = getattr(item, "SeriesInstanceUID", None)
                                if uid:
                                    referenced.add(uid)
            except Exception:
                pass

        if referenced and not (referenced & valid_series):
            try:
                os.remove(fpath)
            except Exception:
                pass


def get_patient_name(directory_path: str) -> str:
    """Return the PatientName from the first DICOM file found in directory_path."""
    for root_dir, _, files in os.walk(directory_path):
        for fname in files:
            fpath = os.path.join(root_dir, fname)
            try:
                ds = pydicom.dcmread(fpath, stop_before_pixels=True, force=True)
                name = getattr(ds, "PatientName", None)
                if name:
                    return str(name)
            except Exception:
                pass
    return ""


def _extract_dicom_date(ds) -> str:
    """Return a YYYYMMDD date string from common DICOM date fields, or empty."""

    for attr in (
        "SeriesDate",
        "StudyDate",
        "AcquisitionDate",
        "ContentDate",
        "InstanceCreationDate",
    ):
        value = getattr(ds, attr, None)
        if not value:
            continue
        text = "".join(ch for ch in str(value) if ch.isdigit())
        if len(text) >= 8:
            return text[:8]
    return ""


def _has_old_dicom_files(directory: str, today_yyyymmdd: str) -> bool:
    """Return True if any DICOM file has a date older than today."""

    for root_dir, _, files in os.walk(directory):
        for fname in files:
            if not fname.lower().endswith(".dcm"):
                continue
            fpath = os.path.join(root_dir, fname)
            try:
                ds = pydicom.dcmread(fpath, stop_before_pixels=True, force=True)
            except Exception:
                continue
            dcm_date = _extract_dicom_date(ds)
            if dcm_date and dcm_date < today_yyyymmdd:
                return True
    return False


def main():
    """Launch the MRgTB preprocessing GUI and initialise automation state."""
    load_environment(".env")
    configure_sitk_threads()

    # If the user passed at least three args, use them…
    if len(sys.argv) >= 4:
        patient_id, rtplan_label, rtplan_uid = sys.argv[1:4]
    else:
        # …otherwise fall back to environment variables
        patient_id = os.environ.get('PATIENT_ID')
        rtplan_label = os.environ.get('RTPLAN_LABEL')
        rtplan_uid = os.environ.get('RTPLAN_UID')

    # Now verify that we actually have all three values
    if not (patient_id and rtplan_label and rtplan_uid):
        print("Missing parameters! Either pass "
              "<patient_id> <rtplan_label> <rtplan_uid> on the command line, "
              "or set PATIENT_ID, RTPLAN_LABEL and RTPLAN_UID in your env.")
        sys.exit(1)

    try:
        input_root = Path(require_env("INPUT_DIR"))
        baseplan_root = Path(require_env("BASEPLAN_DIR"))
    except EnvironmentError as exc:
        print(str(exc))
        sys.exit(1)

    input_dir = input_root / patient_id
    patient_name = get_patient_name(str(input_dir))

    root = tk.Tk()
    root.title("ARTEMIS Preprocessing")
    # root.geometry("1200x900")

    if rtplan_label and rtplan_label[-1].isalpha() and rtplan_label[-1].isupper():
        messagebox.showwarning(
            "Warning",
            "The script has been started from an adapted plan instead of the base plan.",
        )

    # Queue used to marshal callbacks from worker threads back to Tk safely.
    tk_call_queue: queue.Queue = queue.Queue()

    def run_on_tk_thread(func, *args, wait: bool = False, **kwargs):
        """Execute *func* on the Tk thread, optionally waiting for the result."""

        if threading.current_thread() is threading.main_thread():
            return func(*args, **kwargs)

        payload = {} if wait else None
        event = threading.Event() if wait else None
        tk_call_queue.put((func, args, kwargs, event, payload))
        if wait:
            event.wait()
            if payload and "error" in payload:
                raise payload["error"]
            return payload.get("result") if payload else None
        return None

    def process_tk_queue():
        """Drain any pending cross-thread Tk operations."""

        while True:
            try:
                func, args, kwargs, event, payload = tk_call_queue.get_nowait()
            except queue.Empty:
                break
            try:
                result = func(*args, **kwargs)
                if payload is not None:
                    payload["result"] = result
            except Exception as exc:  # pragma: no cover - surfaces in GUI console
                if payload is not None:
                    payload["error"] = exc
                else:
                    print(f"{get_datetime()} Error executing Tk callback: {exc}")
                    traceback.print_exc()
            finally:
                if event is not None:
                    event.set()
        root.after(20, process_tk_queue)

    # Kick off the polling loop so worker threads can post results immediately.
    process_tk_queue()

    # Configure grid to accommodate sent panel and console on the right
    root.grid_columnconfigure(2, weight=0)
    root.grid_columnconfigure(3, weight=1)

    right_panel = tk.Frame(root)
    right_panel.grid(row=0, column=3, rowspan=25, sticky="nsew", padx=(10, 10), pady=(0, 10))
    right_panel.grid_columnconfigure(0, weight=1)
    right_panel.grid_rowconfigure(0, weight=2)
    right_panel.grid_rowconfigure(2, weight=1)

    # Console output widget
    console_label = tk.Label(right_panel, text="Console:")
    console_label.grid(row=1, column=0, sticky="w", pady=(10, 0))
    console = ScrolledText(right_panel, state="disabled", width=140, height=18)
    console.grid(row=2, column=0, sticky="nsew")

    # Redirect stdout and stderr to the console widget
    sys.stdout = ConsoleRedirector(console)
    sys.stderr = ConsoleRedirector(console)

    # Title label
    lbl_title = tk.Label(root, text="ARTEMIS Preprocessing", font=("Helvetica", 16))
    lbl_title.grid(row=0, column=0, columnspan=2, sticky="w", padx=10, pady=10)

    # Patient information displayed under the title
    lbl_name = tk.Label(
        root,
        text=f"{patient_name}",
        font=("Helvetica", 12),
    )
    lbl_name.grid(row=1, column=0, columnspan=2, sticky="w", padx=10)

    lbl_patient = tk.Label(
        root,
        text=f"{patient_id}",
        font=("Helvetica", 12),
    )
    lbl_patient.grid(row=2, column=0, columnspan=2, sticky="w", padx=10)

    lbl_rtplan = tk.Label(
        root,
        text=f"{rtplan_label}",
        font=("Helvetica", 12),
    )
    lbl_rtplan.grid(row=3, column=0, columnspan=2, sticky="w", padx=10, pady=(0, 10))

    # Full automation checkbox above the base plan button
    full_automation_var = tk.BooleanVar(value=False)
    automation_state = {
        "active": False,
        "job": None,
        "registration_attempted": False,
        "registration_completed": False,
        "registration_successful": False,
        "registration_in_progress": False,
        "sending_in_progress": False,
        "known_imaging_uids": set(),
        "sent_series_uids": set(),
        "pending_send_uids": set(),
    }

    placeholder_rtstructs: dict[str, tuple[str, str]] = {}

    def automation_log(message: str) -> None:
        """Emit a timestamped log entry for automation-specific events."""

        print(f"{get_datetime()} [Automation] {message}")

    def schedule_automation_next(delay_ms: int = 5_000) -> None:
        """Schedule the next automation iteration with *delay_ms* milliseconds."""

        if not automation_state["active"]:
            return
        job = automation_state.get("job")
        if job is not None:
            root.after_cancel(job)
        automation_state["job"] = root.after(delay_ms, automation_loop)

    def automation_after_refresh(success: bool) -> None:
        """Continue the automation workflow once imaging refresh completes."""

        if not automation_state["active"]:
            return

        if not success:
            automation_log("Imaging refresh failed. Retrying in 5 seconds.")
            schedule_automation_next()
            return

        imaging_now = set(latest_imaging_uids)

        if not imaging_now:
            automation_log("No imaging series available yet. Retrying in 5 seconds.")
            schedule_automation_next()
            return

        new_imaging = imaging_now - automation_state["known_imaging_uids"]
        if new_imaging:
            automation_log(f"Detected {len(new_imaging)} new imaging series.")
        automation_state["known_imaging_uids"].update(imaging_now)

        if not automation_state["registration_attempted"]:
            automation_log("Imaging available. Starting registration.")
            automation_state["registration_attempted"] = True
            automation_state["registration_completed"] = False
            automation_state["registration_successful"] = False
            automation_state["registration_in_progress"] = True
            def automation_confirm(cost_value: float, quality_line: str) -> bool:
                """Ask the user to accept the automated registration result."""

                def ask_user():
                    details = [f"Cost: {cost_value:.4f}"]
                    if quality_line:
                        details.append(quality_line)
                    details.append("Rejecting will stop full automation.")
                    msg = "Accept registration result?\n" + "\n".join(details)
                    return messagebox.askyesno("Registration", msg)

                return run_on_tk_thread(ask_user, wait=True)
            on_register(
                confirm_override=automation_confirm,
                triggered_by_automation=True,
            )
            schedule_automation_next()
            return

        if not automation_state["registration_completed"]:
            schedule_automation_next()
            return

        unsent_imaging = imaging_now - automation_state["sent_series_uids"]

        reg_uids = {
            uid
            for uid, info in series_info.items()
            if info.get("modality") == "REG"
        }

        if automation_state["registration_successful"]:
            unsent_regs = reg_uids - automation_state["sent_series_uids"]
        else:
            unsent_regs = set()

        if unsent_imaging or unsent_regs:
            selected_uids = set(unsent_imaging) | set(unsent_regs)
            automation_log(
                "Sending series to Aria: "
                + ", ".join(sorted(selected_uids))
                if selected_uids
                else "Sending series to Aria."
            )
            on_send_to_aria(
                selected_uids=selected_uids,
                triggered_by_automation=True,
            )
            schedule_automation_next()
            return

        schedule_automation_next()

    def automation_loop() -> None:
        """Poll for imaging updates, run registration, and trigger sends."""

        if not automation_state["active"]:
            return

        automation_state["job"] = None

        if (
            automation_state["registration_in_progress"]
            or automation_state["sending_in_progress"]
            or imaging_refresh_in_progress
        ):
            schedule_automation_next()
            return

        on_get_images(completion_callback=automation_after_refresh)

    def start_full_automation() -> None:
        """Reset state and kick off the automated workflow."""

        if automation_state["active"]:
            return
        automation_state["active"] = True
        automation_state["registration_attempted"] = False
        automation_state["registration_completed"] = False
        automation_state["registration_successful"] = False
        automation_state["registration_in_progress"] = False
        automation_state["sending_in_progress"] = False
        automation_state["pending_send_uids"] = set()
        automation_state["known_imaging_uids"] = set()
        automation_state["sent_series_uids"] = set()
        automation_log("Starting full automation workflow.")
        on_get_base_plan()
        automation_loop()

    def stop_full_automation(*, from_internal: bool = False) -> None:
        """Cancel pending automation callbacks and clear state flags."""

        if not automation_state["active"]:
            if from_internal and full_automation_var.get():
                full_automation_var.set(False)
            return
        automation_state["active"] = False
        job = automation_state.get("job")
        if job is not None:
            root.after_cancel(job)
        automation_state["job"] = None
        automation_state["registration_in_progress"] = False
        automation_state["sending_in_progress"] = False
        automation_state["pending_send_uids"] = set()
        automation_log("Full automation stopped.")
        if from_internal and full_automation_var.get():
            full_automation_var.set(False)

    def toggle_full_automation() -> None:
        """Enable or disable automation in response to the checkbox state."""

        if full_automation_var.get():
            start_full_automation()
        else:
            stop_full_automation()

    chk_full_automation = tk.Checkbutton(
        root,
        text="Full automation",
        variable=full_automation_var,
        command=toggle_full_automation,
    )
    chk_full_automation.grid(row=4, column=0, sticky="w", padx=10, pady=(0, 5))

    auto_approve_var = tk.BooleanVar(value=False)
    chk_auto_approve = tk.Checkbutton(
        root,
        text="Automatic registration approval",
        variable=auto_approve_var,
    )
    chk_auto_approve.grid(row=5, column=0, columnspan=2, sticky="w", padx=10, pady=(0, 5))

    # Get Base Plan button with status label
    baseplan_status = tk.Label(root, text="", font=("Helvetica", 14))

    def on_get_base_plan():
        """Download the base plan and refresh the base series controls."""

        print(f"{get_datetime()} Getting the base plan...")
        start_time = time.time()
        baseplan_status.config(text="\u23F3", fg="orange")  # hourglass
        root.update_idletasks()
        try:
            get_base_plan(patient_id, rtplan_label, rtplan_uid)
            # List series in the base plan directory
            base_dir = baseplan_root / patient_id / rtplan_label
            if base_dir.exists():
                base_series_info.clear()
                base_series_info.update(list_dicom_series(str(base_dir)))
                update_bp_selection()
            baseplan_status.config(text="\u2705", fg="green")
            end_time = time.time()
            print(f"{get_datetime()} Getting the base plan took {end_time - start_time:.2f} seconds")
            print(f"{get_datetime()} DONE\n")
        except Exception as exc:
            baseplan_status.config(text="\u274C", fg="red")
            print(f"{get_datetime()} Failed to get base plan: {exc}")

    btn_baseplan = tk.Button(root, text="Get base plan", command=on_get_base_plan)
    btn_baseplan.grid(row=6, column=0, sticky="w", padx=10)
    baseplan_status.grid(row=6, column=1, sticky="w")

    # Store base plan series information
    base_series_info = {}
    bp_default_uid = None
    bp_default_modality = None

    # Get Imaging button with status label
    images_status = tk.Label(root, text="", font=("Helvetica", 14))
    series_info = {}
    series_vars = {}
    checkbox_texts = {}
    references_map = {}
    sent_info = {}
    latest_imaging_uids: set[str] = set()

    imaging_refresh_in_progress = False

    def on_get_images(completion_callback=None):
        """Refresh imaging list, creating empty RTSTRUCTs for orphan studies."""

        print(f"{get_datetime()} Getting images from {input_dir}...")
        start_time = time.time()
        nonlocal series_info, series_vars, checkbox_texts, references_map
        nonlocal latest_imaging_uids, imaging_refresh_in_progress, placeholder_rtstructs
        if imaging_refresh_in_progress:
            print(f"{get_datetime()} Imaging refresh already in progress; skipping new request.")
            if completion_callback:
                completion_callback(False)
            return

        images_status.config(text="\u23F3", fg="orange")  # hourglass
        root.update_idletasks()
        latest_imaging_uids = set()
        imaging_refresh_in_progress = True

        def handle_success(result):
            nonlocal series_info, series_vars, checkbox_texts, references_map
            nonlocal latest_imaging_uids, imaging_refresh_in_progress
            local_series, references, imaging_uids, registration_uids = result
            series_info = local_series
            references_map = references
            latest_imaging_uids = set(imaging_uids)
            rtstruct_uids = [
                uid
                for uid, info in local_series.items()
                if info.get("modality") == "RTSTRUCT"
            ]
            display_uids = imaging_uids + registration_uids + rtstruct_uids

            for widget in series_frame.winfo_children():
                widget.destroy()
            series_vars.clear()
            checkbox_texts.clear()

            tk.Label(series_frame, text="Series available:").pack(anchor="w")
            for uid in display_uids:
                info = series_info[uid]
                text = (
                    f"{info['date']} {info['time']} – {info['modality']} - {info['description']}"
                    f" (files={len(info['files'])})"
                )
                var = tk.BooleanVar(value=False)
                chk = tk.Checkbutton(series_frame, text=text, variable=var)
                chk.pack(anchor='w')
                series_vars[uid] = var
                checkbox_texts[uid] = text

            update_dropdown()
            images_status.config(text="\u2705", fg="green")
            end_time = time.time()
            print(f"{get_datetime()} Getting the images {end_time - start_time:.2f} seconds")
            print(f"{get_datetime()} DONE\n")
            imaging_refresh_in_progress = False
            if completion_callback:
                completion_callback(True)

        def handle_failure(err):
            nonlocal series_info, series_vars, checkbox_texts
            nonlocal references_map, latest_imaging_uids, imaging_refresh_in_progress
            images_status.config(text="\u274C", fg="red")
            print(f"{get_datetime()} Failed to get images: {err}")
            series_info = {}
            references_map = {}
            latest_imaging_uids = set()
            for widget in series_frame.winfo_children():
                widget.destroy()
            series_vars.clear()
            checkbox_texts.clear()
            update_dropdown()
            imaging_refresh_in_progress = False
            if completion_callback:
                completion_callback(False)

        def worker():
            try:
                if not input_dir.exists():
                    raise FileNotFoundError(
                        f"Input directory '{input_dir}' does not exist"
                    )
                wait_for_stable_imaging(str(input_dir))
                today_yyyymmdd = datetime.date.today().strftime("%Y%m%d")
                if _has_old_dicom_files(str(input_dir), today_yyyymmdd):
                    def warn_and_stop():
                        messagebox.showwarning(
                            "Warning",
                            "DICOM files older than today were detected and they should be deleted before proceeding.",
                        )
                        if automation_state["active"]:
                            stop_full_automation(from_internal=True)
                    run_on_tk_thread(warn_and_stop, wait=True)
                rename_all_dicom_files(str(input_dir))
                if check_if_ct_present(str(input_dir)) and not ct_already_resampled(str(input_dir)):
                    print(f"{get_datetime()} Resampling sCT...")
                    resample_ct(str(input_dir))
                print(f"{get_datetime()} Resampling MR sCT_sp_Pel_T2...")
                resample_mr_series_by_description(
                    str(input_dir),
                    series_description="sCT_sp_Pel_T2",
                    target_slice_thickness=3.0,
                )

                local_series = list_dicom_series(str(input_dir))
                if not local_series:
                    raise RuntimeError(
                        f"No imaging series found in '{input_dir}'."
                    )

                # Drop stale placeholder bookkeeping when the files vanish.
                for ref, (_, path) in list(placeholder_rtstructs.items()):
                    if not os.path.exists(path):
                        placeholder_rtstructs.pop(ref, None)

                real_rtstruct_refs: dict[str, str] = {}
                for uid, info in list(local_series.items()):
                    if info.get("modality") != "RTSTRUCT":
                        continue
                    refs = info.get("references", []) or []
                    files = info.get("files", [])
                    entry_is_placeholder = True
                    for fpath in files:
                        if _is_placeholder_rtstruct(fpath):
                            for ref in refs:
                                placeholder_rtstructs[ref] = (uid, fpath)
                        else:
                            entry_is_placeholder = False
                    if not entry_is_placeholder:
                        for ref in refs:
                            real_rtstruct_refs[ref] = uid

                for ref, real_uid in list(real_rtstruct_refs.items()):
                    placeholder_entry = placeholder_rtstructs.pop(ref, None)
                    if not placeholder_entry:
                        continue
                    placeholder_uid, placeholder_path = placeholder_entry
                    if placeholder_uid != real_uid:
                        if placeholder_path and os.path.exists(placeholder_path):
                            real_files = set(
                                local_series.get(real_uid, {}).get("files", [])
                            )
                            if placeholder_path not in real_files:
                                try:
                                    os.remove(placeholder_path)
                                except Exception:
                                    pass
                        local_series.pop(placeholder_uid, None)

                imaging_uids = [
                    uid
                    for uid, info in local_series.items()
                    if info.get("modality") not in ("RTSTRUCT", "REG")
                ]

                references: dict[str, list[str]] = {}
                for uid, info in local_series.items():
                    if info.get("modality") == "RTSTRUCT":
                        for ref in info.get("references", []) or []:
                            references.setdefault(ref, []).append(uid)

                for uid in imaging_uids:
                    if uid not in references:
                        create_empty_rtstruct(str(input_dir), uid, local_series[uid]["files"])
                        rs_path = os.path.join(str(input_dir), f"RS_{uid}.dcm")
                        try:
                            ds = pydicom.dcmread(rs_path, stop_before_pixels=True, force=True)
                            new_uid = getattr(ds, "SeriesInstanceUID", None)
                            date = getattr(ds, "SeriesDate", getattr(ds, "StudyDate", ""))
                            time_str = getattr(ds, "SeriesTime", getattr(ds, "StudyTime", ""))
                            if not date or not time_str:
                                date = date or getattr(ds, "StructureSetDate", "")
                                time_str = time_str or getattr(ds, "StructureSetTime", "")
                            time_str = format_dicom_time(time_str)
                            desc = getattr(ds, "SeriesDescription", "").strip() or "<no description>"
                            local_series[new_uid] = {
                                "date": date,
                                "time": time_str,
                                "modality": "RTSTRUCT",
                                "description": desc,
                                "files": [rs_path],
                                "references": [uid],
                            }
                            references.setdefault(uid, []).append(new_uid)
                            placeholder_rtstructs[uid] = (new_uid, rs_path)
                        except Exception:
                            pass

                valid_series = set(imaging_uids)
                to_remove = []
                for uid, info in local_series.items():
                    if info.get("modality") in ("RTSTRUCT", "REG"):
                        refs = set(info.get("references", []))
                        if refs and not (refs & valid_series):
                            for fpath in info.get("files", []):
                                try:
                                    os.remove(fpath)
                                except Exception:
                                    pass
                            to_remove.append(uid)

                for uid in to_remove:
                    local_series.pop(uid, None)
                for ref, (rs_uid, _) in list(placeholder_rtstructs.items()):
                    if rs_uid in to_remove:
                        placeholder_rtstructs.pop(ref, None)

                registration_uids = [
                    uid
                    for uid, info in local_series.items()
                    if info.get("modality") == "REG"
                ]

                result = (local_series, references, imaging_uids, registration_uids)
                run_on_tk_thread(handle_success, result)
            except Exception as err:
                run_on_tk_thread(handle_failure, err)

        threading.Thread(target=worker, daemon=True).start()

    btn_images = tk.Button(root, text="Get imaging", command=on_get_images)
    btn_images.grid(row=7, column=0, sticky="w", padx=10, pady=(0, 5))
    images_status.grid(row=7, column=1, sticky="w")

    # Imaging series frame (initially empty)
    series_frame = tk.Frame(root)
    series_frame.grid(row=8, column=0, columnspan=2, sticky="w", padx=10, pady=10)


    # Delete selected series button
    cleanup_status = tk.Label(root, text="", font=("Helvetica", 14))

    def on_cleanup():
        cleanup_status.config(text="\u23F3", fg="orange")  # hourglass
        root.update_idletasks()

        uids_to_delete = [uid for uid, var in series_vars.items() if var.get()]
        if not uids_to_delete:
            cleanup_status.config(text="", fg="orange")
            return

        def worker():
            success = True
            removed_rtstruct_uids: set[str] = set()
            try:
                for uid in uids_to_delete:
                    info = series_info.get(uid, {})
                    for fpath in info.get("files", []):
                        try:
                            os.remove(fpath)
                        except Exception:
                            success = False
                    for rs_uid in references_map.get(uid, []):
                        rs_info = series_info.get(rs_uid, {})
                        for fpath in rs_info.get("files", []):
                            try:
                                os.remove(fpath)
                            except Exception:
                                success = False
                        removed_rtstruct_uids.add(rs_uid)
                for imaging_uid in uids_to_delete:
                    placeholder_rtstructs.pop(imaging_uid, None)
                for ref, (rs_uid, _) in list(placeholder_rtstructs.items()):
                    if rs_uid in removed_rtstruct_uids:
                        placeholder_rtstructs.pop(ref, None)
            except Exception:
                success = False

            def finalize():
                cleanup_status.config(
                    text="\u2705" if success else "\u274C",
                    fg="green" if success else "red",
                )
                on_get_images()

            run_on_tk_thread(finalize)

        threading.Thread(target=worker, daemon=True).start()

    btn_cleanup = tk.Button(
        root,
        text="Delete selected series",
        command=on_cleanup,
        bg="#ffbbbb",
        activebackground="#ff9999",
    )
    btn_cleanup.grid(row=21, column=0, sticky="w", padx=10, pady=(50, 10))
    cleanup_status.grid(row=21, column=1, sticky="w", pady=(50, 10))

    # Dropdown menu for registration series
    selected_label_var = tk.StringVar()
    selected_uid_var = tk.StringVar()

    tk.Label(root, text="Select Daily Series for Registration").grid(row=12, column=0, columnspan=2, sticky="w", padx=10)
    dropdown = tk.OptionMenu(root, selected_label_var, '')
    dropdown.grid(row=13, column=0, columnspan=2, sticky="w", padx=10, pady=(0, 10))

    # Register button
    register_status = tk.Label(root, text="", font=("Helvetica", 14))
    register_progress = ttk.Progressbar(root, length=200, mode="determinate")

    last_rigid_transform = None
    last_fixed_uid = None
    last_moving_uid = None

    def on_register(confirm_override=None, triggered_by_automation: bool = False):
        """Execute the registration workflow and copy structures if successful."""

        nonlocal last_rigid_transform, last_fixed_uid, last_moving_uid
        register_status.config(text="\u23F3", fg="orange")
        root.update_idletasks()
        automation_triggered = triggered_by_automation and automation_state["active"]
        registration_was_successful = False

        if automation_triggered:
            automation_state["registration_in_progress"] = True

        selected_uid = selected_uid_var.get() or None
        selected_info = series_info.get(selected_uid, {})
        selected_modality = selected_info.get("modality")

        bp_uid = bp_default_uid
        bp_modality = bp_default_modality

        def confirm_threadsafe(cost_value, quality_line):
            """Invoke the confirmation dialog on the Tk thread."""

            if confirm_override is not None:
                return confirm_override(cost_value, quality_line)

            def ask_user():
                details = [f"Cost: {cost_value:.4f}"]
                if quality_line:
                    details.append(quality_line)
                msg = "Accept registration result?\n" + "\n".join(details)
                return messagebox.askyesno("Registration", msg)

            return run_on_tk_thread(ask_user, wait=True)

        def view_registration(*args, **kwargs):
            """Run the matplotlib viewer on the Tk thread."""

            block = kwargs.pop("block", True)

            def launch():
                return run_viewer(*args, **kwargs, block=block)

            return run_on_tk_thread(launch, wait=block)

        def finalize_failure(err=None, rejected=False):
            """Update UI and automation flags when registration fails."""

            register_status.config(text="\u274C", fg="red")
            if rejected and not automation_triggered:
                messagebox.showinfo("Registration", "Registration was rejected.")
            elif err and not automation_triggered:
                messagebox.showerror("Registration", f"Registration failed: {err}")
            if automation_triggered:
                if err:
                    automation_log(f"Registration failed: {err}")
                elif rejected:
                    automation_log("Registration rejected. Stopping automation.")
                    stop_full_automation(from_internal=True)
                automation_state["registration_in_progress"] = False
                automation_state["registration_completed"] = True
                automation_state["registration_successful"] = False

        def handle_registration_result(result):
            """Process the registration outcome on the Tk thread."""

            nonlocal last_rigid_transform, last_fixed_uid, last_moving_uid, registration_was_successful
            rigid_transform, _, used_fixed_uid, used_moving_uid, auto_approved = result
            if not rigid_transform:
                finalize_failure(rejected=True)
                return

            registration_was_successful = True
            last_rigid_transform = rigid_transform
            last_fixed_uid = used_fixed_uid
            last_moving_uid = used_moving_uid
            register_status.config(text="\u2705", fg="green")

            copy_status.config(text="\u23F3", fg="orange")
            root.update_idletasks()
            register_progress["value"] = 0
            register_progress.grid()
            progress_q = queue.Queue()
            result_state = {"success": False, "error": None}

            def progress_cb(idx, total):
                progress_q.put((idx, total))

            gc_enabled = gc.isenabled()
            if gc_enabled:
                gc.disable()

            copy_process = None
            transform_path = None

            def copy_worker():
                try:
                    print(f"{get_datetime()} Copying the structures...")
                    copy_structures(
                        str(input_dir),
                        patient_id,
                        rtplan_label,
                        rigid_transform,
                        series_uid=used_fixed_uid,
                        base_series_uid=used_moving_uid,
                        progress_callback=progress_cb,
                    )
                    result_state["success"] = True
                except Exception as exc:
                    result_state["error"] = exc
                finally:
                    progress_q.put(None)

            if auto_approved:
                progress_q = multiprocessing.Queue()
                fd, transform_path = tempfile.mkstemp(suffix=".tfm")
                os.close(fd)
                sitk.WriteTransform(rigid_transform, transform_path)
                copy_process = multiprocessing.Process(
                    target=_copy_structures_process,
                    args=(
                        transform_path,
                        str(input_dir),
                        patient_id,
                        rtplan_label,
                        used_fixed_uid,
                        used_moving_uid,
                        progress_q,
                    ),
                    daemon=True,
                )
                copy_process.start()
            else:
                threading.Thread(target=copy_worker, daemon=True).start()

            def finish_copy():
                register_progress.grid_remove()
                if gc_enabled:
                    gc.enable()
                    gc.collect()
                if copy_process is not None:
                    copy_process.join(timeout=1)
                if transform_path and os.path.exists(transform_path):
                    try:
                        os.remove(transform_path)
                    except Exception:
                        pass
                if result_state.get("error") is None and not result_state.get("success"):
                    result_state["success"] = True
                if result_state.get("success"):
                    copy_status.config(text="\u2705", fg="green")
                else:
                    copy_status.config(text="\u274C", fg="red")
                    err = result_state.get("error")
                    if err:
                        if automation_triggered:
                            automation_log(f"Copy structures failed: {err}")
                        else:
                            messagebox.showerror(
                                "Copy structures",
                                f"Failed to copy structures: {err}",
                            )
                    else:
                        if automation_triggered:
                            automation_log("Some structures failed to copy.")
                        else:
                            messagebox.showerror(
                                "Copy structures",
                                "Some structures failed to copy.",
                            )
                if automation_triggered:
                    automation_state["registration_in_progress"] = False
                    automation_state["registration_completed"] = True
                    automation_state["registration_successful"] = registration_was_successful
                on_get_images()

            def poll_queue():
                try:
                    while True:
                        item = progress_q.get_nowait()
                        if item is None:
                            finish_copy()
                            return
                        if isinstance(item, tuple) and item:
                            tag = item[0]
                            if tag == "error":
                                result_state["error"] = item[1]
                                result_state["success"] = False
                                continue
                            if tag == "log":
                                print(item[1], end="")
                                continue
                        idx, total = item
                        register_progress["maximum"] = total
                        register_progress["value"] = idx
                except queue.Empty:
                    pass
                root.after(100, poll_queue)

            poll_queue()

        def worker():
            try:
                result = perform_registration(
                    str(input_dir),
                    patient_id,
                    rtplan_label,
                    selected_series_uid=selected_uid,
                    selected_modality=selected_modality,
                    moving_series_uid=bp_uid,
                    moving_modality=bp_modality,
                    confirm_fn=confirm_threadsafe,
                    viewer_fn=view_registration,
                    auto_approve=auto_approve_var.get(),
                )
            except Exception as exc:
                run_on_tk_thread(finalize_failure, err=exc)
                return

            run_on_tk_thread(handle_registration_result, result)

        threading.Thread(target=worker, daemon=True).start()


    btn_register = tk.Button(root, text="Register", command=lambda: on_register())
    btn_register.grid(row=14, column=0, sticky="w", padx=10, pady=(0, 10))
    register_status.grid(row=14, column=1, sticky="w")

    copy_status = tk.Label(root, text="", font=("Helvetica", 14))
    copy_status.grid(row=15, column=1, sticky="w")

    register_progress.grid(row=16, column=0, columnspan=2, sticky="w", padx=10, pady=(0, 10))
    register_progress.grid_remove()

    send_status = tk.Label(root, text="", font=("Helvetica", 14))
    send_progress = ttk.Progressbar(root, length=200, mode="determinate")
    sent_panel = tk.Frame(right_panel)
    sent_label = tk.Label(sent_panel, text="Sent to Aria:")
    sent_tree = ttk.Treeview(
        sent_panel,
        columns=("series", "modality", "files", "sent_at"),
        show="headings",
        height=10,
    )
    sent_tree.heading("series", text="Series")
    sent_tree.heading("modality", text="Modality")
    sent_tree.heading("files", text="Files")
    sent_tree.heading("sent_at", text="Sent At")
    sent_tree.column("series", width=260, anchor="w")
    sent_tree.column("modality", width=80, anchor="w")
    sent_tree.column("files", width=80, anchor="e")
    sent_tree.column("sent_at", width=140, anchor="w")

    def _sent_file_count(uid: str) -> int:
        info = series_info.get(uid, {})
        count = len(info.get("files", []))
        if info.get("modality") in ("RTSTRUCT", "REG"):
            return count
        for ref_uid in references_map.get(uid, []):
            ref_info = series_info.get(ref_uid, {})
            count += len(ref_info.get("files", []))
        return count

    def _update_sent_list(uids: set[str]) -> None:
        sent_time = get_datetime()
        expanded_uids = set(uids)
        for uid in list(uids):
            expanded_uids.update(references_map.get(uid, []))
        for uid in sorted(expanded_uids):
            info = series_info.get(uid, {})
            series_label = checkbox_texts.get(uid, info.get("description", uid))
            modality = info.get("modality", "")
            file_count = _sent_file_count(uid)
            sent_info[uid] = {
                "label": series_label,
                "modality": modality,
                "files": file_count,
                "sent_at": sent_time,
            }
            values = (series_label, modality, file_count, sent_time)
            if sent_tree.exists(uid):
                sent_tree.item(uid, values=values)
            else:
                sent_tree.insert("", "end", iid=uid, values=values)

    def on_send_to_aria(selected_uids=None, triggered_by_automation: bool = False):
        """Send selected series to Aria and track automation-related selections."""

        print(f"{get_datetime()} Sending to Aria {input_dir}...")
        start_time = time.time()
        target_set: set[str] = set()
        if selected_uids is not None:
            target_set = set(selected_uids)
            for uid, var in series_vars.items():
                var.set(uid in target_set)
        else:
            for uid, var in series_vars.items():
                if var.get():
                    target_set.add(uid)

        if not target_set:
            if triggered_by_automation:
                automation_log("No series selected for Aria; skipping send.")
            else:
                messagebox.showinfo("Send to Aria", "No series selected.")
            return

        selected_files = []
        for uid, var in series_vars.items():
            if var.get():
                info = series_info.get(uid, {})
                selected_files.extend(info.get("files", []))
                for rs_uid in references_map.get(uid, []):
                    rs_info = series_info.get(rs_uid, {})
                    selected_files.extend(rs_info.get("files", []))

        if not selected_files:
            if triggered_by_automation:
                automation_log("No files resolved for selected series; skipping send.")
            else:
                messagebox.showinfo("Send to Aria", "No files resolved for selected series.")
            return

        if triggered_by_automation and automation_state["active"]:
            automation_state["sending_in_progress"] = True
            automation_state["pending_send_uids"] = set(target_set)

        send_status.config(text="\u23F3", fg="orange")
        root.update_idletasks()
        send_progress["value"] = 0
        send_progress.grid()
        progress_q = queue.Queue()
        result = {"success": False, "error": None}

        def progress_cb(idx, total):
            progress_q.put((idx, total))

        gc_enabled = gc.isenabled()
        if gc_enabled:
            gc.disable()

        def worker():
            try:
                result["success"] = send_files_to_aria(
                    selected_files, progress_callback=progress_cb
                )
            except Exception as exc:
                result["error"] = exc
            finally:
                progress_q.put(None)  # sentinel to signal completion

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

        def poll_queue():
            try:
                while True:
                    item = progress_q.get_nowait()
                    if item is None:
                        finish()
                        return
                    idx, total = item
                    send_progress["maximum"] = total
                    send_progress["value"] = idx
            except queue.Empty:
                pass
            root.after(100, poll_queue)

        def finish():
            send_progress.grid_remove()
            if gc_enabled:
                gc.enable()
                gc.collect()
            end_time = time.time()
            if result.get("success"):
                send_status.config(text="\u2705", fg="green")
                _update_sent_list(target_set)
                if triggered_by_automation:
                    automation_log("Files sent to Aria successfully.")
                else:
                    messagebox.showinfo("Send to Aria", "Files sent successfully.")
            else:
                send_status.config(text="\u274C", fg="red")
                err = result.get("error")
                if err:
                    if triggered_by_automation:
                        automation_log(f"Failed to send files: {err}")
                    else:
                        messagebox.showerror("Send to Aria", f"Failed to send files: {err}")
                else:
                    if triggered_by_automation:
                        automation_log("Some files failed to send.")
                    else:
                        messagebox.showerror("Send to Aria", "Some files failed to send.")

            if triggered_by_automation:
                automation_state["sending_in_progress"] = False
                if result.get("success"):
                    pending = automation_state.get("pending_send_uids", set())
                    automation_state["sent_series_uids"].update(pending)
                    for uid in list(pending):
                        for rs_uid in references_map.get(uid, []):
                            automation_state["sent_series_uids"].add(rs_uid)
                automation_state["pending_send_uids"] = set()

            print(f"{get_datetime()} Sending finished in {end_time - start_time:.2f} seconds")
            print(f"{get_datetime()} DONE\n")
            on_get_images()

        poll_queue()

    btn_send = tk.Button(root, text="Send to Aria", command=on_send_to_aria)
    btn_send.grid(row=17, column=0, sticky="w", padx=10, pady=(0, 10))
    send_status.grid(row=17, column=1, sticky="w")
    send_progress.grid(row=18, column=0, columnspan=2, sticky="w", padx=10, pady=(0, 10))
    send_progress.grid_remove()
    sent_panel.grid(row=0, column=0, sticky="nsew")
    sent_label.pack(anchor="w")
    sent_tree.pack(fill="both", expand=True)

    def set_selected_series(uid, label):
        """Update the dropdown selection variables with *uid* and *label*."""

        selected_uid_var.set(uid or "")
        selected_label_var.set(label)

    def update_dropdown(*args):
        # show MR series in the dropdown for the fixed image
        menu = dropdown["menu"]
        menu.delete(0, 'end')
        filtered_uids = [
            uid
            for uid, info in series_info.items()
            if info.get('modality') in ('MR')
               # and (
               #         info.get('description', '').startswith('t2_tse_tra')
               #         or info.get('description', '').startswith('sCT_sp')
               #         or info.get('description', '').endswith('dixon_tra_Siemens_in')
               # )
        ]

        for uid in filtered_uids:
            info = series_info[uid]
            text = checkbox_texts.get(uid, info['description'])

            def callback(value=uid, label=text):
                set_selected_series(value, label)

            menu.add_command(label=text, command=callback)

        if filtered_uids:
            first_uid = filtered_uids[0]
            first_label = checkbox_texts.get(first_uid, series_info[first_uid]['description'])
            set_selected_series(first_uid, first_label)
        else:
            set_selected_series(None, '')

    def update_bp_selection():
        nonlocal bp_default_uid, bp_default_modality
        filtered_uids = [uid for uid, info in base_series_info.items() if info.get('modality') in ('CT', 'MR')]
        if filtered_uids:
            first_uid = filtered_uids[0]
            first_info = base_series_info[first_uid]
            bp_default_uid = first_uid
            bp_default_modality = first_info.get('modality')
        else:
            bp_default_uid = None
            bp_default_modality = None

    root.mainloop()

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
