import os
import sys
import time
import tkinter as tk
from tkinter.scrolledtext import ScrolledText
from pathlib import Path

import pydicom

from preprocessing import (
    list_dicom_series,
    process_single_dicom_file,
)
from register import get_base_plan, perform_registration
from tkinter import messagebox
from tkinter import ttk
from resampling import resample_ct
from export import send_files_to_aria
from utils import (
    load_environment,
    check_if_ct_present,
    configure_sitk_threads, count_files, get_datetime,
)
from segmentation import create_empty_rtstruct
from copy_structures import copy_structures, _rtstruct_references_series
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import queue
import gc


class ConsoleRedirector:
    """Redirect writes to a Tkinter text widget from any thread."""

    def __init__(self, widget):
        self.widget = widget
        self.queue: queue.Queue[str] = queue.Queue()
        self.widget.after(100, self._poll_queue)

    def write(self, text: str) -> None:
        """Thread-safe write that schedules GUI updates on the main thread."""
        self.queue.put(text)

    def _poll_queue(self) -> None:
        try:
            while True:
                text = self.queue.get_nowait()
                self.widget.configure(state="normal")
                self.widget.insert("end", text)
                self.widget.see("end")
                self.widget.configure(state="disabled")
        except queue.Empty:
            pass
        self.widget.after(100, self._poll_queue)

    def flush(self) -> None:
        pass


def rename_all_dicom_files(directory_path: str) -> None:
    """Ensure all DICOM files in *directory_path* have consistent names."""

    print(f"{get_datetime()} Renaming DICOM files…")
    with os.scandir(directory_path) as it:
        files = [
            entry.name for entry in it
            if entry.is_file() and entry.name.lower().endswith('.dcm')
        ]

    with ThreadPoolExecutor(max_workers=os.cpu_count() or 1) as pool:
        for _ in as_completed([
            pool.submit(process_single_dicom_file, directory_path, fname)
            for fname in files
        ]):
            pass  # you could catch exceptions here if needed


def wait_for_stable_imaging(directory: str, interval: float = 1.0,
                            stable_checks: int = 2) -> dict:
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

    input_dir = Path(os.environ.get("INPUT_DIR")) / patient_id
    patient_name = get_patient_name(str(input_dir))

    root = tk.Tk()
    root.title("MRgTB Preprocessing")

    # Configure grid to accommodate console on the right
    root.grid_columnconfigure(2, weight=1)

    # Console output widget
    console = ScrolledText(root, state="disabled", width=140)
    console.grid(row=0, column=2, rowspan=19, sticky="nsew", padx=(10, 10), pady=10)

    # Redirect stdout and stderr to the console widget
    sys.stdout = ConsoleRedirector(console)
    sys.stderr = ConsoleRedirector(console)

    # Title label
    lbl_title = tk.Label(root, text="MRgTB Preprocessing", font=("Helvetica", 16))
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

    def automation_log(message: str) -> None:
        """Emit a timestamped log entry for automation-specific events."""

        print(f"{get_datetime()} [Automation] {message}")

    def schedule_automation_next(delay_ms: int = 10_000) -> None:
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
            automation_log("Imaging refresh failed. Retrying in 10 seconds.")
            schedule_automation_next()
            return

        imaging_now = set(latest_imaging_uids)

        if not imaging_now:
            automation_log("No imaging series available yet. Retrying in 10 seconds.")
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
            on_register(
                mode_override="auto",
                confirm_override=lambda _cost, _quality: True,
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

    def stop_full_automation() -> None:
        """Cancel pending automation callbacks and clear state flags."""

        if not automation_state["active"]:
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
            base_dir = Path(os.environ.get("BASEPLAN_DIR")) / patient_id / rtplan_label
            if base_dir.exists():
                base_series_info.clear()
                base_series_info.update(list_dicom_series(str(base_dir)))
                update_bp_selection()
            baseplan_status.config(text="\u2705", fg="green")
            end_time = time.time()
            print(f"{get_datetime()} Getting the base plan took {end_time - start_time:.2f} seconds")
            print(f"{get_datetime()} DONE\n")
        except Exception:
            baseplan_status.config(text="\u274C", fg="red")

    btn_baseplan = tk.Button(root, text="Get base plan", command=on_get_base_plan)
    btn_baseplan.grid(row=5, column=0, sticky="w", padx=10)
    baseplan_status.grid(row=5, column=1, sticky="w")

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
    latest_imaging_uids: set[str] = set()

    imaging_refresh_in_progress = False

    def on_get_images(completion_callback=None):
        """Refresh imaging list, creating empty RTSTRUCTs for orphan studies."""

        print(f"{get_datetime()} Getting images from {input_dir}...")
        start_time = time.time()
        nonlocal series_info, series_vars, checkbox_texts, references_map
        nonlocal latest_imaging_uids, imaging_refresh_in_progress
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
            display_uids = imaging_uids + registration_uids

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
            nonlocal imaging_refresh_in_progress
            images_status.config(text="\u274C", fg="red")
            print(f"{get_datetime()} Failed to get images: {err}")
            imaging_refresh_in_progress = False
            if completion_callback:
                completion_callback(False)

        def worker():
            try:
                wait_for_stable_imaging(str(input_dir))
                rename_all_dicom_files(str(input_dir))
                if check_if_ct_present(str(input_dir)) and not ct_already_resampled(str(input_dir)):
                    print(f"{get_datetime()} Resampling sCT...")
                    resample_ct(str(input_dir))

                local_series = list_dicom_series(str(input_dir))

                imaging_uids = [
                    uid
                    for uid, info in local_series.items()
                    if info.get("modality") not in ("RTSTRUCT", "REG")
                ]

                references = {}
                for uid, info in local_series.items():
                    if info.get("modality") == "RTSTRUCT":
                        for ref in info.get("references", []):
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

                registration_uids = [
                    uid
                    for uid, info in local_series.items()
                    if info.get("modality") == "REG"
                ]

                result = (local_series, references, imaging_uids, registration_uids)
                root.after(0, lambda: handle_success(result))
            except Exception as err:
                root.after(0, lambda: handle_failure(err))

        threading.Thread(target=worker, daemon=True).start()

    btn_images = tk.Button(root, text="Get imaging", command=on_get_images)
    btn_images.grid(row=6, column=0, sticky="w", padx=10, pady=(0, 5))
    images_status.grid(row=6, column=1, sticky="w")

    # Imaging series frame (initially empty)
    series_frame = tk.Frame(root)
    series_frame.grid(row=7, column=0, columnspan=2, sticky="w", padx=10, pady=10)


    # Delete selected series button
    cleanup_status = tk.Label(root, text="", font=("Helvetica", 14))

    def on_cleanup():
        cleanup_status.config(text="\u23F3", fg="orange")  # hourglass
        root.update_idletasks()
        try:
            # delete the series with checkboxes ticked
            uids_to_delete = [uid for uid, var in series_vars.items() if var.get()]
            for uid in uids_to_delete:
                info = series_info.get(uid, {})
                for fpath in info.get("files", []):
                    try:
                        os.remove(fpath)
                    except Exception:
                        pass
                for rs_uid in references_map.get(uid, []):
                    rs_info = series_info.get(rs_uid, {})
                    for fpath in rs_info.get("files", []):
                        try:
                            os.remove(fpath)
                        except Exception:
                            pass
            cleanup_status.config(text="\u2705", fg="green")
        except Exception:
            cleanup_status.config(text="\u274C", fg="red")
        # refresh displayed series after cleanup
        on_get_images()

    btn_cleanup = tk.Button(
        root,
        text="Delete selected series",
        command=on_cleanup,
        bg="#ffbbbb",
        activebackground="#ff9999",
    )
    btn_cleanup.grid(row=18, column=0, sticky="w", padx=10, pady=(50, 10))
    cleanup_status.grid(row=18, column=1, sticky="w", pady=(50, 10))

    # Dropdown menu for registration series
    selected_var = tk.StringVar()
    selection_map = {}
    tk.Label(root, text="Select Daily Series for Registration").grid(row=11, column=0, columnspan=2, sticky="w", padx=10)
    dropdown = tk.OptionMenu(root, selected_var, '')
    dropdown.grid(row=12, column=0, columnspan=2, sticky="w", padx=10, pady=(0, 10))

    # Register button
    register_status = tk.Label(root, text="", font=("Helvetica", 14))
    register_progress = ttk.Progressbar(root, length=200, mode="determinate")

    def ask_registration_mode():
        result = {"mode": None}

        dialog = tk.Toplevel(root)
        dialog.title("Registration mode")
        tk.Label(dialog, text="Choose registration mode:").pack(padx=20, pady=(10, 5))

        def choose(mode):
            result["mode"] = mode
            dialog.destroy()

        btn_auto = tk.Button(dialog, text="Automatic", width=15,
                             command=lambda: choose("auto"))
        btn_semi = tk.Button(dialog, text="Semi-automatic", width=15,
                             command=lambda: choose("semi"))
        btn_auto.pack(padx=10, pady=5)
        btn_semi.pack(padx=10, pady=(0, 10))

        dialog.transient(root)
        # Center the dialog on the screen
        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() // 2) - (dialog.winfo_width() // 2)
        y = (dialog.winfo_screenheight() // 2) - (dialog.winfo_height() // 2)
        dialog.geometry(f"+{x}+{y}")
        dialog.grab_set()
        root.wait_window(dialog)
        return result["mode"]

    last_rigid_transform = None
    last_fixed_uid = None
    last_moving_uid = None

    def on_register(mode_override=None, confirm_override=None,
                    triggered_by_automation: bool = False):
        """Execute the registration workflow and copy structures if successful."""

        nonlocal last_rigid_transform, last_fixed_uid, last_moving_uid
        register_status.config(text="\u23F3", fg="orange")
        root.update_idletasks()
        automation_triggered = triggered_by_automation and automation_state["active"]
        registration_was_successful = False
        if automation_triggered:
            automation_state["registration_in_progress"] = True
        try:

            def confirm(cost_value, quality_line):
                """Ask the user to confirm the registration unless overridden."""

                if confirm_override is not None:
                    return confirm_override(cost_value, quality_line)
                details = [f"Cost: {cost_value:.4f}"]
                if quality_line:
                    details.append(quality_line)
                msg = "Accept registration result?\n" + "\n".join(details)
                return messagebox.askyesno("Registration", msg)

            # Determine which series the user selected in the dropdowns
            selected_label = selected_var.get()
            selected_uid = selection_map.get(selected_label)
            selected_info = series_info.get(selected_uid, {})
            selected_modality = selected_info.get("modality")

            bp_uid = bp_default_uid
            bp_modality = bp_default_modality

            mode = mode_override or ask_registration_mode()
            if not mode:
                register_status.config(text="", fg="orange")
                if automation_triggered:
                    automation_state["registration_in_progress"] = False
                    automation_state["registration_completed"] = True
                    automation_state["registration_successful"] = False
                return

            try:
                rigid_transform, _, used_fixed_uid, used_moving_uid = perform_registration(
                    str(input_dir),
                    patient_id,
                    rtplan_label,
                    selected_series_uid=selected_uid,
                    selected_modality=selected_modality,
                    moving_series_uid=bp_uid,
                    moving_modality=bp_modality,
                    confirm_fn=confirm,
                    manual_fine_tuning=(mode == "semi"),
                )
            except Exception:
                rigid_transform, used_fixed_uid, used_moving_uid = None, None, None

            if rigid_transform:
                last_rigid_transform = rigid_transform
                last_fixed_uid = used_fixed_uid
                last_moving_uid = used_moving_uid
                register_status.config(text="\u2705", fg="green")
                registration_was_successful = True

                copy_status.config(text="\u23F3", fg="orange")
                root.update_idletasks()
                register_progress["value"] = 0
                register_progress.grid()
                progress_q = queue.Queue()
                result = {"success": False, "error": None}

                def progress_cb(idx, total):
                    progress_q.put((idx, total))

                gc_enabled = gc.isenabled()
                if gc_enabled:
                    gc.disable()

                def worker():
                    """Copy structures in a worker thread to keep the UI responsive."""

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
                        result["success"] = True
                    except Exception as exc:
                        result["error"] = exc
                    finally:
                        progress_q.put(None)

                thread = threading.Thread(target=worker, daemon=True)
                thread.start()

                def poll_queue():
                    """Consume progress updates from the worker thread."""

                    try:
                        while True:
                            item = progress_q.get_nowait()
                            if item is None:
                                finish()
                                return
                            idx, total = item
                            register_progress["maximum"] = total
                            register_progress["value"] = idx
                    except queue.Empty:
                        pass
                    root.after(100, poll_queue)

                def finish():
                    """Tear down progress UI and update automation state flags."""

                    register_progress.grid_remove()
                    if gc_enabled:
                        gc.enable()
                        gc.collect()
                    if result.get("success"):
                        copy_status.config(text="\u2705", fg="green")
                    else:
                        copy_status.config(text="\u274C", fg="red")
                        err = result.get("error")
                        if err:
                            if automation_triggered:
                                automation_log(f"Copy structures failed: {err}")
                            else:
                                messagebox.showerror(
                                    "Copy structures",
                                    f"Failed to copy structures: {err}",
                                )
                    if automation_triggered:
                        automation_state["registration_in_progress"] = False
                        automation_state["registration_completed"] = True
                        automation_state["registration_successful"] = registration_was_successful
                    on_get_images()

                poll_queue()
            else:
                register_status.config(text="\u274C", fg="red")
                if automation_triggered:
                    automation_state["registration_in_progress"] = False
                    automation_state["registration_completed"] = True
                    automation_state["registration_successful"] = False
        except Exception:
            register_status.config(text="\u274C", fg="red")
            if automation_triggered:
                automation_state["registration_in_progress"] = False
                automation_state["registration_completed"] = True
                automation_state["registration_successful"] = False


    btn_register = tk.Button(root, text="Register", command=lambda: on_register())
    btn_register.grid(row=13, column=0, sticky="w", padx=10, pady=(0, 10))
    register_status.grid(row=13, column=1, sticky="w")

    copy_status = tk.Label(root, text="", font=("Helvetica", 14))
    copy_status.grid(row=14, column=1, sticky="w")

    register_progress.grid(row=15, column=0, columnspan=2, sticky="w", padx=10, pady=(0, 10))
    register_progress.grid_remove()

    send_status = tk.Label(root, text="", font=("Helvetica", 14))
    send_progress = ttk.Progressbar(root, length=200, mode="determinate")

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
    btn_send.grid(row=16, column=0, sticky="w", padx=10, pady=(0, 10))
    send_status.grid(row=16, column=1, sticky="w")
    send_progress.grid(row=17, column=0, columnspan=2, sticky="w", padx=10, pady=(0, 10))
    send_progress.grid_remove()

    def update_dropdown(*args):
        # show CT and MR series in the dropdown for the fixed image
        menu = dropdown["menu"]
        menu.delete(0, 'end')
        selection_map.clear()
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
            selection_map[text] = uid
            menu.add_command(label=text, command=tk._setit(selected_var, text))

        # Default to first label
        if filtered_uids:
            first_uid = filtered_uids[0]
            selected_var.set(checkbox_texts.get(first_uid, series_info[first_uid]['description']))
        else:
            selected_var.set('')

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
    main()
