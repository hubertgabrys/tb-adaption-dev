import os
import sys
import tkinter as tk
from tkinter.scrolledtext import ScrolledText
from pathlib import Path

import pydicom

from preprocessing import (
    list_dicom_series,
    zip_directory,
    process_single_dicom_file,
)
from register import get_base_plan, perform_registration
from tkinter import messagebox
from tkinter import ttk
from resampling import resample_ct
from utils import (
    load_environment,
    check_if_ct_present,
    find_series_uids,
    configure_sitk_threads,
)
from segmentation import create_empty_rtstruct
from copy_structures import copy_structures, _rtstruct_references_series


class ConsoleRedirector:
    """Redirect writes to a Tkinter text widget."""

    def __init__(self, widget):
        self.widget = widget

    def write(self, text):
        self.widget.configure(state="normal")
        self.widget.insert("end", text)
        self.widget.see("end")
        self.widget.configure(state="disabled")

    def flush(self):
        pass


def rename_all_dicom_files(directory_path: str) -> None:
    """Rename all DICOM files within directory_path using modality prefixes."""
    for root_dir, _, files in os.walk(directory_path):
        for fname in files:
            try:
                process_single_dicom_file(root_dir, fname)
            except Exception:
                pass


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


def confirm_overwrite(existing_paths: list[str], description: str) -> bool:
    """Ask whether RTSTRUCTs referencing a series should be overwritten."""
    if not existing_paths:
        return True
    count = len(existing_paths)
    return messagebox.askyesno(
        "Overwrite RTSTRUCT",
        (
            f"{count} RTSTRUCT file{'s' if count > 1 else ''} for series "
            f"'{description}' exist. Overwrite?"
        ),
    )


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
    load_environment(".env")
    configure_sitk_threads()
    try:
        patient_id = sys.argv[1]
        rtplan_label = sys.argv[2]
        rtplan_uid = sys.argv[3]
        # patient_id = os.environ.get('PATIENT_ID')
        # rtplan_label = os.environ.get('RTPLAN_LABEL')
        # rtplan_uid = os.environ.get('RTPLAN_UID')
    except IndexError:
        print("Usage: python run_gui.py <patient_id> <rtplan_label> <rtplan_uid>")
        sys.exit(1)

    input_dir = Path(os.environ.get("INPUT_DIR")) / patient_id
    patient_name = get_patient_name(str(input_dir))

    root = tk.Tk()
    root.title("MRgTB Preprocessing")

    # Configure grid to accommodate console on the right
    root.grid_columnconfigure(2, weight=1)

    # Console output widget
    console = ScrolledText(root, state="disabled", width=80)
    console.grid(row=0, column=2, rowspan=18, sticky="nsew", padx=(10, 10), pady=10)

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

    # Get Base Plan button with status label
    baseplan_status = tk.Label(root, text="", font=("Helvetica", 14))

    def on_get_base_plan():
        baseplan_status.config(text="\u23F3", fg="orange")  # hourglass
        root.update_idletasks()
        try:
            get_base_plan(patient_id, rtplan_label, rtplan_uid)
            # List series in the base plan directory
            base_dir = Path(os.environ.get("BASEPLAN_DIR")) / patient_id / rtplan_label
            if base_dir.exists():
                base_series_info.clear()
                base_series_info.update(list_dicom_series(str(base_dir)))
                update_bp_dropdown()
            baseplan_status.config(text="\u2705", fg="green")
        except Exception:
            baseplan_status.config(text="\u274C", fg="red")

    btn_baseplan = tk.Button(root, text="Get base plan", command=on_get_base_plan)
    btn_baseplan.grid(row=4, column=0, sticky="w", padx=10)
    baseplan_status.grid(row=4, column=1, sticky="w")

    # Store base plan series information
    base_series_info = {}
    bp_selected_var = tk.StringVar()
    bp_selection_map = {}

    # Get Imaging button with status label
    images_status = tk.Label(root, text="", font=("Helvetica", 14))
    series_info = {}
    series_vars = {}
    checkbox_texts = {}

    def on_get_images():
        nonlocal series_info, series_vars, checkbox_texts
        images_status.config(text="\u23F3", fg="orange")  # hourglass
        root.update_idletasks()
        try:
            rename_all_dicom_files(str(input_dir))
            # Load imaging series only when button is clicked
            series_info = list_dicom_series(str(input_dir))
            # Clear previous entries
            for widget in series_frame.winfo_children():
                widget.destroy()
            series_vars.clear()
            checkbox_texts.clear()

            tk.Label(series_frame, text="Imaging series available:").pack(anchor="w")
            # Populate checkboxes for each series
            for uid, info in series_info.items():
                text = (
                    f"{info['date']} {info['time']} – {info['modality']} - {info['description']}"
                    f" (files={len(info['files'])})"
                )
                var = tk.BooleanVar(value=False)
                chk = tk.Checkbutton(series_frame, text=text, variable=var)
                chk.pack(anchor='w')
                series_vars[uid] = var
                checkbox_texts[uid] = text

            # Update dropdown after loading
            update_dropdown()
            images_status.config(text="\u2705", fg="green")
        except Exception:
            images_status.config(text="\u274C", fg="red")

    btn_images = tk.Button(root, text="Get imaging", command=on_get_images)
    btn_images.grid(row=5, column=0, sticky="w", padx=10, pady=(0, 5))
    images_status.grid(row=5, column=1, sticky="w")

    # Backup button (create backup of input directory)
    backup_status = tk.Label(root, text="", font=("Helvetica", 14))

    def on_backup():
        backup_status.config(text="\u23F3", fg="orange")  # hourglass
        root.update_idletasks()
        try:
            zip_directory(str(input_dir), str(input_dir) + ".zip")
            backup_status.config(text="\u2705", fg="green")
        except Exception:
            backup_status.config(text="\u274C", fg="red")

    btn_backup = tk.Button(root, text="Create backup", command=on_backup)
    btn_backup.grid(row=6, column=0, sticky="w", padx=10)
    backup_status.grid(row=6, column=1, sticky="w")

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
            cleanup_status.config(text="\u2705", fg="green")
        except Exception:
            cleanup_status.config(text="\u274C", fg="red")
        # refresh displayed series after cleanup
        on_get_images()

    btn_cleanup = tk.Button(root, text="Delete selected series", command=on_cleanup)
    btn_cleanup.grid(row=10, column=0, sticky="w", padx=10, pady=(0, 10))
    cleanup_status.grid(row=10, column=1, sticky="w")

    # Resample button
    resample_status = tk.Label(root, text="", font=("Helvetica", 14))

    def on_resample():
        resample_status.config(text="\u23F3", fg="orange")  # hourglass
        root.update_idletasks()
        try:
            if check_if_ct_present(str(input_dir)):
                resample_ct(str(input_dir))
                resample_status.config(text="\u2705", fg="green")
                # refresh series list after new sCT is generated
                on_get_images()
            else:
                resample_status.config(text="\u274C", fg="red")
        except Exception:
            resample_status.config(text="\u274C", fg="red")

    btn_resample = tk.Button(root, text="Resample sCT", command=on_resample)
    btn_resample.grid(row=11, column=0, sticky="w", padx=10, pady=(0, 10))
    resample_status.grid(row=11, column=1, sticky="w")

    # Rename images and create RTSTRUCTs button
    rename_status = tk.Label(root, text="", font=("Helvetica", 14))

    def on_rename():
        rename_status.config(text="\u23F3", fg="orange")  # hourglass
        root.update_idletasks()
        try:
            dicom_series = list_dicom_series(str(input_dir), imaging_only=True)
            for uid, info in dicom_series.items():
                existing_paths = find_rtstructs_for_series(str(input_dir), uid)
                new_path = os.path.join(str(input_dir), f"RS_{uid}.dcm")
                if not confirm_overwrite(existing_paths, info["description"]):
                    continue
                for old in existing_paths:
                    if os.path.abspath(old) == os.path.abspath(new_path):
                        continue
                    try:
                        os.remove(old)
                    except Exception:
                        pass
                create_empty_rtstruct(str(input_dir), uid, info["files"])
            rename_status.config(text="\u2705", fg="green")
        except Exception:
            rename_status.config(text="\u274C", fg="red")
        on_get_images()

    btn_rename = tk.Button(root, text="Rename images", command=on_rename)
    btn_rename.grid(row=12, column=0, sticky="w", padx=10, pady=(0, 10))
    rename_status.grid(row=12, column=1, sticky="w")

    # Dropdown for base plan series
    tk.Label(root, text="Select Base Plan Series for Registration").grid(row=13, column=0, columnspan=2, sticky="w", padx=10)
    bp_dropdown = tk.OptionMenu(root, bp_selected_var, '')
    bp_dropdown.grid(row=14, column=0, columnspan=2, sticky="w", padx=10, pady=(0, 10))

    # Dropdown menu for registration series
    selected_var = tk.StringVar()
    selection_map = {}
    tk.Label(root, text="Select Daily Series for Registration").grid(row=15, column=0, columnspan=2, sticky="w", padx=10)
    dropdown = tk.OptionMenu(root, selected_var, '')
    dropdown.grid(row=16, column=0, columnspan=2, sticky="w", padx=10, pady=(0, 10))

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

    def on_register():
        nonlocal last_rigid_transform, last_fixed_uid, last_moving_uid
        register_status.config(text="\u23F3", fg="orange")
        root.update_idletasks()
        try:

            def confirm():
                return messagebox.askyesno("Registration", "Accept registration result?")

            # Determine which series the user selected in the dropdowns
            selected_label = selected_var.get()
            selected_uid = selection_map.get(selected_label)
            selected_info = series_info.get(selected_uid, {})
            selected_modality = selected_info.get("modality")

            bp_label = bp_selected_var.get()
            bp_uid = bp_selection_map.get(bp_label)
            bp_info = base_series_info.get(bp_uid, {})
            bp_modality = bp_info.get("modality")

            mode = ask_registration_mode()
            if not mode:
                register_status.config(text="", fg="orange")
                return

            try:
                rigid_transform, used_fixed_uid, used_moving_uid = perform_registration(
                    str(input_dir),
                    patient_id,
                    rtplan_label,
                    selected_series_uid=selected_uid,
                    selected_modality=selected_modality,
                    moving_series_uid=bp_uid,
                    moving_modality=bp_modality,
                    confirm_fn=confirm,
                    prealign=(mode == "semi"),
                )
            except Exception:
                rigid_transform, used_fixed_uid, used_moving_uid = None, None, None

            if rigid_transform:
                last_rigid_transform = rigid_transform
                last_fixed_uid = used_fixed_uid
                last_moving_uid = used_moving_uid
                register_status.config(text="\u2705", fg="green")
            else:
                register_status.config(text="\u274C", fg="red")
        except Exception:
            register_status.config(text="\u274C", fg="red")
        finally:
            # refresh displayed series after cleanup
            on_get_images()

    btn_register = tk.Button(root, text="Register", command=on_register)
    btn_register.grid(row=17, column=0, sticky="w", padx=10, pady=(0, 10))
    register_status.grid(row=17, column=1, sticky="w")

    copy_status = tk.Label(root, text="", font=("Helvetica", 14))

    def on_copy_structures():
        if last_rigid_transform is None:
            messagebox.showerror("Copy structures", "Please perform registration first.")
            copy_status.config(text="\u274C", fg="red")
            return

        copy_status.config(text="\u23F3", fg="orange")
        root.update_idletasks()
        register_progress["value"] = 0
        register_progress.grid()

        try:
            def progress_cb(idx, total):
                register_progress["maximum"] = total
                register_progress["value"] = idx
                root.update_idletasks()

            copy_structures(
                str(input_dir),
                patient_id,
                rtplan_label,
                last_rigid_transform,
                series_uid=last_fixed_uid,
                base_series_uid=last_moving_uid,
                progress_callback=progress_cb,
            )
            copy_status.config(text="\u2705", fg="green")
        except Exception:
            copy_status.config(text="\u274C", fg="red")
        finally:
            register_progress.grid_remove()
            on_get_images()

    btn_copy_structures = tk.Button(root, text="Copy structures", command=on_copy_structures)
    btn_copy_structures.grid(row=18, column=0, sticky="w", padx=10, pady=(0, 10))
    copy_status.grid(row=18, column=1, sticky="w")

    register_progress.grid(row=19, column=0, columnspan=2, sticky="w", padx=10, pady=(0, 10))
    register_progress.grid_remove()

    # Reset button to clear status indicators
    def on_reset():
        for lbl in (
            baseplan_status,
            images_status,
            backup_status,
            cleanup_status,
            resample_status,
            rename_status,
            register_status,
            copy_status,
        ):
            lbl.config(text="")
        register_progress.grid_remove()
        register_progress["value"] = 0

    btn_reset = tk.Button(root, text="Reset", command=on_reset)
    btn_reset.grid(row=20, column=0, sticky="w", padx=10, pady=(0, 10))

    def update_dropdown(*args):
        # show CT and MR series in the dropdown for the fixed image
        menu = dropdown["menu"]
        menu.delete(0, 'end')
        selection_map.clear()
        filtered_uids = [uid for uid, info in series_info.items() if info.get('modality') in ('CT', 'MR')]
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

    def update_bp_dropdown(*args):
        # populate base plan dropdown with CT and MR series only
        menu = bp_dropdown["menu"]
        menu.delete(0, 'end')
        bp_selection_map.clear()
        filtered_uids = [uid for uid, info in base_series_info.items() if info.get('modality') in ('CT', 'MR')]
        for uid in filtered_uids:
            info = base_series_info[uid]
            text = f"{info['date']} {info['time']} – {info['modality']} - {info['description']}"
            bp_selection_map[text] = uid
            menu.add_command(label=text, command=tk._setit(bp_selected_var, text))

        if filtered_uids:
            first_uid = filtered_uids[0]
            first_info = base_series_info[first_uid]
            bp_selected_var.set(f"{first_info['date']} {first_info['time']} – {first_info['modality']} - {first_info['description']}")
        else:
            bp_selected_var.set('')

    root.mainloop()

if __name__ == '__main__':
    main()

