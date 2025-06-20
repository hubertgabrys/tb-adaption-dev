import os
import sys
import tkinter as tk
from pathlib import Path

from preprocessing import (
    list_imaging_series,
    zip_directory,
    process_single_dicom_file,
)
from register import get_base_plan, perform_registration
from tkinter import messagebox
from tkinter import ttk
from resampling import resample_ct
from utils import load_environment, check_if_ct_present, find_series_uids
from segmentation import create_empty_rtstruct
from copy_structures import copy_structures


def rename_all_dicom_files(directory_path: str) -> None:
    """Rename all DICOM files within directory_path using modality prefixes."""
    for root_dir, _, files in os.walk(directory_path):
        for fname in files:
            try:
                process_single_dicom_file(root_dir, fname)
            except Exception:
                pass


def main():
    load_environment(".env")
    try:
        patient_id = os.environ.get('PATIENT_ID')
        rtplan_label = os.environ.get('RTPLAN_LABEL')
        rtplan_uid = os.environ.get('RTPLAN_UID')
    except IndexError:
        print("Usage: python gui.py <patient_id> <rtplan_label> <rtplan_uid>")
        sys.exit(1)

    input_dir = Path(os.environ.get("INPUT_DIR")) / patient_id

    root = tk.Tk()
    root.title("MRgTB Preprocessing")

    # Title label
    lbl_title = tk.Label(root, text="MRgTB Preprocessing", font=("Helvetica", 16))
    lbl_title.grid(row=0, column=0, columnspan=2, sticky="w", padx=10, pady=10)

    # Get Base Plan button with status label
    baseplan_status = tk.Label(root, text="", font=("Helvetica", 14))

    def on_get_base_plan():
        baseplan_status.config(text="\u23F3", fg="orange")  # hourglass
        root.update_idletasks()
        try:
            get_base_plan(patient_id, rtplan_label, rtplan_uid)
            baseplan_status.config(text="\u2705", fg="green")
        except Exception:
            baseplan_status.config(text="\u274C", fg="red")

    btn_baseplan = tk.Button(root, text="Get Base Plan", command=on_get_base_plan)
    btn_baseplan.grid(row=1, column=0, sticky="w", padx=10)
    baseplan_status.grid(row=1, column=1, sticky="w")

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
            series_info = list_imaging_series(str(input_dir))
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
                # attach trace to update dropdown when toggled
                var.trace_add('write', update_dropdown)

            # Update dropdown after loading
            update_dropdown()
            images_status.config(text="\u2705", fg="green")
        except Exception:
            images_status.config(text="\u274C", fg="red")

    btn_images = tk.Button(root, text="Get Imaging", command=on_get_images)
    btn_images.grid(row=2, column=0, sticky="w", padx=10, pady=(0, 5))
    images_status.grid(row=2, column=1, sticky="w")

    # Imaging series frame (initially empty)
    series_frame = tk.Frame(root)
    series_frame.grid(row=3, column=0, columnspan=2, sticky="w", padx=10, pady=10)

    # Dropdown menu for registration series
    selected_var = tk.StringVar()
    selection_map = {}
    tk.Label(root, text="Select Series for Matching").grid(row=4, column=0, columnspan=2, sticky="w", padx=10)
    dropdown = tk.OptionMenu(root, selected_var, '')
    dropdown.grid(row=5, column=0, columnspan=2, sticky="w", padx=10, pady=(0, 10))

    # Backup button under the dropdown
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

    # Clean up button
    cleanup_status = tk.Label(root, text="", font=("Helvetica", 14))

    def on_cleanup():
        cleanup_status.config(text="\u23F3", fg="orange")  # hourglass
        root.update_idletasks()
        try:
            # keep only the series with checkboxes ticked
            uids_to_keep = [uid for uid, var in series_vars.items() if var.get()]
            for uid, info in list(series_info.items()):
                if uid not in uids_to_keep:
                    for fpath in info["files"]:
                        try:
                            os.remove(fpath)
                        except Exception:
                            pass
            cleanup_status.config(text="\u2705", fg="green")
        except Exception:
            cleanup_status.config(text="\u274C", fg="red")
        # refresh displayed series after cleanup
        on_get_images()

    btn_cleanup = tk.Button(root, text="Clean up", command=on_cleanup)
    btn_cleanup.grid(row=7, column=0, sticky="w", padx=10, pady=(0, 10))
    cleanup_status.grid(row=7, column=1, sticky="w")

    # Resample button
    resample_status = tk.Label(root, text="", font=("Helvetica", 14))

    def on_resample():
        resample_status.config(text="\u23F3", fg="orange")  # hourglass
        root.update_idletasks()
        try:
            if check_if_ct_present(str(input_dir)):
                resample_ct(str(input_dir))
                resample_status.config(text="\u2705", fg="green")
            else:
                resample_status.config(text="\u274C", fg="red")
        except Exception:
            resample_status.config(text="\u274C", fg="red")

    btn_resample = tk.Button(root, text="Resample", command=on_resample)
    btn_resample.grid(row=8, column=0, sticky="w", padx=10, pady=(0, 10))
    resample_status.grid(row=8, column=1, sticky="w")

    # Register button
    register_status = tk.Label(root, text="", font=("Helvetica", 14))

    def on_register():
        register_status.config(text="\u23F3", fg="orange")
        root.update_idletasks()

        progress_win = tk.Toplevel(root)
        progress_win.title("Registering")
        progress_bar = ttk.Progressbar(progress_win, length=300, mode="determinate")
        progress_bar.pack(padx=20, pady=20)

        try:
            dicom_series = find_series_uids(str(input_dir))
            for series_uid, filepaths in dicom_series.items():
                create_empty_rtstruct(str(input_dir), series_uid, filepaths)

            def confirm():
                return messagebox.askyesno("Registration", "Accept registration result?")

            try:
                rigid_transform = perform_registration(str(input_dir), patient_id, rtplan_label, confirm_fn=confirm)
            except Exception:
                rigid_transform = None

            if rigid_transform:
                def progress_cb(idx, total):
                    progress_bar["maximum"] = total
                    progress_bar["value"] = idx
                    progress_win.update_idletasks()

                copy_structures(
                    str(input_dir),
                    patient_id,
                    rtplan_label,
                    rigid_transform,
                    progress_callback=progress_cb,
                )
            register_status.config(text="\u2705", fg="green")
        except Exception:
            register_status.config(text="\u274C", fg="red")
        finally:
            progress_win.destroy()

    btn_register = tk.Button(root, text="Register", command=on_register)
    btn_register.grid(row=9, column=0, sticky="w", padx=10, pady=(0, 10))
    register_status.grid(row=9, column=1, sticky="w")

    def update_dropdown(*args):
        # get all selected series
        selected_uids = [uid for uid, var in series_vars.items() if var.get()]
        menu = dropdown["menu"]
        menu.delete(0, 'end')
        selection_map.clear()
        for uid in selected_uids:
            text = checkbox_texts.get(uid, series_info[uid]['description'])
            selection_map[text] = uid
            menu.add_command(label=text, command=tk._setit(selected_var, text))
        # Default to first selection label
        if selected_uids:
            first = selected_uids[0]
            selected_var.set(checkbox_texts[first])
        else:
            selected_var.set('')

    root.mainloop()

if __name__ == '__main__':
    main()
