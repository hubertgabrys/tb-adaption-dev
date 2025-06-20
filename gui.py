import os
import sys
import tkinter as tk
from tkinter.scrolledtext import ScrolledText
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

    # Configure grid to accommodate console on the right
    root.grid_columnconfigure(2, weight=1)

    # Console output widget
    console = ScrolledText(root, state="disabled", width=80)
    console.grid(row=0, column=2, rowspan=15, sticky="nsew", padx=(10, 10), pady=10)

    # Redirect stdout and stderr to the console widget
    sys.stdout = ConsoleRedirector(console)
    sys.stderr = ConsoleRedirector(console)

    # Title label
    lbl_title = tk.Label(root, text="MRgTB Preprocessing", font=("Helvetica", 16))
    lbl_title.grid(row=0, column=0, columnspan=2, sticky="w", padx=10, pady=10)

    # Patient information displayed under the title
    lbl_patient = tk.Label(
        root,
        text=f"{patient_id}",
        font=("Helvetica", 12),
    )
    lbl_patient.grid(row=1, column=0, columnspan=2, sticky="w", padx=10)

    lbl_rtplan = tk.Label(
        root,
        text=f"{rtplan_label}",
        font=("Helvetica", 12),
    )
    lbl_rtplan.grid(row=2, column=0, columnspan=2, sticky="w", padx=10, pady=(0, 10))

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

    btn_baseplan = tk.Button(root, text="Get base plan", command=on_get_base_plan)
    btn_baseplan.grid(row=3, column=0, sticky="w", padx=10)
    baseplan_status.grid(row=3, column=1, sticky="w")

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

            # Update dropdown after loading
            update_dropdown()
            images_status.config(text="\u2705", fg="green")
        except Exception:
            images_status.config(text="\u274C", fg="red")

    btn_images = tk.Button(root, text="Get imaging", command=on_get_images)
    btn_images.grid(row=4, column=0, sticky="w", padx=10, pady=(0, 5))
    images_status.grid(row=4, column=1, sticky="w")

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
    btn_backup.grid(row=5, column=0, sticky="w", padx=10)
    backup_status.grid(row=5, column=1, sticky="w")

    # Imaging series frame (initially empty)
    series_frame = tk.Frame(root)
    series_frame.grid(row=6, column=0, columnspan=2, sticky="w", padx=10, pady=10)


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
    btn_cleanup.grid(row=9, column=0, sticky="w", padx=10, pady=(0, 10))
    cleanup_status.grid(row=9, column=1, sticky="w")

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
    btn_resample.grid(row=10, column=0, sticky="w", padx=10, pady=(0, 10))
    resample_status.grid(row=10, column=1, sticky="w")

    # Dropdown menu for registration series
    selected_var = tk.StringVar()
    selection_map = {}

    tk.Label(root, text="Select Series for Registration").grid(row=11, column=0, columnspan=2, sticky="w", padx=10)
    dropdown = tk.OptionMenu(root, selected_var, '')
    dropdown.grid(row=12, column=0, columnspan=2, sticky="w", padx=10, pady=(0, 10))

    # Register button
    register_status = tk.Label(root, text="", font=("Helvetica", 14))

    def on_register():
        register_status.config(text="\u23F3", fg="orange")
        root.update_idletasks()
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
                copy_structures(
                    str(input_dir),
                    patient_id,
                    rtplan_label,
                    rigid_transform,
                )
            register_status.config(text="\u2705", fg="green")
        except Exception:
            register_status.config(text="\u274C", fg="red")
        finally:
            # refresh displayed series after cleanup
            on_get_images()

    btn_register = tk.Button(root, text="Register", command=on_register)
    btn_register.grid(row=13, column=0, sticky="w", padx=10, pady=(0, 10))
    register_status.grid(row=13, column=1, sticky="w")

    def update_dropdown(*args):
        # show all available series in the dropdown
        menu = dropdown["menu"]
        menu.delete(0, 'end')
        selection_map.clear()
        for uid, info in series_info.items():
            text = checkbox_texts.get(uid, info['description'])
            selection_map[text] = uid
            menu.add_command(label=text, command=tk._setit(selected_var, text))

        # Default to first label
        if series_info:
            first_uid = next(iter(series_info))
            selected_var.set(checkbox_texts.get(first_uid, series_info[first_uid]['description']))
        else:
            selected_var.set('')

    root.mainloop()

if __name__ == '__main__':
    main()
