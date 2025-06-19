import os
import sys
import tkinter as tk
from pathlib import Path

from preprocessing import list_imaging_series, zip_directory
from register import get_base_plan
from utils import load_environment


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
        get_base_plan(patient_id, rtplan_label, rtplan_uid)
        baseplan_status.config(text="\u2705", fg="green")

    btn_baseplan = tk.Button(root, text="Get Base Plan", command=on_get_base_plan)
    btn_baseplan.grid(row=1, column=0, sticky="w", padx=10)
    baseplan_status.grid(row=1, column=1, sticky="w")

    # Get Images button with status label
    images_status = tk.Label(root, text="", font=("Helvetica", 14))
    series_info = {}
    series_vars = {}
    checkbox_texts = {}

    def on_get_images():
        nonlocal series_info, series_vars, checkbox_texts
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
            text = f"{info['date']} {info['time']} – {info['modality']} - {info['description']} (files={len(info['files'])})"
            var = tk.BooleanVar(value=False)
            chk = tk.Checkbutton(series_frame, text=text, variable=var)
            chk.pack(anchor='w')
            series_vars[uid] = var
            checkbox_texts[uid] = text
            # attach trace to update dropdown when toggled
            var.trace_add('write', update_dropdown)

        # Update dropdown after loading
        update_dropdown()
        # Show success
        images_status.config(text="\u2705", fg="green")

    btn_images = tk.Button(root, text="Get Images", command=on_get_images)
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
        zip_directory(str(input_dir), str(input_dir) + ".zip")
        backup_status.config(text="\u2705", fg="green")

    btn_backup = tk.Button(root, text="Create backup", command=on_backup)
    btn_backup.grid(row=6, column=0, sticky="w", padx=10)
    backup_status.grid(row=6, column=1, sticky="w")

    # Clean up button
    cleanup_status = tk.Label(root, text="", font=("Helvetica", 14))

    def on_cleanup():
        selected_text = selected_var.get()
        uid_to_keep = selection_map.get(selected_text)
        if not uid_to_keep:
            return

        for uid, info in series_info.items():
            if uid != uid_to_keep:
                for fpath in info["files"]:
                    try:
                        os.remove(fpath)
                    except Exception:
                        pass

        cleanup_status.config(text="\u2705", fg="green")
        # refresh displayed series after cleanup
        on_get_images()

    btn_cleanup = tk.Button(root, text="Clean up", command=on_cleanup)
    btn_cleanup.grid(row=7, column=0, sticky="w", padx=10, pady=(0, 10))
    cleanup_status.grid(row=7, column=1, sticky="w")

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
