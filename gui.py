import os
import sys
import tkinter as tk
from pathlib import Path

from preprocessing import list_imaging_series
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

    def on_get_images():
        nonlocal series_info, series_vars
        # Load imaging series only when button is clicked
        series_info = list_imaging_series(str(input_dir))
        # Clear any previous entries
        for widget in series_frame.winfo_children()[1:]:
            widget.destroy()
        series_vars.clear()
        # Populate checkboxes for each series
        for uid, info in series_info.items():
            text = f"{info['date']} {info['time']} – {info['modality']} - {info['description']} (files={len(info['files'])})"
            var = tk.BooleanVar(value=False)
            chk = tk.Checkbutton(series_frame, text=text, variable=var)
            chk.pack(anchor='w')
            series_vars[uid] = var
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
    tk.Label(series_frame, text="Imaging series available:").pack(anchor="w")

    # Dropdown menu for registration series
    selected_var = tk.StringVar()
    selection_map = {}
    tk.Label(root, text="Select Series for Matching").grid(row=4, column=0, columnspan=2, sticky="w", padx=10)
    dropdown = tk.OptionMenu(root, selected_var, '')
    dropdown.grid(row=5, column=0, columnspan=2, sticky="w", padx=10, pady=(0, 10))

    def update_dropdown(*args):
        selected_uids = [uid for uid, var in series_vars.items() if var.get()]
        menu = dropdown["menu"]
        menu.delete(0, 'end')
        selection_map.clear()
        for uid in selected_uids:
            desc = series_info[uid]['description']
            selection_map[desc] = uid
            menu.add_command(label=desc, command=tk._setit(selected_var, desc))
        # Default to first selection
        if selected_uids:
            selected_var.set(series_info[selected_uids[0]]['description'])
        else:
            selected_var.set('')

    # Trace checkbox changes to update dropdown
    # (will start having traces when series loaded)
    def attach_traces():
        for var in series_vars.values():
            var.trace_add('write', update_dropdown)

    # Call attach_traces after loading series
    attach_traces()

    root.mainloop()

if __name__ == '__main__':
    main()
