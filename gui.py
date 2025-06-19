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
        patient_id = sys.argv[1]
        rtplan_label = sys.argv[2]
        rtplan_uid = sys.argv[3]
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
    status_label = tk.Label(root, text="", font=("Helvetica", 14))

    def on_get_base_plan():
        get_base_plan(patient_id, rtplan_label, rtplan_uid)
        status_label.config(text="\u2705", fg="green")

    btn_baseplan = tk.Button(root, text="Get Base Plan", command=on_get_base_plan)
    btn_baseplan.grid(row=1, column=0, sticky="w", padx=10)
    status_label.grid(row=1, column=1, sticky="w")

    # Imaging series checkboxes
    series_frame = tk.Frame(root)
    series_frame.grid(row=2, column=0, columnspan=2, sticky="w", padx=10, pady=10)

    tk.Label(series_frame, text="Imaging series available:").pack(anchor="w")

    series_info = list_imaging_series(str(input_dir))
    series_vars = {}
    for uid, info in series_info.items():
        text = f"{info['date']} {info['time']} – {info['modality']} - {info['description']} (files={len(info['files'])})"
        var = tk.BooleanVar(value=False)
        chk = tk.Checkbutton(series_frame, text=text, variable=var)
        chk.pack(anchor="w")
        series_vars[uid] = var

    # Dropdown menu for registration series
    selected_var = tk.StringVar()
    dropdown = tk.OptionMenu(root, selected_var, "")
    dropdown.grid(row=3, column=0, columnspan=2, sticky="w", padx=10, pady=(0, 10))

    def update_dropdown(*args):
        selected = [uid for uid, v in series_vars.items() if v.get()]
        menu = dropdown["menu"]
        menu.delete(0, "end")
        for uid in selected:
            menu.add_command(label=uid, command=tk._setit(selected_var, uid))
        if selected:
            selected_var.set(selected[0])
        else:
            selected_var.set("")

    for v in series_vars.values():
        v.trace_add("write", update_dropdown)

    root.mainloop()


if __name__ == "__main__":
    main()
