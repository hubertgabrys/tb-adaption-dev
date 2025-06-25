import os
import sys
import traceback
import warnings
from pathlib import Path

from tqdm import tqdm

from copy_structures import copy_structures
from export import send2aria
from preprocessing import preprocess_input_directory, check_input_directory
from register import perform_registration, get_base_plan
from resampling import resample_ct
from segmentation import create_empty_rtstruct
from utils import get_datetime, find_series_uids, check_if_ct_present, load_environment

warnings.filterwarnings("ignore", category=UserWarning, module="pydicom")



def main():
    print("###########################")
    print("#                         #")
    print("#   MRgTB Preprocessing   #")
    print("#   version 2025.06.18    #")
    print("#                         #")
    print("###########################")
    print()

    # Load the .env
    load_environment(".env")

    # Production
    patient_id = sys.argv[1]
    rtplan_label = sys.argv[2]
    rtplan_uid = sys.argv[3]
    input_dir = Path(os.environ.get('INPUT_DIR'))
    current_directory = input_dir / patient_id
    # current_directory = f"\\\\ariaimg.usz.ch\\dicom$\\Import\\CT\\{patient_id}"

    # Testing
    # patient_id = os.environ.get('PATIENT_ID')
    # rtplan_label = os.environ.get('RTPLAN_LABEL')
    # current_directory = f"\\\\ariaimg.usz.ch\\dicom$\\Import\\CT\\{patient_id}"

    # Start
    print(f"{get_datetime()} START")

    # Get the base plan
    print()
    print(f"{get_datetime()} Exporting the base plan from Eclipse (only once for the first adaption)")
    get_base_plan(patient_id, rtplan_label, rtplan_uid)
    print("In the next step, the files exported from the MR scanner will be checked.")
    if input("Press ENTER to continue or type 'n' to exit..."):
        return

    # Checking the files in CT import folder
    print()
    print(f"{get_datetime()} Checking the files in CT import folder ({current_directory})")
    print("Please check if all imaging series were successfully exported from the MR scanner (the list below):")
    check_input_directory(current_directory)
    print("In the next step, the processing of the in CT import folder will start.")
    if input("Press ENTER to continue or type 'n' to exit..."):
        return

    # Step 1: Clean up input directory
    print()
    print(f"{get_datetime()} Preprocessing input directory".upper())
    preprocess_input_directory(current_directory)

    # Step 2: Resample CT
    print()
    print(f"{get_datetime()} Resampling CT".upper())
    ct_present = check_if_ct_present(current_directory)
    if ct_present:
        resample_ct(current_directory)
    else:
        print(f"{get_datetime()} sCT is not present in the input directory -> no resampling performed")

    # Step 3: Generate RTSTRUCT files
    print()
    print(f"{get_datetime()} Generating RTSTRUCTs".upper())
    dicom_series = find_series_uids(current_directory)
    for series_uid, filepaths in tqdm(dicom_series.items(), desc="Imaging series"):
        create_empty_rtstruct(current_directory, series_uid, filepaths)

    # Step 4: Register images
    print()
    print(f"{get_datetime()} Performing registration".upper())
    try:
        rigid_transform, fixed_uid, moving_uid = perform_registration(current_directory, patient_id, rtplan_label)
    except Exception:
        print(f"{get_datetime()} Registration failed")
        rigid_transform, fixed_uid, moving_uid = None, None, None

    # Step 5: Copy structures from the base plan to the new image
    print()
    print(f"{get_datetime()} Copying structures".upper())
    if rigid_transform:
        copy_structures(current_directory, patient_id, rtplan_label, rigid_transform,
                        series_uid=fixed_uid, base_series_uid=moving_uid)
    else:
        print(f"{get_datetime()} No registration performed -> no structures copied")

    # # Step 6: Send images, registrations, and RTSTRUCT to Aria
    # print()
    # print(f"{get_datetime()} Sending to ARIA (ca. 5 minutes)".upper())
    # try:
    #     send2aria(current_directory)
    # except Exception:
    #     print(f"{get_datetime()} Sending to ARIA failed! Please import the remaining files manually!")

    # Finished
    print()
    print("Please import the files manually in Eclipse. Thank you for using TB adaption script! :)")
    print()
    print(f"{get_datetime()} FINISHED!")


if __name__ == '__main__':
    try:
        main()
    except Exception:
        print("An error occurred:")
        traceback.print_exc()
    finally:
        # This will pause the cmd window so that you can see the messages before closing
        input("Press ENTER to exit...")

