from datetime import datetime


def create_rtstruct_id(meta_data):
    """Generate a proposed ID based on the image modality and metadata."""
    modality = getattr(meta_data, "Modality", "")
    series_desc = getattr(meta_data, "SeriesDescription", "")
    date_str = getattr(meta_data, "StudyDate", "")

    date_str = datetime2yymmdd(date_str)

    if modality == "CT":
        prefix = ct2id_prefix(series_desc)
        return prefix if "%" in series_desc else f"{prefix}_{date_str}"
    elif modality == "MR":
        prefix = mr2id_prefix(series_desc)
        suffix = mr2id_suffix(series_desc)
        return f"{prefix}_{date_str}{suffix}"
    else:
        return f"{modality}_{date_str}"


def ct2id_prefix(comment):
    """Return the prefix for CT based on the comment."""
    ct_prefix_map = {
        "SyntheticCT": "sCT",
        "Synthetic CT": "sCT",
        "10%": "CT_10",
        "20%": "CT_20",
        "30%": "CT_30",
        "40%": "CT_40",
        "50%": "CT_50",
        "60%": "CT_60",
        "70%": "CT_70",
        "80%": "CT_80",
        "90%": "CT_90",
        "100%": "CT_100"
    }
    for key, value in ct_prefix_map.items():
        if key in comment:
            return value
    return "CT"


def mr2id_prefix(comment):
    """Return the prefix for MR based on the comment."""
    mr_prefix_map = {
        "sCT": "DX",
        "t1_mprage": "T1M",
        "t1_space": "T1S",
        "t1": "T1",
        "t2": "T2",
        "flair": "Flair",
        "diff": "DWI",
        "tfi": "TF",
        "MRSIM": "TF"
    }
    for key, value in mr_prefix_map.items():
        if key in comment:
            prefix = value
            break
    else:
        prefix = "MR"

    if "pkm" in comment:
        prefix += "km"

    return prefix


def mr2id_suffix(comment):
    """Return the suffix for MR based on the comment."""
    mr_suffix_map = {
        "tra": "t",
        "cor": "c",
        "sag": "s",
        "LargeFOV": "F"
    }
    for key, value in mr_suffix_map.items():
        if key in comment:
            return value
    return ""


def datetime2yymmdd(date_str):
    """Convert a datetime string to YY-MM-DD format."""
    formatted_date = datetime.strptime(date_str, "%Y%m%d").strftime("%y-%m-%d")
    # if len(in_datetime) >= 10:
    #     return f"{in_datetime[8:10]}-{in_datetime[3:5]}-{in_datetime[0:2]}"
    return datetime.strptime(date_str, "%Y%m%d").strftime("%y-%m-%d")
