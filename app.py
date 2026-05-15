# app.py — XBT Processing Application

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from scipy.interpolate import interp1d

from pathlib import Path
from datetime import datetime

import tempfile
import zipfile
import os

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="XBT Processing",
    layout="wide"
)

st.title("XBT Processing System")

st.markdown(
    """
Interactive XBT processing workflow for:

- EDF generation
- QC profile visualization
- 1m interpolation
- 5m interpolation
"""
)

# =====================================================
# README / USER GUIDE
# =====================================================

with st.expander("README / User Instructions", expanded=True):

    st.markdown(
        """
# XBT Processing Application

This application processes raw XBT files and generates:

- EDF files
- QC temperature profile plots
- 1m interpolated CSV
- 5m interpolated CSV

---

# Workflow

## Step 1 — Enter Cruise Information

Fill in:
- Participants
- Ship Name
- Call Sign
- Start Date
- End Date

---

## Step 2 — Upload Raw XBT Files

Upload all `.XBT` files together.

Supported format:
- MK21 / MK150 exported XBT raw files

---

## Step 3 — Click “Process XBT Data”

The application will automatically:

- generate EDF files
- apply SST correction
- create QC profile plots
- generate:
    - 1m interpolation
    - 5m interpolation

---

# Downloads

After processing, download:

- EDF ZIP archive
- QC figures
- 1m CSV
- 5m CSV

---

# Notes

- EDF files are automatically named:
    - xout_1.edf
    - xout_2.edf
    - etc.

- Uploaded files are processed temporarily.
- No permanent storage is used.

---

# Developed For

Oceanographic XBT processing and quality control workflows.
"""
    )

# =====================================================
# SESSION STATE
# =====================================================

if "processed" not in st.session_state:
    st.session_state.processed = False

if "outputs" not in st.session_state:
    st.session_state.outputs = {}

# =====================================================
# HELPER FUNCTIONS
# =====================================================

def extractor(index):

    return str(index)


# =====================================================
# SST CORRECTION
# =====================================================

def corrector(df_raw):

    df_raw = df_raw.dropna(
        subset=['Depth', 'Temperature']
    )

    if df_raw.empty:
        return df_raw

    closest_depth_idx = (
        df_raw['Depth'] - 5
    ).abs().idxmin()

    temp_at_5m = df_raw.loc[
        closest_depth_idx,
        'Temperature'
    ]

    df_raw.loc[
        df_raw['Depth'] < 5,
        'Temperature'
    ] = temp_at_5m

    return df_raw


# =====================================================
# CREATE EDF
# =====================================================

def create_edf(
    file_path,
    participants,
    cd,
    ed,
    output_dir,
    idx
):

    file_name = os.path.basename(file_path)

    formatted_number = extractor(idx)

    with open(file_path, "r") as file:

        first_line = file.readline().strip()

    metadata = first_line.split(',')

    metadata_dict = {

        "Date of Launch":
            datetime.strptime(
                metadata[2].strip(),
                "%Y%m%d"
            ).strftime("%m/%d/%Y"),

        "Time of Launch":
            datetime.strptime(
                metadata[3].strip(),
                "%H%M%S"
            ).strftime("%H:%M:%S"),

        "Sequence #":
            metadata[1].strip(),

        "Latitude":
            metadata[4].strip(),

        "Longitude":
            metadata[5].strip(),

        "Probe type":
            metadata[6].strip(),

        "Transect route dated from":
            f"{cd} to {ed}",

        "Participants are":
            participants
    }

    metadata_text = (
        " // THIS IS AN MK-150 EXPORT DATA FILE (EDF)\n"
    )

    for key, value in metadata_dict.items():

        metadata_text += f"{key}: {value}\n"

    df = pd.read_csv(
        file_path,
        skiprows=1,
        header=None
    )

    df = df.iloc[:, :2]

    df.columns = [
        "Depth",
        "Temperature"
    ]

    df["Depth"] = pd.to_numeric(
        df["Depth"],
        errors="coerce"
    )

    df["Temperature"] = pd.to_numeric(
        df["Temperature"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["Depth", "Temperature"]
    )

    if df.empty:
        return None

    df["Depth"] = df["Depth"].round(1)

    df = corrector(df)

    df["Resistance"] = 9999.99

    output_name = (
        output_dir
        / f"xout_{formatted_number}.edf"
    )

    with open(output_name, "w") as f:

        f.write(metadata_text)

    df.to_csv(
        output_name,
        sep="\t",
        index=False,
        mode="a",
        float_format="%.3f"
    )

    return output_name


# =====================================================
# EDF READER
# =====================================================

def read_edf(edf_file):

    with open(edf_file, 'r') as f:

        lines = f.readlines()

    header_idx = None

    for idx, line in enumerate(lines):

        if "Depth" in line and "Temperature" in line:

            header_idx = idx
            break

    df = pd.read_csv(
        edf_file,
        sep=r"\s+",
        skiprows=header_idx,
        engine='python'
    )

    return df


# =====================================================
# EXTRACT METADATA
# =====================================================

def extract_meta(filepath):

    latitude = None
    longitude = None

    date_str = None
    time_str = None

    with open(filepath, 'r') as file:

        for line in file:

            line = line.strip()

            if 'Latitude:' in line:
                latitude = line.split(':', 1)[1].strip()

            elif 'Longitude:' in line:
                longitude = line.split(':', 1)[1].strip()

            elif 'Date of Launch:' in line:
                date_str = line.split(':', 1)[1].strip()

            elif 'Time of Launch:' in line:
                time_str = line.split(':', 1)[1].strip()

    dt = datetime.strptime(
        f"{date_str} {time_str}",
        '%m/%d/%Y %H:%M:%S'
    )

    dt_string = dt.strftime(
        '%Y-%m-%d %H:%M:%S'
    )

    return latitude, longitude, dt_string


# =====================================================
# SIDEBAR INPUTS
# =====================================================

st.sidebar.header("Cruise Information")

participants = st.sidebar.text_input(
    "Participants"
)

ship_name = st.sidebar.text_input(
    "Ship Name"
)

call_sign = st.sidebar.text_input(
    "Call Sign"
)

start_date = st.sidebar.text_input(
    "Start Date"
)

end_date = st.sidebar.text_input(
    "End Date"
)

# =====================================================
# FILE UPLOAD
# =====================================================

uploaded_files = st.file_uploader(
    "Upload Raw XBT Files",
    accept_multiple_files=True,
    type=["xbt", "XBT"]
)

# =====================================================
# PROCESS BUTTON
# =====================================================

if st.button("Process XBT Data"):

    if not uploaded_files:

        st.error("Please upload XBT files.")

    else:

        with tempfile.TemporaryDirectory() as tmpdir:

            tmpdir = Path(tmpdir)

            raw_dir = tmpdir / "raw"
            edf_dir = tmpdir / "edf"
            ivdata_dir = tmpdir / "ivdata"

            raw_dir.mkdir()
            edf_dir.mkdir()
            ivdata_dir.mkdir()

            # =============================================
            # SAVE UPLOADED FILES
            # =============================================

            for uploaded_file in uploaded_files:

                save_path = (
                    raw_dir
                    / uploaded_file.name
                )

                with open(save_path, "wb") as f:

                    f.write(
                        uploaded_file.getbuffer()
                    )

            xbt_files = sorted(
                raw_dir.glob("*")
            )

            edf_files = []

            st.info(
                f"Found {len(xbt_files)} XBT files"
            )

            # =============================================
            # GENERATE EDF
            # =============================================

            for idx, file in enumerate(
                xbt_files,
                start=1
            ):

                edf_file = create_edf(
                    file,
                    participants,
                    start_date,
                    end_date,
                    edf_dir,
                    idx
                )

                if edf_file is not None:

                    edf_files.append(edf_file)

            st.success(
                f"Generated {len(edf_files)} EDF files"
            )

            # =============================================
            # QC FIGURES
            # =============================================

            fig = plt.figure(
                figsize=(12, 14)
            )

            gs = gridspec.GridSpec(
                2,
                2,
                width_ratios=[3, 1],
                height_ratios=[1, 1],
                hspace=0.16,
                wspace=0.05
            )

            ax1 = fig.add_subplot(gs[0, :])
            ax2 = fig.add_subplot(gs[1, 0])
            ax3 = fig.add_subplot(gs[1, 1])

            increm = 0
            offset = 3

            lines = []
            labels = []

            for file in edf_files:

                df = read_edf(file)

                line, = ax1.plot(
                    df["Temperature"] + increm,
                    df["Depth"],
                    linewidth=1.5
                )

                ax2.plot(
                    df["Temperature"],
                    df["Depth"],
                    linewidth=1.5
                )

                increm += offset

                lines.append(line)
                labels.append(file.name)

            ax1.invert_yaxis()
            ax2.invert_yaxis()

            ax1.set_title(
                "Temperature Profiles"
            )

            ax2.set_title(
                "Probe-to-Probe Consistency"
            )

            ax1.set_xlabel(
                "Temperature + Offset"
            )

            ax2.set_xlabel(
                "Temperature"
            )

            ax1.set_ylabel("Depth (m)")
            ax2.set_ylabel("Depth (m)")

            ax3.axis('off')

            ax3.legend(
                lines,
                labels,
                fontsize=10,
                frameon=True,
                loc='center'
            )

            st.pyplot(fig)

            # =============================================
            # SAVE FIGURES
            # =============================================

            fig_png = (
                tmpdir
                / "temperature_profiles.png"
            )

            fig_svg = (
                tmpdir
                / "temperature_profiles.svg"
            )

            fig.savefig(
                fig_png,
                dpi=300,
                bbox_inches='tight'
            )

            fig.savefig(
                fig_svg,
                bbox_inches='tight'
            )

            st.session_state.outputs[
                "fig_png"
            ] = fig_png.read_bytes()

            st.session_state.outputs[
                "fig_svg"
            ] = fig_svg.read_bytes()

            # =============================================
            # INTERPOLATION
            # =============================================

            intervals = [1, 5]

            for interp_interval in intervals:

                int_depth = np.arange(
                    0,
                    761,
                    interp_interval
                )

                all_rows = []

                for file in edf_files:

                    lat, lon, dt = extract_meta(file)

                    df = read_edf(file)

                    df = df.dropna(
                        subset=[
                            'Depth',
                            'Temperature'
                        ]
                    )

                    f_interp = interp1d(
                        df['Depth'],
                        df['Temperature'],
                        bounds_error=False,
                        fill_value=np.nan
                    )

                    interpolated_temp = (
                        f_interp(int_depth)
                    )

                    row = [

                        ship_name,
                        call_sign,
                        lat,
                        lon,
                        dt

                    ] + interpolated_temp.tolist()

                    all_rows.append(row)

                column_names = (

                    [
                        'Ship_Name',
                        'Call_Sign',
                        'Latitude',
                        'Longitude',
                        'Datetime'
                    ]

                    +

                    [
                        f'Dep_{int(d)}'
                        for d in int_depth
                    ]
                )

                final_df = pd.DataFrame(
                    all_rows,
                    columns=column_names
                )

                output_csv = (
                    ivdata_dir
                    / f"{interp_interval}m_interpolated.csv"
                )

                final_df.to_csv(
                    output_csv,
                    index=False
                )

                st.subheader(
                    f"{interp_interval}m Interpolated Data"
                )

                st.dataframe(
                    final_df.head()
                )

                st.session_state.outputs[
                    f"{interp_interval}m_csv"
                ] = output_csv.read_bytes()

            # =============================================
            # EDF ZIP
            # =============================================

            zip_path = (
                tmpdir
                / "edf_files.zip"
            )

            with zipfile.ZipFile(
                zip_path,
                'w'
            ) as zipf:

                for file in edf_files:

                    zipf.write(
                        file,
                        arcname=file.name
                    )

            st.session_state.outputs[
                "edf_zip"
            ] = zip_path.read_bytes()

            st.session_state.processed = True

            st.success(
                "Processing completed successfully"
            )

# =====================================================
# DOWNLOAD SECTION
# =====================================================

if st.session_state.processed:

    st.header("Downloads")

    col1, col2 = st.columns(2)

    with col1:

        st.download_button(
            label="Download EDF ZIP",
            data=st.session_state.outputs["edf_zip"],
            file_name="edf_files.zip",
            mime="application/zip"
        )

        st.download_button(
            label="Download 1m CSV",
            data=st.session_state.outputs["1m_csv"],
            file_name="1m_interpolated.csv",
            mime="text/csv"
        )

        st.download_button(
            label="Download 5m CSV",
            data=st.session_state.outputs["5m_csv"],
            file_name="5m_interpolated.csv",
            mime="text/csv"
        )

    with col2:

        st.download_button(
            label="Download Figure PNG",
            data=st.session_state.outputs["fig_png"],
            file_name="temperature_profiles.png",
            mime="image/png"
        )

        st.download_button(
            label="Download Figure SVG",
            data=st.session_state.outputs["fig_svg"],
            file_name="temperature_profiles.svg",
            mime="image/svg+xml"
        )

#end
