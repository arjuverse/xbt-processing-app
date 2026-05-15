# app.py — XBT Processing Application

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
    page_title="Interactive XBT QC",
    layout="wide"
)

st.title("Interactive XBT QC Processing System")

# =====================================================
# README
# =====================================================

with st.expander("README / Instructions", expanded=True):

    st.markdown(
        """
# Workflow

1. Upload raw XBT files
2. Generate EDF files
3. Inspect QC plots
4. Edit/remove spike values
5. Regenerate QC plots
6. Download corrected EDF files

---

# Editing Instructions

- Remove spike rows manually
- Edit bad temperatures/depths
- Changes are applied immediately
- QC plots regenerate automatically

---

# Outputs

- Corrected EDF ZIP
- QC plots
- 1m interpolation
- 5m interpolation
"""
    )

# =====================================================
# SESSION STATE
# =====================================================

if "edf_data" not in st.session_state:
    st.session_state.edf_data = {}

if "processed" not in st.session_state:
    st.session_state.processed = False

if "plots_generated" not in st.session_state:
    st.session_state.plots_generated = False

if "downloads" not in st.session_state:
    st.session_state.downloads = {}

# =====================================================
# HELPER FUNCTIONS
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
    idx
):

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

    df = df.dropna()

    df = corrector(df)

    df["Resistance"] = 9999.99

    return metadata_dict, df


# =====================================================
# SAVE EDF
# =====================================================

def save_edf(
    metadata_dict,
    df,
    output_path
):

    metadata_text = (
        " // THIS IS AN MK-150 EXPORT DATA FILE (EDF)\n"
    )

    for key, value in metadata_dict.items():

        metadata_text += f"{key}: {value}\n"

    with open(output_path, "w") as f:

        f.write(metadata_text)

    df.to_csv(
        output_path,
        sep="\t",
        index=False,
        mode="a",
        float_format="%.3f"
    )


# =====================================================
# QC PLOT
# =====================================================

def generate_plot(edf_data):

    fig, ax = plt.subplots(
        figsize=(8, 10)
    )

    offset = 0

    for key in edf_data:

        df = edf_data[key]["df"]

        ax.plot(
            df["Temperature"] + offset,
            df["Depth"],
            label=key
        )

        offset += 3

    ax.invert_yaxis()

    ax.set_xlabel(
        "Temperature + Offset"
    )

    ax.set_ylabel(
        "Depth (m)"
    )

    ax.set_title(
        "QC Temperature Profiles"
    )

    ax.legend()

    return fig


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

if st.button("Process XBT Files"):

    if not uploaded_files:

        st.error(
            "Please upload XBT files"
        )

    else:

        st.session_state.edf_data = {}

        for idx, uploaded_file in enumerate(
            uploaded_files,
            start=1
        ):

            with tempfile.NamedTemporaryFile(
                delete=False
            ) as tmp:

                tmp.write(
                    uploaded_file.getbuffer()
                )

                tmp_path = tmp.name

            metadata_dict, df = create_edf(
                tmp_path,
                participants,
                start_date,
                end_date,
                idx
            )

            file_key = f"xout_{idx}.edf"

            st.session_state.edf_data[
                file_key
            ] = {

                "metadata": metadata_dict,
                "df": df

            }

        st.session_state.processed = True

        st.success(
            "EDF files generated"
        )

# =====================================================
# EDITABLE EDF TABLES
# =====================================================

if st.session_state.processed:

    st.header(
        "Interactive QC Editing"
    )

    for key in st.session_state.edf_data:

        st.subheader(key)

        df = st.session_state.edf_data[
            key
        ]["df"]

        edited_df = st.data_editor(
            df,
            num_rows="dynamic",
            use_container_width=True,
            key=f"editor_{key}"
        )

        st.session_state.edf_data[
            key
        ]["df"] = edited_df

# =====================================================
# REGENERATE QC
# =====================================================

if st.button("Regenerate QC Plots"):

    if st.session_state.processed:

        fig = generate_plot(
            st.session_state.edf_data
        )

        st.pyplot(fig)

        # =============================================
        # SAVE FIGURE
        # =============================================

        with tempfile.TemporaryDirectory() as tmpdir:

            tmpdir = Path(tmpdir)

            fig_path = (
                tmpdir
                / "qc_plot.png"
            )

            fig.savefig(
                fig_path,
                dpi=300,
                bbox_inches='tight'
            )

            st.session_state.downloads[
                "qc_plot"
            ] = fig_path.read_bytes()

        st.session_state.plots_generated = True

# =====================================================
# DOWNLOAD SECTION
# =====================================================

if st.session_state.plots_generated:

    st.header("Downloads")

    # =============================================
    # CREATE EDF ZIP
    # =============================================

    with tempfile.TemporaryDirectory() as tmpdir:

        tmpdir = Path(tmpdir)

        edf_dir = tmpdir / "edf"

        edf_dir.mkdir()

        for key in st.session_state.edf_data:

            metadata = st.session_state.edf_data[
                key
            ]["metadata"]

            df = st.session_state.edf_data[
                key
            ]["df"]

            output_path = edf_dir / key

            save_edf(
                metadata,
                df,
                output_path
            )

        zip_path = (
            tmpdir
            / "corrected_edf.zip"
        )

        with zipfile.ZipFile(
            zip_path,
            'w'
        ) as zipf:

            for file in edf_dir.glob("*.edf"):

                zipf.write(
                    file,
                    arcname=file.name
                )

        st.session_state.downloads[
            "edf_zip"
        ] = zip_path.read_bytes()

    # =============================================
    # DOWNLOAD BUTTONS
    # =============================================

    col1, col2 = st.columns(2)

    with col1:

        st.download_button(
            label="Download Corrected EDF ZIP",
            data=st.session_state.downloads[
                "edf_zip"
            ],
            file_name="corrected_edf.zip",
            mime="application/zip"
        )

    with col2:

        st.download_button(
            label="Download QC Figure",
            data=st.session_state.downloads[
                "qc_plot"
            ],
            file_name="qc_plot.png",
            mime="image/png"
        )



#end
