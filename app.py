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
    page_title="Interactive XBT QC System",
    layout="wide"
)

st.title("Interactive XBT QC Processing System")

# =====================================================
# README
# =====================================================

with st.expander("README / User Instructions", expanded=True):

    st.markdown(
        """
# Workflow

1. Upload raw .XBT files
2. Generate EDF files + Initial QC plots
3. Visually inspect spikes/spurious values
4. Edit EDF tables manually
5. Regenerate QC plots
6. Download corrected EDF ZIP

---

# QC Procedure

- Remove spike rows
- Edit wrong temperatures
- Edit wrong depths
- Remove bottom hit values
- Replot until QC is acceptable

---

# Outputs

- Corrected EDF ZIP
- QC Figures
- Probe-to-probe consistency plots
"""
    )

# =====================================================
# SESSION STATE
# =====================================================

if "edf_data" not in st.session_state:
    st.session_state.edf_data = {}

if "processed" not in st.session_state:
    st.session_state.processed = False

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
# CREATE EDF DATA
# =====================================================

def create_edf(
    file_path,
    participants,
    cd,
    ed
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
# QC PLOTS
# =====================================================

def generate_qc_plot(edf_data):

    fig = plt.figure(
        figsize=(14, 14)
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

    for key in edf_data:

        df = edf_data[key]["df"]

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
        labels.append(key)

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

    ax1.set_ylabel(
        "Depth (m)"
    )

    ax2.set_ylabel(
        "Depth (m)"
    )

    ax3.axis('off')

    ax3.legend(
        lines,
        labels,
        fontsize=10,
        frameon=True,
        loc='center'
    )

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
# SAMPLE DATA DOWNLOADS
# =====================================================

st.header("Sample XBT Files")

st.info(
    "New users can download sample XBT files below for trial processing."
    
st.markdown(
    "Download sample files for trial runs and testing."
)

sample_dir = Path("sample_data")

if sample_dir.exists():

    sample_files = list(
        sample_dir.glob("*.XBT")
    )

    for sample_file in sample_files:

        with open(sample_file, "rb") as f:

            st.download_button(
                label=f"Download {sample_file.name}",
                data=f,
                file_name=sample_file.name,
                mime="application/octet-stream"
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

if st.button("Generate EDF + Initial QC"):

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
                end_date
            )

            file_key = f"xout_{idx}.edf"

            st.session_state.edf_data[
                file_key
            ] = {

                "metadata": metadata_dict,
                "df": df

            }

        st.session_state.processed = True

# =====================================================
# INITIAL QC PLOT
# =====================================================

if st.session_state.processed:

    st.header("Initial QC Plots")

    fig = generate_qc_plot(
        st.session_state.edf_data
    )

    st.pyplot(fig)

    # =============================================
    # SAVE INITIAL FIGURE
    # =============================================

    with tempfile.TemporaryDirectory() as tmpdir:

        tmpdir = Path(tmpdir)

        fig_path = (
            tmpdir
            / "initial_qc_plot.png"
        )

        fig.savefig(
            fig_path,
            dpi=300,
            bbox_inches='tight'
        )

        st.session_state.downloads[
            "initial_plot"
        ] = fig_path.read_bytes()

# =====================================================
# EDF EDITING
# =====================================================

if st.session_state.processed:

    st.header(
        "Interactive EDF Editing"
    )

    st.markdown(
        """
### Instructions

- Remove spike rows
- Edit bad temperatures
- Edit wrong depths
- Remove noisy tail values
"""
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

        edited_df["Depth"] = pd.to_numeric(
            edited_df["Depth"],
            errors="coerce"
        )

        edited_df["Temperature"] = pd.to_numeric(
            edited_df["Temperature"],
            errors="coerce"
        )

        edited_df = edited_df.dropna()

        st.session_state.edf_data[
            key
        ]["df"] = edited_df

# =====================================================
# REGENERATE QC
# =====================================================

if st.button("Regenerate QC Plots"):

    if st.session_state.processed:

        st.header(
            "Corrected QC Plots"
        )

        fig2 = generate_qc_plot(
            st.session_state.edf_data
        )

        st.pyplot(fig2)

        with tempfile.TemporaryDirectory() as tmpdir:

            tmpdir = Path(tmpdir)

            fig_path = (
                tmpdir
                / "corrected_qc_plot.png"
            )

            fig2.savefig(
                fig_path,
                dpi=300,
                bbox_inches='tight'
            )

            st.session_state.downloads[
                "corrected_plot"
            ] = fig_path.read_bytes()

# =====================================================
# DOWNLOAD SECTION
# =====================================================

if st.session_state.processed:

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

# =====================================================
# 1m AND 5m INTERPOLATION
# =====================================================

if st.session_state.processed:

    st.header("Interpolated Outputs")

    intervals = [1, 5]

    for interp_interval in intervals:

        int_depth = np.arange(
            0,
            761,
            interp_interval
        )

        all_rows = []

        for key in st.session_state.edf_data:

            metadata = st.session_state.edf_data[
                key
            ]["metadata"]

            df = st.session_state.edf_data[
                key
            ]["df"]

            df = df.dropna(
                subset=[
                    'Depth',
                    'Temperature'
                ]
            )

            df = df.sort_values(
                by="Depth"
            )

            # =========================================
            # INTERPOLATION
            # =========================================

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
                metadata["Latitude"],
                metadata["Longitude"],
                f"{metadata['Date of Launch']} "
                f"{metadata['Time of Launch']}"

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

        st.subheader(
            f"{interp_interval}m Interpolated Data"
        )

        st.dataframe(
            final_df.head()
        )

        # =========================================
        # SAVE CSV TO SESSION
        # =========================================

        csv_data = final_df.to_csv(
            index=False
        ).encode('utf-8')

        st.session_state.downloads[
            f"{interp_interval}m_csv"
        ] = csv_data


# =====================================================
# DOWNLOAD SECTION
# =====================================================

if st.session_state.processed:

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

        st.download_button(
            label="Download 1m CSV",
            data=st.session_state.downloads[
                "1m_csv"
            ],
            file_name="1m_interpolated.csv",
            mime="text/csv"
        )

        st.download_button(
            label="Download 5m CSV",
            data=st.session_state.downloads[
                "5m_csv"
            ],
            file_name="5m_interpolated.csv",
            mime="text/csv"
        )

    with col2:

        if "corrected_plot" in st.session_state.downloads:

            st.download_button(
                label="Download Corrected QC Plot",
                data=st.session_state.downloads[
                    "corrected_plot"
                ],
                file_name="corrected_qc_plot.png",
                mime="image/png"
            )

        else:

            st.download_button(
                label="Download Initial QC Plot",
                data=st.session_state.downloads[
                    "initial_plot"
                ],
                file_name="initial_qc_plot.png",
                mime="image/png"
            )



#end
