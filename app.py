import streamlit as st
import pandas as pd
import numpy as np
import os
from pathlib import Path
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="XBT Processing",
    layout="wide"
)

st.title("XBT Processing System")

# =====================================================
# USER INPUTS
# =====================================================

participants = st.text_input("Participants")

ship_name = st.text_input("Ship Name")

call_sign = st.text_input("Call Sign")

start_date = st.text_input("Start Date")

end_date = st.text_input("End Date")

# =====================================================
# FILE UPLOAD
# =====================================================

uploaded_files = st.file_uploader(
    "Upload XBT Files",
    accept_multiple_files=True
)

# =====================================================
# PROCESS BUTTON
# =====================================================

if st.button("Process XBT Data"):

    st.success(
        f"{len(uploaded_files)} files uploaded"
    )

    # Here call:
    # create_edf()
    # plotting()
    # interpolation()

    st.write("Processing completed")
