"""
Upload.py
Page: File Upload UI
Allows user to upload a single .xlsx file, shows basic metadata and a small preview.
This page is UI only — heavy validation belongs to backend.
"""

import streamlit as st
import pandas as pd
from utils import inject_css, set_session_defaults

# Inject CSS and ensure session keys
inject_css()
set_session_defaults({"uploaded_df": None, "transformations": {}, "preview_df": None})

st.title("📁 Upload Excel File")
st.write("Upload a single `.xlsx` file. This page shows a quick preview (first rows) and metadata.")

# File uploader
uploaded_file = st.file_uploader(
    "Upload Excel (.xlsx)", 
    type=["xlsx"], 
    accept_multiple_files=False,
    help="Select an Excel .xlsx file."
)

if uploaded_file:
    try:
        # Read only the first sheet to keep memory low (UI preview)
        df = pd.read_excel(uploaded_file, engine="openpyxl")
        st.session_state["uploaded_df"] = df
        st.success(f"✅ Loaded file: {uploaded_file.name}")
        
        # Metadata display
        st.markdown(f"**Rows:** {df.shape[0]}  &nbsp;&nbsp; **Columns:** {df.shape[1]}")

        # Premium card-style preview
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("**Preview (first 10 rows)**")
        st.dataframe(df.head(10), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    except Exception as e:
        st.error("⚠️ Unable to read Excel file. Please ensure the file is a valid .xlsx.")
        st.exception(e)
else:
    st.info("ℹ️ No file uploaded yet. Try the sample file or upload your own.")

# Small actions row
c1, c2 = st.columns([1,1])

with c1:
    if st.button("Clear file"):
        st.session_state["uploaded_df"] = None
        st.session_state["preview_df"] = None
        st.session_state["transformations"] = {}
        st.success("Cleared uploaded file.")

with c2:
    if st.button("Go to Transformations →"):
        if st.session_state.get("uploaded_df") is None:
            st.warning("⚠️ Please upload a file first.")
        else:
            st.success("➡️ Navigating to Transformation Selection page...")
            st.switch_page("pages/Transformation.py")
