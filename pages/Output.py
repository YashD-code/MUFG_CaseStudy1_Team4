"""
Output.py
Page: Output
Allows downloading the frontend-generated preview as Excel or CSV.
Backend handles full validation / multi-sheet exports.
"""

import streamlit as st
from utils import inject_css, save_df_to_excel_bytes

# Inject CSS
inject_css()

st.title("📤 Output")
st.write(
    "Download the preview result generated on the frontend. "
    "In production, backend would handle full validation and exports."
)

# ---------------------------------------
# SAFE FIX: Do NOT use "or" with DataFrames
# ---------------------------------------
preview_df = st.session_state.get("preview_df")
uploaded_df = st.session_state.get("uploaded_df")

df = preview_df if preview_df is not None else uploaded_df

if df is None:
    st.warning("⚠️ No data available. Upload and preview a file first.")
    st.stop()

# Excel download
excel_bytes = save_df_to_excel_bytes(df)
st.download_button(
    label="⬇️ Download Excel (.xlsx)",
    data=excel_bytes,
    file_name="preview_output.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    use_container_width=True
)

# CSV download
csv_bytes = df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="⬇️ Download CSV (.csv)",
    data=csv_bytes,
    file_name="preview_output.csv",
    mime="text/csv",
    use_container_width=True
)

st.markdown("---")
st.markdown("### Transformations applied (UI-level)")
st.json(st.session_state.get("transformations", {}))

# Navigation hint
st.info("⬅️ Use the left sidebar to go back to other pages (Upload / Transform / Preview).")
