"""
Transformation.py
Page: Transformation Selection UI
Collects user-specified transformation operations and saves them in session_state as JSON-like dict.
UI only — frontend collects inputs; backend will apply full transformations.
"""

import streamlit as st
from utils import inject_css, safe_apply_preview_transformations

# -----------------------------
# Inject CSS
# -----------------------------
inject_css()

st.title("⚙️ Transformation Selection")
st.write("Choose the transformations you want to apply. These controls collect user input only (frontend).")

# -----------------------------
# Ensure uploaded file exists
# -----------------------------
df = st.session_state.get("uploaded_df")
if df is None:
    st.warning("⚠️ No uploaded file found. Please upload an Excel file first on the 'Upload' page.")
    st.stop()

cols = df.columns.tolist()
trans = st.session_state.get("transformations", {})

st.markdown("### 🧹 Data Cleaning Operations")

# -----------------------------
# Remove duplicates
# -----------------------------
if st.checkbox("Remove duplicates"):
    cols_sel = st.multiselect("Columns to consider for duplicates", cols, help="Select columns to identify duplicate rows.")
    trans["remove_duplicates"] = {"columns": cols_sel}

# -----------------------------
# Replace values
# -----------------------------
if st.checkbox("Replace values"):
    rep_col = st.selectbox("Column", cols, key="rep_col")
    old = st.text_input("Value to replace (old)", key="old_val")
    new = st.text_input("New value", key="new_val")
    trans["replace_values"] = {"column": rep_col, "old": old, "new": new}

# -----------------------------
# Filter rows
# -----------------------------
if st.checkbox("Filter rows"):
    fcol = st.selectbox("Column to filter", cols, key="filter_col")
    cond = st.selectbox("Condition", [">", "<", "==", ">=", "<=", "!="], key="filter_cond")
    val = st.text_input("Value to compare (as string)", key="filter_val")
    trans["filter_rows"] = {"column": fcol, "condition": cond, "value": val}

# -----------------------------
# Merge columns
# -----------------------------
if st.checkbox("Merge columns"):
    merge_cols = st.multiselect("Columns to merge", cols, help="Order matters: left-to-right.")
    sep = st.text_input("Separator", value=" ")
    out_name = st.text_input("Output column name", value="merged")
    trans["merge_columns"] = {"columns": merge_cols, "sep": sep, "output": out_name}

# -----------------------------
# Date formatting
# -----------------------------
if st.checkbox("Date formatting"):
    date_col = st.selectbox("Date column", cols, key="date_col")
    to_fmt = st.text_input("Target format (strftime), e.g. %Y-%m-%d", value="%Y-%m-%d")
    trans["date_formatting"] = {"column": date_col, "to_format": to_fmt}

st.markdown("---")
st.markdown("### ➗ Mathematical Operations (UI only)")

# -----------------------------
# Simple math operation
# -----------------------------
if st.checkbox("Create new column from two columns (Add/Sub/Mul/Div)"):
    mcol1 = st.selectbox("Column 1", cols, key="math_col1")
    mop = st.selectbox("Operation", ["+", "-", "*", "/"], key="math_op")
    mcol2 = st.selectbox("Column 2", cols, key="math_col2")
    mout = st.text_input("Output column name", value="calc_result")
    trans["math_op"] = {"col1": mcol1, "op": mop, "col2": mcol2, "output": mout}

# -----------------------------
# Normalize text
# -----------------------------
if st.checkbox("Normalize text (trim/lowercase)"):
    ncol = st.selectbox("Text column", cols, key="norm_col")
    lower = st.checkbox("Lowercase", key="norm_lower")
    trim = st.checkbox("Trim spaces", key="norm_trim")
    trans["normalize_text"] = {"column": ncol, "lower": lower, "trim": trim}

# -----------------------------
# Save transformations
# -----------------------------
st.session_state["transformations"] = trans

# -----------------------------
# Show current transformation JSON
# -----------------------------
st.markdown("### Current transformation JSON (UI-level)")
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.code(str(trans))
st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# Navigation buttons
# -----------------------------
st.write("")
c1, c2 = st.columns([1, 1])

with c1:
    if st.button("Back to Upload"):
        st.success("⬅️ Navigating to Upload page...")
        st.switch_page("pages/Upload.py")

with c2:
    if st.button("Preview →"):
        if not df.empty:
            preview_df = safe_apply_preview_transformations(df, trans)
            st.session_state["preview_df"] = preview_df

        st.success("➡️ Navigating to Preview page...")
        st.switch_page("pages/Preview.py")
