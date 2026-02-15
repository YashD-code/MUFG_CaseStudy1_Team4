import streamlit as st
import pandas as pd
from io import BytesIO
import os

# ===================== Page Config =====================
st.set_page_config(page_title="Data Preprocess & Math Ops", layout="wide")

st.title("Data Preprocessing & Mathematical Operations — Streamlit")
st.markdown("Upload an **Excel file (.xls or .xlsx)**")

# ===================== Excel Loader =====================
def load_excel(file) -> pd.DataFrame:
    ext = os.path.splitext(file.name)[1].lower()

    if ext == ".xlsx":
        return pd.read_excel(file, engine="openpyxl")

    elif ext == ".xls":
        return pd.read_excel(file, engine="xlrd")

    else:
        st.error("Unsupported file format")
        st.stop()

def download_df(df, filename="output.xlsx"):
    buffer = BytesIO()
    df.to_excel(buffer, index=False)
    st.download_button(
        label="Download Excel",
        data=buffer.getvalue(),
        file_name=filename,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# ===================== Upload =====================
uploaded_file = st.file_uploader(
    "Upload Excel file",
    type=["xls", "xlsx"]
)

if uploaded_file is None:
    st.info("Please upload an Excel file.")
    st.stop()

df = load_excel(uploaded_file)

# ===================== Preview =====================
st.subheader("Preview (First 5 Rows)")
st.dataframe(df.head())

work_df = df.copy()

# ===================== Sidebar =====================
st.sidebar.header("Choose Operation Group")
group = st.sidebar.radio(
    "Operation Group",
    ["Data preprocessing", "Mathematical operations"]
)

# =====================================================
# DATA PREPROCESSING
# =====================================================
if group == "Data preprocessing":

    ops = st.sidebar.multiselect(
        "Select operations",
        [
            "Remove duplicates",
            "Filter rows",
            "Replace values",
            "Merge columns",
            "Convert date format",
            "Normalize text"
        ]
    )

    if "Remove duplicates" in ops:
        dup_cols = st.sidebar.multiselect(
            "Columns (empty = all)",
            work_df.columns.tolist()
        )
        dup_keep = st.sidebar.selectbox(
            "Keep duplicate",
            ["first", "last", "none"]
        )

    if "Filter rows" in ops:
        f_col = st.sidebar.selectbox("Column", work_df.columns)
        f_op = st.sidebar.selectbox(
            "Operator", [">", "<", ">=", "<=", "==", "!=", "contains"]
        )
        f_val = st.sidebar.text_input("Value")

    if "Replace values" in ops:
        r_col = st.sidebar.selectbox("Column", work_df.columns)
        old_val = st.sidebar.text_input("Old value")
        new_val = st.sidebar.text_input("New value")

    if "Merge columns" in ops:
        m_cols = st.sidebar.multiselect(
            "Columns to merge",
            work_df.columns.tolist()
        )
        m_name = st.sidebar.text_input("New column name", "merged")
        m_sep = st.sidebar.text_input("Separator", " ")

    if "Convert date format" in ops:
        d_col = st.sidebar.selectbox("Date column", work_df.columns)
        d_fmt = st.sidebar.text_input("Output format", "%Y-%m-%d")

    if "Normalize text" in ops:
        n_col = st.sidebar.selectbox("Text column", work_df.columns)
        to_lower = st.sidebar.checkbox("Lowercase", True)
        to_strip = st.sidebar.checkbox("Trim spaces", True)

    if st.sidebar.button("Apply Preprocessing"):
        for op in ops:

            if op == "Remove duplicates":
                keep = False if dup_keep == "none" else dup_keep
                work_df = work_df.drop_duplicates(
                    subset=dup_cols if dup_cols else None,
                    keep=keep
                )

            elif op == "Filter rows":
                if f_op == "contains":
                    work_df = work_df[
                        work_df[f_col].astype(str).str.contains(f_val, na=False)
                    ]
                else:
                    work_df = work_df.query(f"`{f_col}` {f_op} @f_val")

            elif op == "Replace values":
                work_df[r_col] = work_df[r_col].replace(old_val, new_val)

            elif op == "Merge columns":
                work_df[m_name] = (
                    work_df[m_cols].astype(str).agg(m_sep.join, axis=1)
                )

            elif op == "Convert date format":
                work_df[d_col] = (
                    pd.to_datetime(work_df[d_col], errors="coerce")
                    .dt.strftime(d_fmt)
                )

            elif op == "Normalize text":
                s = work_df[n_col].astype(str)
                if to_strip:
                    s = s.str.strip()
                if to_lower:
                    s = s.str.lower()
                work_df[n_col] = s

        st.success("Preprocessing applied successfully")
        st.dataframe(work_df.head())
        download_df(work_df, "preprocessed.xlsx")

# =====================================================
# MATHEMATICAL OPERATIONS (FIXED)
# =====================================================
else:

    math_ops = st.sidebar.multiselect(
        "Select operations",
        [
            "Add / Subtract / Multiply / Divide",
            "Percentage change",
            "Weighted average",
            "Aggregate functions"
        ]
    )

    if "Add / Subtract / Multiply / Divide" in math_ops:
        c1 = st.sidebar.selectbox("Column 1", work_df.columns)
        c2 = st.sidebar.selectbox("Column 2", work_df.columns)
        op = st.sidebar.selectbox("Operation", ["+", "-", "*", "/"])
        new_col = st.sidebar.text_input("New column name", f"{c1}{op}{c2}")

    if "Percentage change" in math_ops:
        pc_col = st.sidebar.selectbox("Column", work_df.columns)
        pc_name = st.sidebar.text_input("New column", f"{pc_col}_pct_change")

    if "Weighted average" in math_ops:
        v_col = st.sidebar.selectbox("Value column", work_df.columns)
        w_col = st.sidebar.selectbox("Weight column", work_df.columns)
        wa_col = st.sidebar.text_input("New column", "weighted_avg")

    if "Aggregate functions" in math_ops:
        agg_cols = st.sidebar.multiselect(
            "Columns", work_df.columns.tolist()
        )
        agg_funcs = st.sidebar.multiselect(
            ["sum", "mean", "median", "min", "max"],
            default=["sum"]
        )

    if st.sidebar.button("Apply Math Operations"):
        for m in math_ops:

            if m == "Add / Subtract / Multiply / Divide":
                a = pd.to_numeric(work_df[c1], errors="coerce")
                b = pd.to_numeric(work_df[c2], errors="coerce")

                if op == "+":
                    work_df[new_col] = a + b
                elif op == "-":
                    work_df[new_col] = a - b
                elif op == "*":
                    work_df[new_col] = a * b
                elif op == "/":
                    work_df[new_col] = a / b

            elif m == "Percentage change":
                work_df[pc_name] = pd.to_numeric(
                    work_df[pc_col], errors="coerce"
                ).pct_change()

            elif m == "Weighted average":
                v = pd.to_numeric(work_df[v_col], errors="coerce")
                w = pd.to_numeric(work_df[w_col], errors="coerce")
                work_df[wa_col] = (v * w) / w.replace(0, pd.NA)

            elif m == "Aggregate functions":
                st.subheader("Aggregate Results")
                st.dataframe(
                    work_df[agg_cols]
                    .apply(pd.to_numeric, errors="coerce")
                    .agg(agg_funcs)
                )

        st.success("Mathematical operations applied successfully")
        st.dataframe(work_df.head())
        download_df(work_df, "math_results.xlsx")

# =====================================================
# FINAL PREVIEW
# =====================================================
st.markdown("---")
st.subheader("Final Data Preview")
st.dataframe(work_df.head(10))
