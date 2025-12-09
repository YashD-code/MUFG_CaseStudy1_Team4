import streamlit as st
import pandas as pd
from io import StringIO

st.set_page_config(page_title="Data Preprocess & Math Ops", layout="wide")

st.title("Data Preprocessing & Mathematical Operations — Streamlit")
st.markdown(
    """
    Upload a CSV and apply common data-preprocessing operations (remove duplicates, filter rows,
    replace values, merge columns, convert date formats, normalize text) and mathematical
    operations (add/sub/mul/div columns, percentage change, weighted average, aggregates, conditional calculations).

    Instructions:
    1. Upload a CSV.
    2. Use the sidebar to pick operation groups and tune parameters.
    3. Click **Apply** to modify the dataframe and preview results.
    4. Download the modified CSV if desired.
    """
)

def load_csv(file) -> pd.DataFrame:
    return pd.read_csv(file)


def download_df(df, name="modified.csv"):
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(label="Download CSV", data=csv, file_name=name, mime='text/csv')


# -------------------- UI: upload and preview --------------------

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"]) 

if uploaded_file is None:
    st.info("Upload a CSV to get started. You can also paste a tiny sample below.")
    sample = st.text_area("Or paste CSV text (optional, small datasets)")
    if sample:
        try:
            df = pd.read_csv(StringIO(sample))
        except Exception as e:
            st.error(f"Could not read pasted CSV: {e}")
            st.stop()
    else:
        st.stop()
else:
    try:
        df = load_csv(uploaded_file)
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        st.stop()

st.subheader("Preview — first 5 rows")
st.dataframe(df.head())

# Work on a copy
work_df = df.copy()

# -------------------- Sidebar: choose operation --------------------

st.sidebar.header("Choose operation group")
group = st.sidebar.radio("Operation group", ["Data preprocessing", "Mathematical operations"]) 

# -------------------- Data preprocessing controls --------------------
if group == "Data preprocessing":
    st.sidebar.subheader("Preprocessing operations")
    ops = st.sidebar.multiselect("Select operations to apply (order: as listed)", [
        "Remove duplicates",
        "Filter rows based on condition",
        "Replace values",
        "Merge columns",
        "Convert date formats",
        "Normalize text (lowercase/trim)"
    ])

    # Remove duplicates options
    if "Remove duplicates" in ops:
        dedup_subset = st.sidebar.multiselect("Subset columns for duplicate check (empty = all columns)", options=work_df.columns.tolist())
        dedup_keep = st.sidebar.selectbox("Keep which duplicate", options=["first", "last", "none"], index=0)

    # Filter rows options (simple builder)
    if "Filter rows based on condition" in ops:
        st.sidebar.markdown("**Build a single simple condition**")
        filt_col = st.sidebar.selectbox("Column", options=work_df.columns.tolist(), key='fcol')
        filt_op = st.sidebar.selectbox("Operator", options=[">", "<", ">=", "<=", "==", "!=", "contains"], key='fop')
        filt_val = st.sidebar.text_input("Value (as text). For numeric comparisons enter a number", key='fval')

    # Replace values
    if "Replace values" in ops:
        rep_col = st.sidebar.selectbox("Column to replace values in", options=work_df.columns.tolist(), key='rcol')
        old_val = st.sidebar.text_input("Old value (exact match)", key='old')
        new_val = st.sidebar.text_input("New value", key='new')

    # Merge columns
    if "Merge columns" in ops:
        merge_cols = st.sidebar.multiselect("Columns to merge", options=work_df.columns.tolist(), key='mcols')
        merge_name = st.sidebar.text_input("New column name", value="merged", key='mname')
        merge_sep = st.sidebar.text_input("Separator", value=" ", key='msep')

    # Convert date formats
    if "Convert date formats" in ops:
        date_col = st.sidebar.selectbox("Date column", options=work_df.columns.tolist(), key='dcol')
        # Provide a hint about formats
        st.sidebar.markdown("Provide a python-style format string for output, e.g. `%Y-%m-%d` or `%d/%m/%Y`. Input `infer` to parse automatically.")
        date_out_fmt = st.sidebar.text_input("Output format (or 'infer')", value="%Y-%m-%d", key='dfmt')

    # Normalize text
    if "Normalize text (lowercase/trim)" in ops:
        norm_col = st.sidebar.selectbox("Text column", options=work_df.columns.tolist(), key='ncol')
        to_lower = st.sidebar.checkbox("Lowercase", value=True, key='nlower')
        to_strip = st.sidebar.checkbox("Trim whitespace", value=True, key='nstrip')

    apply = st.sidebar.button("Apply preprocessing")

    if apply:
        # Apply in the order selected by ops list
        try:
            for op in ops:
                if op == "Remove duplicates":
                    if dedup_keep == 'none':
                        work_df = work_df.drop_duplicates(subset=dedup_subset if dedup_subset else None, keep=False)
                    else:
                        work_df = work_df.drop_duplicates(subset=dedup_subset if dedup_subset else None, keep=dedup_keep)

                if op == "Filter rows based on condition":
                    if filt_op == 'contains':
                        work_df = work_df[work_df[filt_col].astype(str).str.contains(filt_val, na=False)]
                    else:
                        # try numeric comparison first
                        try:
                            num = float(filt_val)
                            expr = f"`{filt_col}` {filt_op} {num}"
                        except Exception:
                            expr = f"`{filt_col}` {filt_op} '{filt_val}'"
                        work_df = work_df.query(expr)

                if op == "Replace values":
                    work_df[rep_col] = work_df[rep_col].replace(old_val, new_val)

                if op == "Merge columns":
                    if len(merge_cols) >= 1:
                        work_df[merge_name] = work_df[merge_cols].astype(str).agg(merge_sep.join, axis=1)

                if op == "Convert date formats":
                    if date_out_fmt.lower() == 'infer':
                        work_df[date_col] = pd.to_datetime(work_df[date_col], errors='coerce')
                    else:
                        work_df[date_col] = pd.to_datetime(work_df[date_col], errors='coerce').dt.strftime(date_out_fmt)

                if op == "Normalize text (lowercase/trim)":
                    s = work_df[norm_col].astype(str)
                    if to_strip:
                        s = s.str.strip()
                    if to_lower:
                        s = s.str.lower()
                    work_df[norm_col] = s

            st.success("Preprocessing applied — preview updated below")
            st.dataframe(work_df.head())
            download_df(work_df, name="preprocessed.csv")
        except Exception as e:
            st.error(f"Error applying preprocessing: {e}")

# -------------------- Mathematical operations controls --------------------
else:
    st.sidebar.subheader("Mathematical operations")
    math_ops = st.sidebar.multiselect("Select operations to apply", [
        "Add/Subtract/Multiply/Divide columns",
        "Percentage change (colA -> colB)",
        "Weighted average",
        "Aggregate functions (sum, mean, median, min, max)",
        "Conditional calculation (if column A > value then ... else ... )"
    ])

    # Binary arithmetic
    if "Add/Subtract/Multiply/Divide columns" in math_ops:
        bin_col1 = st.sidebar.selectbox("Column 1", options=work_df.columns.tolist(), key='b1')
        bin_col2 = st.sidebar.selectbox("Column 2", options=work_df.columns.tolist(), key='b2')
        bin_op = st.sidebar.selectbox("Operation", options=["+", "-", "*", "/"], key='bop')
        bin_new = st.sidebar.text_input("New column name", value=f"{bin_col1}{bin_op}{bin_col2}", key='bnew')

    # Percentage change
    if "Percentage change (colA -> colB)" in math_ops:
        pc_col = st.sidebar.selectbox("Column for percent change", options=work_df.columns.tolist(), key='pc')
        pc_new = st.sidebar.text_input("New column name", value=f"{pc_col}_pct_change", key='pcnew')

    # Weighted average
    if "Weighted average" in math_ops:
        wa_val = st.sidebar.selectbox("Value column", options=work_df.columns.tolist(), key='wav')
        wa_wt = st.sidebar.selectbox("Weight column", options=work_df.columns.tolist(), key='waw')
        wa_new = st.sidebar.text_input("Result column name", value=f"{wa_val}_wavg", key='wan')

    # Aggregates
    if "Aggregate functions (sum, mean, median, min, max)" in math_ops:
        agg_cols = st.sidebar.multiselect("Columns to aggregate (numeric columns recommended)", options=work_df.columns.tolist(), key='aggc')
        agg_group = st.sidebar.selectbox("Optional: group by column (or None)", options=[None] + work_df.columns.tolist(), key='agg_grp')
        agg_funcs = st.sidebar.multiselect("Aggregate functions", options=["sum", "mean", "median", "min", "max"], default=["sum"] , key='aggf')

    # Conditional calc
    if "Conditional calculation (if column A > value then ... else ... )" in math_ops:
        c_col = st.sidebar.selectbox("Column for condition", options=work_df.columns.tolist(), key='ccol')
        c_op = st.sidebar.selectbox("Operator", options=[">", "<", ">=", "<=", "==", "!="], key='cop')
        c_val = st.sidebar.text_input("Threshold value", key='cval')
        c_true = st.sidebar.text_input("Value if True (literal or column name)", key='ctrue')
        c_false = st.sidebar.text_input("Value if False (literal or column name)", key='cfalse')
        c_new = st.sidebar.text_input("New column name", value=f"cond_{c_col}", key='cnew')

    apply_math = st.sidebar.button("Apply mathematical ops")

    if apply_math:
        try:
            for op in math_ops:
                if op == "Add/Subtract/Multiply/Divide columns":
                    a = pd.to_numeric(work_df[bin_col1], errors='coerce')
                    b = pd.to_numeric(work_df[bin_col2], errors='coerce')
                    if bin_op == '+':
                        work_df[bin_new] = a + b
                    elif bin_op == '-':
                        work_df[bin_new] = a - b
                    elif bin_op == '*':
                        work_df[bin_new] = a * b
                    elif bin_op == '/':
                        work_df[bin_new] = a / b

                if op == "Percentage change (colA -> colB)":
                    work_df[pc_new] = work_df[pc_col].pct_change()

                if op == "Weighted average":
                    vals = pd.to_numeric(work_df[wa_val], errors='coerce')
                    wts = pd.to_numeric(work_df[wa_wt], errors='coerce')
                    # elementwise weighted value; store per-row weighted value
                    work_df[wa_new] = (vals * wts) / wts.replace(0, pd.NA)
                    # note: overall weighted average can be computed as a single scalar:
                    overall_wavg = (vals * wts).sum(skipna=True) / wts.sum(skipna=True)
                    st.write(f"Overall weighted average (scalar): {overall_wavg}")

                if op == "Aggregate functions (sum, mean, median, min, max)":
                    if agg_group and agg_group in work_df.columns:
                        grouped = work_df.groupby(agg_group)[agg_cols]
                        agg_res = grouped.agg(agg_funcs)
                    else:
                        agg_res = work_df[agg_cols].agg(agg_funcs)
                    st.subheader("Aggregate results")
                    st.dataframe(agg_res)

                if op == "Conditional calculation (if column A > value then ... else ... )":
                    # build condition
                    try:
                        num = float(c_val)
                        cond = None
                        if c_op == '>':
                            cond = pd.to_numeric(work_df[c_col], errors='coerce') > num
                        elif c_op == '<':
                            cond = pd.to_numeric(work_df[c_col], errors='coerce') < num
                        elif c_op == '>=':
                            cond = pd.to_numeric(work_df[c_col], errors='coerce') >= num
                        elif c_op == '<=':
                            cond = pd.to_numeric(work_df[c_col], errors='coerce') <= num
                        elif c_op == '==':
                            cond = pd.to_numeric(work_df[c_col], errors='coerce') == num
                        elif c_op == '!=':
                            cond = pd.to_numeric(work_df[c_col], errors='coerce') != num
                        # evaluate true/false expressions: allow literals or reference to other columns
                        def eval_val(v):
                            if v in work_df.columns:
                                return work_df[v]
                            try:
                                return float(v)
                            except Exception:
                                return v
                        tval = eval_val(c_true)
                        fval = eval_val(c_false)
                        work_df[c_new] = work_df.apply(lambda row: tval[row.name] if isinstance(tval, pd.Series) else tval if cond.loc[row.name] else (fval[row.name] if isinstance(fval, pd.Series) else fval), axis=1)
                    except Exception:
                        st.error("Could not parse threshold as numeric for conditional calculation. For non-numeric conditions use preprocessing filters instead.")

            st.success("Mathematical operations applied — preview updated below")
            st.dataframe(work_df.head())
            download_df(work_df, name="math_ops_result.csv")

        except Exception as e:
            st.error(f"Error applying mathematical operations: {e}")

# -------------------- Footer: final preview --------------------

st.markdown("---")
st.subheader("Final dataframe preview (first 10 rows)")
st.dataframe(work_df.head(10))

st.caption("This app is a starter template — extend it with more robust parsing, multiple filters, safe eval guards, and richer UIs (e.g., multiple-step pipelines) as needed.")

