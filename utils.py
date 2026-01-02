"""
utils.py
Shared helpers for the Streamlit frontend (Task 1).
Contains CSS injector, session helpers and safe transformation helpers for UI preview.
All functions include docstrings as required.
"""

from io import BytesIO
import streamlit as st
import pandas as pd


def inject_css(path: str = "assets/style.css") -> None:
    """
    Inject CSS content from a file into the current Streamlit page.
    Args:
        path: path to the CSS file relative to the project root.
    """
    try:
        with open(path, "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        # Optional improvement: show a small warning
        st.warning("⚠ CSS file not found: assets/style.css (UI will still work)")


def set_session_defaults(defaults: dict) -> None:
    """
    Ensure session_state contains given keys with defaults.
    Args:
        defaults: dict of key: default_value pairs.
    """
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def save_df_to_excel_bytes(df: pd.DataFrame) -> bytes:
    """
    Convert DataFrame to xlsx bytes for download.
    Args:
        df: pandas DataFrame
    Returns:
        bytes: in-memory xlsx file bytes
    """
    buffer = BytesIO()

    # Improvement: specify the Excel writer engine explicitly
    df.to_excel(buffer, index=False, engine="openpyxl")

    buffer.seek(0)
    return buffer.read()


def safe_apply_preview_transformations(df: pd.DataFrame, transformations: dict) -> pd.DataFrame:
    """
    Apply a small subset of safe transformations to a copy of df for preview purposes.
    NOTE: This is UI-level preview only and not backend-grade processing.

    Args:
        df: original DataFrame
        transformations: dict describing operations selected in UI

    Returns:
        DataFrame: transformed copy for preview display
    """
    preview = df.copy()

    # ---------------------------
    # REMOVE DUPLICATES
    # ---------------------------
    rd = transformations.get("remove_duplicates")
    if rd and rd.get("columns"):
        cols = [c for c in rd["columns"] if c in preview.columns]
        if cols:
            preview = preview.drop_duplicates(subset=cols)

    # ---------------------------
    # REPLACE VALUES
    # ---------------------------
    rv = transformations.get("replace_values")
    if rv:
        col = rv.get("column")
        old = rv.get("old")
        new = rv.get("new")
        if col in preview.columns and old is not None:
            preview[col] = preview[col].replace(old, new)

    # ---------------------------
    # MERGE COLUMNS
    # ---------------------------
    mc = transformations.get("merge_columns")
    if mc and mc.get("columns"):
        cols = [c for c in mc["columns"] if c in preview.columns]
        if cols:
            sep = mc.get("sep", " ")
            out = mc.get("output", "merged")
            preview[out] = preview[cols].astype(str).agg(sep.join, axis=1)

    # ---------------------------
    # NORMALIZE TEXT
    # ---------------------------
    nt = transformations.get("normalize_text")
    if nt:
        col = nt.get("column")
        if col in preview.columns:
            if nt.get("trim"):
                preview[col] = preview[col].astype(str).str.strip()
            if nt.get("lower"):
                preview[col] = preview[col].astype(str).str.lower()

    # ---------------------------
    # SIMPLE MATH OP
    # ---------------------------
    mo = transformations.get("math_op")
    if mo:
        c1 = mo.get("col1")
        c2 = mo.get("col2")
        op = mo.get("op")
        out = mo.get("output", "calc_result")

        if c1 in preview.columns and c2 in preview.columns:
            try:
                a = pd.to_numeric(preview[c1], errors="coerce")
                b = pd.to_numeric(preview[c2], errors="coerce")

                if op == "+":
                    preview[out] = a + b
                elif op == "-":
                    preview[out] = a - b
                elif op == "*":
                    preview[out] = a * b
                elif op == "/":
                    # avoid divide-by-zero using NA
                    preview[out] = a / b.replace(0, pd.NA)

            except Exception:
                # Skip silently — preview MUST NOT crash
                pass

    return preview
