# With GUI
# Author: Trivenee
import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Mathematical Operations Engine", layout="wide")

# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------

def percentage_change(series1, series2):
    """Calculate percentage change between two columns."""
    return ((series2 - series1) / series1) * 100

def weighted_average(df, cols, weights):
    """Calculate weighted average of selected columns."""
    weights = np.array(weights)
    return (df[cols] * weights).sum(axis=1) / weights.sum()

# ---------------------------------------------------------
# Main UI
# ---------------------------------------------------------

st.title("📊 Mathematical Operations Engine")
st.write("A professional tool to perform numerical transformations on Excel data.")

uploaded_file = st.file_uploader("Upload Excel File (.xlsx)", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.subheader("📄 Preview of Uploaded File")
    st.dataframe(df, use_container_width=True)

    st.divider()

    st.subheader("🛠 Select Mathematical Operation")

    operation = st.selectbox(
        "Choose an operation",
        [
            "Add Columns",
            "Subtract Columns",
            "Multiply Columns",
            "Divide Columns",
            "Aggregate Function",
            "Percentage Change",
            "Weighted Average",
            "Custom Formula"
        ]
    )

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    if operation in ["Add Columns", "Subtract Columns", "Multiply Columns", "Divide Columns"]:
        col1 = st.selectbox("Select Column 1", numeric_cols)
        col2 = st.selectbox("Select Column 2", numeric_cols)
        new_name = st.text_input("Name for New Column")

        if st.button("Apply Operation"):
            if operation == "Add Columns":
                df[new_name] = df[col1] + df[col2]
            elif operation == "Subtract Columns":
                df[new_name] = df[col1] - df[col2]
            elif operation == "Multiply Columns":
                df[new_name] = df[col1] * df[col2]
            elif operation == "Divide Columns":
                df[new_name] = df[col1] / df[col2].replace(0, np.nan)

            st.success("Operation Applied Successfully!")
            st.dataframe(df)

    elif operation == "Aggregate Function":
        col = st.selectbox("Select Column", numeric_cols)
        func = st.selectbox("Choose Function", ["sum", "mean", "median", "min", "max"])
        new_name = st.text_input("New column name")

        if st.button("Apply Aggregation"):
            value = None
            if func == "sum":
                value = df[col].sum()
            elif func == "mean":
                value = df[col].mean()
            elif func == "median":
                value = df[col].median()
            elif func == "min":
                value = df[col].min()
            elif func == "max":
                value = df[col].max()

            df[new_name] = value
            st.success("Aggregation Applied!")
            st.dataframe(df)

    elif operation == "Percentage Change":
        base_col = st.selectbox("Select Old Value Column", numeric_cols)
        new_col = st.selectbox("Select New Value Column", numeric_cols)
        new_name = st.text_input("Name for % Change Column")

        if st.button("Calculate Percentage Change"):
            df[new_name] = percentage_change(df[base_col], df[new_col])
            st.success("Percentage Change Calculated!")
            st.dataframe(df)

    elif operation == "Weighted Average":
        selected_cols = st.multiselect("Select Columns", numeric_cols)
        weights = st.text_input("Enter Weights (comma separated)")

        new_name = st.text_input("Name for Weighted Average Column")

        if st.button("Apply Weighted Average"):
            try:
                weight_list = list(map(float, weights.split(",")))

                if len(selected_cols) != len(weight_list):
                    st.error("Number of weights must match number of columns.")
                else:
                    df[new_name] = weighted_average(df, selected_cols, weight_list)
                    st.success("Weighted Average Calculated!")
                    st.dataframe(df)
            except:
                st.error("Invalid weights input.")

    elif operation == "Custom Formula":
        st.write("Use formula like: `A + B * 0.18` or `(Price * Quantity) / 100`")
        formula = st.text_input("Enter formula")
        new_name = st.text_input("Name for New Column")

        if st.button("Apply Formula"):
            try:
                df[new_name] = df.eval(formula)
                st.success("Formula Applied Successfully!")
                st.dataframe(df)
            except Exception as e:
                st.error(f"Invalid formula: {e}")

    st.divider()

    # ---------------------------------------------------------
    # Download Output
    # ---------------------------------------------------------
    st.subheader("⬇ Download Processed File")

    output_excel = df.to_excel("processed_output.xlsx", index=False)

    with open("processed_output.xlsx", "rb") as f:
        st.download_button(
            label="Download Excel File",
            data=f,
            file_name="processed_output.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
