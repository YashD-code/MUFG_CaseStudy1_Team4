import streamlit as st
import pandas as pd
import plotly.express as px

from data_cleaning import DataCleaningEngine
from math_operations import MathOperationsEngine
from template_engine import (
    apply_template_to_dataframe,
    get_template_required_columns
)
from utils import (
    inject_css,
    initialize_session_state,
    load_saved_templates,
    save_templates,
    dataframe_to_excel_bytes,
    render_download_buttons,
    log_operation,
    save_current_operations_as_template
)
# ============================================================================
# MAIN UI LAYOUT
# ============================================================================

def main():
    """Main function to initialize the app and control the flow of the application.
    
    This function sets up the Streamlit app's layout, navigation, and page content, including the file upload,
    data transformation, template application, and logging functionalities.
    """
    st.set_page_config(
        page_title="Excel Data Transformer",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    initialize_session_state()
    inject_css()
    
    st.markdown("---")
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("Excel Data Transformation Tool")
        st.markdown("*Professional data cleaning & mathematical operations for Excel files*")
    with col2:
        st.info(f"Session: {st.session_state.get('current_sheet', 'None')}")
    st.markdown("---")
    
    with st.sidebar:
        st.header("⚙️ Navigation")
        page = st.radio(
            "Select a page:",
            ["🏠 Home",  "🔄 Transform Data", "💾 Templates", "📋 Logs"],
            key="page_selector"
        )
    
    # ========================================================================
    # PAGE: HOME
    # ========================================================================
    if page == "🏠 Home":
        st.header("Excel Data Transformation Tool!")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("● Key Features")
            st.markdown("""
- **File Upload & Validation**
- **Data Cleaning**
- **Math Operations**
- **Real-time Preview**
- **Templates** 
- **Multi-sheet Support**
            """)
        with col2:
            st.subheader("Quick Start")
            st.markdown("""
1. Upload your Excel file
2. Perform transformations
3. Save as template
4. On a new file → Templates → Apply → Map columns → Done!
            """)
        st.header("Upload Excel File")
        col1, col2 = st.columns([2, 1])
        with col1:
            uploaded_file = st.file_uploader(
                "Upload Excel file (.xlsx, .xls)",
                type=["xlsx", "xls"],
                help="Upload your Excel file here"
            )
        with col2:
            max_file_size = st.number_input(
                "Max file size (MB):",
                min_value=10,
                max_value=500,
                value=200
            )
        
        if uploaded_file:
            file_bytes = uploaded_file.read()
            file_size_mb = len(file_bytes) / (1024 * 1024)
            
            if file_size_mb > max_file_size:
                st.error(f"File size {file_size_mb:.2f} MB exceeds limit of {max_file_size} MB")
            else:
                try:
                    xls = pd.ExcelFile(uploaded_file)
                    sheet_names = xls.sheet_names
                    st.session_state["sheet_names"] = sheet_names
                    st.session_state["uploaded_filename"] = uploaded_file.name
                    
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        selected_sheet = st.selectbox(
                            "Select sheet:",
                            sheet_names,
                            index=0
                        )
                    
                    df = pd.read_excel(xls, sheet_name=selected_sheet)
                    st.session_state["uploaded_df"] = df
                    st.session_state["working_df"] = df.copy()
                    st.session_state["current_sheet"] = selected_sheet
                    st.session_state["current_operations"] = []
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1: st.metric("Rows", len(df))
                    with col2: st.metric("Columns", len(df.columns))
                    with col3: st.metric("File Size", f"{file_size_mb:.2f} MB")
                    with col4: st.metric("Sheets", len(sheet_names))
                    
                    st.subheader("- Data Preview")
                    preview_rows = st.slider("Rows to preview:", 5, min(100, len(df)), 10)
                    st.dataframe(df.head(preview_rows), use_container_width=True)
                    

                    # Assuming df is the uploaded dataframe
                    st.subheader("- Choose Visualization Type")

                    # Step 1: Let user select the type of chart
                    visualization_type = st.radio(
                        "Select Visualization Type",
                        ("Histogram", "Scatter Plot")
                    )

                    # Step 2: Let user select columns based on chosen chart type
                    if visualization_type == "Histogram":
                        # Allow user to select a numeric column for histogram
                        numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()
                        column = st.selectbox("Select a numeric column for the histogram", numeric_columns)
                        fig = px.histogram(
                            df,
                            x=column,
                            nbins=30,
                            title=f"- Distribution of {column}",
                            opacity=0.8,
                        )

                        # Improve layout
                        fig.update_layout(
                            template="plotly_dark",
                            title_font=dict(size=20, family="Arial"),
                            xaxis_title=f"{column} Values",
                            yaxis_title="Frequency",
                            bargap=0.05,
                        )

                        #  Add mean line
                        mean_val = df[column].mean()
                        fig.add_vline(
                            x=mean_val,
                            line_dash="dash",
                            line_color="red",
                            annotation_text=f"Mean: {mean_val:.2f}",
                            annotation_position="top"
                        )

                        st.plotly_chart(fig, use_container_width=True)

                    elif visualization_type == "Scatter Plot":
                        # Allow user to select two numeric columns for scatter plot
                        numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()
                        x_column = st.selectbox("Select the X-axis column for the scatter plot", numeric_columns)
                        y_column = st.selectbox("Select the Y-axis column for the scatter plot", numeric_columns)
                        if x_column and y_column:
                            fig = px.scatter(df, x=x_column, y=y_column, title=f"Scatter Plot: {x_column} vs {y_column}")
                            st.plotly_chart(fig)
                    
                    st.subheader("Column Information")
                    col_info = pd.DataFrame({
                        "Column": df.columns,
                        "Type": [str(dtype) for dtype in df.dtypes],
                        "Non-Null": [df[col].notna().sum() for col in df.columns],
                        "Missing": [df[col].isna().sum() for col in df.columns]
                    })
                    st.dataframe(col_info, use_container_width=True)
                    
                    st.success("File loaded successfully! Ready for transformation.")
                except Exception as e:
                    st.error(f"Error reading file: {str(e)}")
        else:
            st.info("Upload an Excel file to get started")
    
    # ========================================================================
    # PAGE: TRANSFORM DATA 
    # ========================================================================
    elif page == "🔄 Transform Data":
        if st.session_state["uploaded_df"] is None:
            st.warning(" X Please upload a file first!")
            st.stop()
        df = st.session_state["uploaded_df"]
        df_working = st.session_state["working_df"].copy()
        
        st.header("Transform Data")
        col_reset, col_info = st.columns([1, 3])

        with col_reset:
            if st.button("🔄 Reset Data"):
                st.session_state["working_df"] = st.session_state["uploaded_df"].copy()
                st.rerun()

        
        df_working = st.session_state["working_df"].copy()
        
        st.subheader("Select Operations")
        tab1, tab2 = st.tabs(["Data Cleaning", "Math Operations"])
        
        with tab1:
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.checkbox("Remove Duplicates"):
                    st.write("**Remove Duplicates Configuration**")
                    dup_cols = st.multiselect("Columns to check (empty = all):", df.columns.tolist(), key="dup_cols")
                    dup_keep = st.selectbox("Keep:", ["first", "last", "none"], key="dup_keep")
                    if st.button("Apply Remove Duplicates", key="btn_dup"):
                        engine = DataCleaningEngine()
                        df_working, msg = engine.remove_duplicates(df_working, dup_cols if dup_cols else None, dup_keep)
                        st.session_state["working_df"] = df_working
                        st.success(msg)
                        log_operation("remove_duplicates", msg, {"columns": dup_cols if dup_cols else None, "keep": dup_keep})
                    st.divider()
                
                if st.checkbox("Replace Values"):
                    st.write("**Replace Values Configuration**")
                    r_col = st.selectbox("Column:", df.columns, key="r_col")
                    r_old = st.text_input("Old value:", key="r_old")
                    r_new = st.text_input("New value:", key="r_new")
                    if st.button("Apply Replace", key="btn_replace"):
                        engine = DataCleaningEngine()
                        df_working, msg = engine.replace_values(df_working, r_col, r_old, r_new)
                        st.session_state["working_df"] = df_working
                        st.success(msg)
                        log_operation("replace_values", msg, {"column": r_col, "old_value": r_old, "new_value": r_new})
                    st.divider()
                
                if st.checkbox("Merge Columns"):
                    st.write("**Merge Columns Configuration**")
                    m_cols = st.multiselect("Columns to merge:", df.columns.tolist(), key="m_cols")
                    m_name = st.text_input("New column name:", "merged", key="m_name")
                    m_sep = st.text_input("Separator:", " ", key="m_sep")
                    if st.button("Apply Merge", key="btn_merge"):
                        engine = DataCleaningEngine()
                        df_working, msg = engine.merge_columns(df_working, m_cols, m_name, m_sep)
                        st.session_state["working_df"] = df_working
                        st.success(msg)
                        log_operation("merge_columns", msg, {"columns": m_cols, "new_col_name": m_name, "separator": m_sep})
            
            with col2:
                if st.checkbox("Filter Rows"):
                    st.write("**Filter Rows Configuration**")
                    f_col = st.selectbox("Column:", df.columns, key="f_col")
                    f_op = st.selectbox("Operator:", ["==", "!=", ">", "<", ">=", "<=", "contains"], key="f_op")
                    f_val = st.text_input("Value:", key="f_val")
                    if st.button("Apply Filter", key="btn_filter"):
                        engine = DataCleaningEngine()
                        df_working, msg = engine.filter_rows(df_working, f_col, f_op, f_val)
                        st.session_state["working_df"] = df_working
                        st.success(msg)
                        log_operation("filter_rows", msg, {"column": f_col, "operator": f_op, "value": f_val})
                    st.divider()
                
                if st.checkbox("Normalize Text"):
                    st.write("**Normalize Text Configuration**")
                    n_col = st.selectbox("Column:", df.columns, key="n_col")
                    n_lower = st.checkbox("Convert to lowercase", True, key="n_lower")
                    n_trim = st.checkbox("Trim spaces", True, key="n_trim")
                    if st.button("Apply Normalize", key="btn_normalize"):
                        engine = DataCleaningEngine()
                        df_working, msg = engine.normalize_text(df_working, n_col, n_lower, n_trim)
                        st.session_state["working_df"] = df_working
                        st.success(msg)
                        log_operation("normalize_text", msg, {"column": n_col, "to_lower": n_lower, "trim_spaces": n_trim})
                    st.divider()
                
                if st.checkbox("Handle Missing Values"):
                    st.write("**Missing Values Configuration**")
                    miss_strategy = st.selectbox("Strategy:", ["drop", "fill_mean", "fill_median"], key="miss_strategy")
                    if st.button("Apply Missing Handler", key="btn_missing"):
                        engine = DataCleaningEngine()
                        df_working, msg = engine.handle_missing_values(df_working, miss_strategy)
                        st.session_state["working_df"] = df_working
                        st.success(msg)
                        log_operation("handle_missing", msg, {"strategy": miss_strategy, "fill_value": None})
        
        with tab2:
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.checkbox("Arithmetic Operation"):
                    st.write("**Arithmetic Configuration**")
                    a_col1 = st.selectbox("Column 1:", df.columns, key="a_col1")
                    a_col2 = st.selectbox("Column 2:", df.columns, key="a_col2")
                    a_op = st.selectbox("Operation:", ["+", "-", "*", "/"], key="a_op")
                    a_new = st.text_input("Result column name:", "result", key="a_new")
                    if st.button("Apply Arithmetic", key="btn_arith"):
                        engine = MathOperationsEngine()
                        df_working, msg = engine.arithmetic_operation(df_working, a_col1, a_col2, a_op, a_new)
                        st.session_state["working_df"] = df_working
                        st.success(msg)
                        log_operation("arithmetic", msg, {"col1": a_col1, "col2": a_col2, "operation": a_op, "result_col": a_new})
                    st.divider()
                
                if st.checkbox("Percentage Change"):
                    st.write("**Percentage Change Configuration**")
                    pc_old = st.selectbox("Old value column:", df.columns, key="pc_old")
                    pc_new = st.selectbox("New value column:", df.columns, key="pc_new")
                    pc_result = st.text_input("Result column:", "pct_change", key="pc_result")
                    if st.button("Calculate Percentage Change", key="btn_pc"):
                        engine = MathOperationsEngine()
                        df_working, msg = engine.percentage_change(df_working, pc_old, pc_new, pc_result)
                        st.session_state["working_df"] = df_working
                        st.success(msg)
                        log_operation("percentage_change", msg, {"old_col": pc_old, "new_col": pc_new, "result_col": pc_result})
            
            with col2:
                if st.checkbox("Weighted Average"):
                    st.write("**Weighted Average Configuration**")
                    wa_cols = st.multiselect("Columns:", df.columns, key="wa_cols")
                    wa_weights_str = st.text_input("Weights (comma-separated):", "1,1,1", key="wa_weights")
                    wa_result = st.text_input("Result column:", "weighted_avg", key="wa_result")
                    if st.button("Calculate Weighted Average", key="btn_wa"):
                        try:
                            wa_weights = [float(x.strip()) for x in wa_weights_str.split(",")]
                            engine = MathOperationsEngine()
                            df_working, msg = engine.weighted_average(df_working, wa_cols, wa_weights, wa_result)
                            st.session_state["working_df"] = df_working
                            st.success(msg)
                            log_operation("weighted_average", msg, {"value_cols": wa_cols, "weights": wa_weights, "result_col": wa_result})
                        except ValueError:
                            st.error("Invalid weight format")
                    st.divider()
                
                if st.checkbox("Aggregate Function"):
                    st.write("**Aggregate Configuration**")
                    agg_cols = st.multiselect("Columns:", df.columns, key="agg_cols")
                    agg_func = st.selectbox("Function:", ["sum", "mean", "median", "min", "max"], key="agg_func")
                    if st.button("Apply Aggregate", key="btn_agg"):
                        engine = MathOperationsEngine()
                        result, msg = engine.aggregate_function(df_working, agg_cols, agg_func)
                        df_working = result if isinstance(result, pd.DataFrame) else df_working
                        st.session_state["working_df"] = df_working
                        st.success(msg)
                        st.write(result)
                        log_operation("aggregate", msg, {"columns": agg_cols, "function": agg_func})
        
        st.divider()
        st.subheader("Preview & Export")
        col1, col2 = st.columns([2, 1])
        with col1:
            st.write("**Transformed Data Preview**")
            preview_rows = st.slider("Rows to show:", 5, min(100, len(df_working)), 10)
            st.dataframe(df_working.head(preview_rows), use_container_width=True)
        with col2:
            st.write("**Statistics**")
            st.metric("Total Rows", len(df_working))
            st.metric("Total Columns", len(df_working.columns))
            st.metric("Changes", len(df_working.columns) - len(df.columns) + (len(df) - len(df_working)))
        
        st.subheader("⬇ Download Results")
        col1, col2 = st.columns(2)
        with col1:
            excel_bytes = dataframe_to_excel_bytes(df_working)
            st.download_button(
                label="Download as Excel",
                data=excel_bytes,
                file_name=f"transformed_{st.session_state['uploaded_filename']}",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        with col2:
            csv_bytes = df_working.to_csv(index=False).encode()
            st.download_button(
                label="Download as CSV",
                data=csv_bytes,
                file_name=f"transformed_{st.session_state['uploaded_filename'].rsplit('.', 1)[0]}.csv",
                mime="text/csv"
            )
        
        st.session_state["preview_df"] = df_working
    
    # ========================================================================
    # PAGE: TEMPLATES 
    # ========================================================================
    elif page == "💾 Templates":
        st.header("Transformation Templates")
        st.markdown("Save and reuse your transformation workflows **across different Excel files**")
        
        templates = load_saved_templates()
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.subheader("Save Current Operations")
            template_name = st.text_input("Template name:", key="tpl_name")
            template_desc = st.text_area("Description:", key="tpl_desc")
            
            if st.button("💾 Save Template", key="btn_save_tpl"):
                if template_name:
                    if save_current_operations_as_template(template_name, template_desc):
                        st.success(f"Template '{template_name}' saved!")
                        st.rerun()
                    else:
                        st.error("Failed to save template")
                else:
                    st.error("Please enter a template name")
        
        with col2:
            st.subheader("Saved Templates")
            
            if templates:
                for tpl_name, tpl in templates.items():
                    col_a, col_b, col_c = st.columns([2, 1, 1])
                    with col_a:
                        st.write(f"**{tpl_name}**")
                        st.caption(f"{tpl.get('description', 'No description')}")
                        st.caption(f"{tpl.get('operation_count', 0)} operations")
                        st.caption(f"{tpl.get('created_at', 'Unknown')[:10]}")
                    
                    with col_b:
                        if st.button("▶ Apply", key=f"apply_{tpl_name}", use_container_width=True):
                            st.session_state["template_to_apply"] = tpl_name
                            st.rerun()
                    
                    with col_c:
                        if st.button("Delete", key=f"delete_{tpl_name}", use_container_width=True):
                            del templates[tpl_name]
                            save_templates(templates)
                            st.success(f"Template '{tpl_name}' deleted!")
                            st.rerun()
                    
                    st.divider()
            else:
                st.info("No templates saved yet. Perform some operations and save them as a template!")
        
        # ====================== COLUMN MAPPING INTERFACE ======================
        if st.session_state.get("template_to_apply"):
            tpl_name = st.session_state["template_to_apply"]
            if tpl_name in templates:
                st.subheader(f"Column Mapping for Template: **{tpl_name}**")
                st.info("Match the columns used when the template was created to the columns in your **current** Excel file.")
                
                operations = templates[tpl_name].get("operations", [])
                required_cols = get_template_required_columns(operations)
                
                current_df = st.session_state.get("uploaded_df")
                
                if current_df is None:
                    st.error("Please upload a file first")
                    if st.button("Cancel"):
                        del st.session_state["template_to_apply"]
                        st.rerun()
                elif not required_cols:
                    st.info("This template has no column-specific operations.")
                    if st.button(" Apply Template", type="primary"):
                        df_result, messages = apply_template_to_dataframe(current_df.copy(), tpl_name)

                        st.session_state["preview_df"] = df_result
                        st.session_state["last_template_applied"] = tpl_name
                        st.session_state["show_template_download"] = True

                        st.success(f"Template '{tpl_name}' applied!")
                        for msg in messages:
                            st.info(msg)

                        del st.session_state["template_to_apply"]
                        st.rerun()
                else:
                    st.write("**Map original columns → current columns**")
                    mapping = {}
                    for old_col in required_cols:
                        options = ["(keep original name)"] + list(current_df.columns)
                        default_idx = 0
                        if old_col in current_df.columns:
                            default_idx = options.index(old_col) if old_col in options else 0
                        
                        selected = st.selectbox(
                            f"**{old_col}** →",
                            options=options,
                            index=default_idx,
                            key=f"map_{tpl_name}_{old_col}"
                        )
                        mapping[old_col] = old_col if selected == "(keep original name)" else selected
                    
                    col_btn1, col_btn2 = st.columns(2)
                    with col_btn1:
                        if st.button("Confirm & Apply Template", type="primary", key="confirm_mapping"):
                            df_result, messages = apply_template_to_dataframe(
                                current_df.copy(), tpl_name, column_mapping=mapping
                            )

                            st.session_state["preview_df"] = df_result
                            st.session_state["last_template_applied"] = tpl_name
                            st.session_state["show_template_download"] = True

                            st.success(f"Template '{tpl_name}' applied.")
                            for msg in messages:
                                st.info(msg)

                            del st.session_state["template_to_apply"]
                            st.rerun()
                    with col_btn2:
                        if st.button("Cancel Mapping", key="cancel_mapping"):
                            del st.session_state["template_to_apply"]
                            st.rerun()
            else:
                del st.session_state["template_to_apply"]
                st.rerun()
        # ---------------- TEMPLATE OUTPUT: PREVIEW + DOWNLOAD ----------------
        if st.session_state.get("show_template_download") and st.session_state.get("preview_df") is not None:
            st.divider()
            tpl_used = st.session_state.get("last_template_applied", "template")
            st.subheader(f"Template Output ({tpl_used})")

            out_df = st.session_state["preview_df"]
            st.dataframe(out_df.head(20), use_container_width=True)

            original_name = st.session_state.get("uploaded_filename") or "uploaded.xlsx"
            safe_tpl = str(tpl_used).replace(" ", "_")
            base_filename = f"transformed_{safe_tpl}_{original_name}"

            render_download_buttons(out_df, base_filename)
    # ========================================================================
    # PAGE: LOGS 
    # ========================================================================
    elif page == "📋 Logs":
        st.header("Operation Logs")
        logs = st.session_state.get("operation_log", [])
        if logs:
            log_df = pd.DataFrame(logs)
            st.dataframe(log_df, use_container_width=True)
            if st.button("Clear Logs"):
                st.session_state["operation_log"] = []
                st.session_state["current_operations"] = []
                st.rerun()
        else:
            st.info("No operations logged yet")

if __name__ == "__main__":
    main()
