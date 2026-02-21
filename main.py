
"""
================================================================================
EXCEL DATA TRANSFORMATION TOOL - MAIN APPLICATION (UPDATED)
================================================================================
ENHANCED WITH COLUMN MAPPING FOR TEMPLATE REUSE

- Templates now work across Excel files with DIFFERENT column names
- When applying a template, you will see an interactive column mapping screen
- Map original template columns → your current file's columns
- Full backward compatibility (if column names match exactly, mapping is automatic)
- All existing functionality preserved

Project: Building an Excel Data Massaging Tool
================================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import copy
from io import BytesIO
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import plotly.express as px

# ============================================================================
# HELPER FUNCTIONS & IMPORTS
# ============================================================================

def inject_css() -> None:
    """Inject custom CSS styling for professional UI appearance.
    
    This function applies custom CSS to modify the appearance of elements in the Streamlit app, such as buttons and headers.
    """
    css = """
    <style>
    .stButton>button { width: 100%; }
    .mapping-header { font-size: 1.1rem; font-weight: 600; margin: 1rem 0 0.5rem 0; }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

def initialize_session_state() -> None:
    """Initialize Streamlit session state variables on app startup.
    
    This function sets up default session state variables to hold the uploaded file, transformations, preview data,
    templates, and other relevant application states for the tool.
    """
    defaults = {
        "uploaded_df": None,
        "uploaded_filename": None,
        "sheet_names": [],
        "current_sheet": None,
        "transformations": {},
        "preview_df": None,
        "processing_stats": {},
        "saved_templates": {},
        "operation_log": [],
        "current_operations": [],  
        "template_to_apply": None,
        "last_template_applied": None,
        "show_template_download": False,   # NEW: For column mapping flow
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def load_saved_templates() -> Dict:
    """Load transformation templates from JSON file.
    
    This function attempts to load previously saved templates from a local 'templates.json' file. 
    If loading fails or the file doesn't exist, it returns an empty dictionary.
    
    Returns:
        dict: The loaded templates or an empty dictionary if no templates are found.
    """
    try:
        if os.path.exists("templates.json"):
            with open("templates.json", "r") as f:
                return json.load(f)
    except Exception as e:
        st.warning(f"Could not load templates: {e}")
    return {}

def save_templates(templates: Dict) -> None:
    """Save transformation templates to JSON file.
    
    This function saves the provided templates dictionary to the 'templates.json' file, ensuring persistence across sessions.
    
    Args:
        templates (dict): A dictionary containing transformation templates to be saved.
    """
    try:
        with open("templates.json", "w") as f:
            json.dump(templates, f, indent=2)
    except Exception as e:
        st.error(f"Failed to save templates: {e}")

def dataframe_to_excel_bytes(df: pd.DataFrame) -> bytes:
    """Convert DataFrame to Excel bytes for download.
    
    This function converts a given DataFrame into Excel file format (as bytes) that can be downloaded by the user.
    
    Args:
        df (pd.DataFrame): The DataFrame to be converted.
    
    Returns:
        bytes: Excel file as bytes.
    """
    buffer = BytesIO()
    df.to_excel(buffer, index=False, engine="openpyxl")
    buffer.seek(0)
    return buffer.getvalue()

def render_download_buttons(df: pd.DataFrame, base_filename: str) -> None:
    """Render Excel/CSV download buttons for a given dataframe.
    
    This function creates download buttons in the Streamlit app, allowing users to download the transformed data in both
    Excel and CSV formats.
    
    Args:
        df (pd.DataFrame): The DataFrame to be downloaded.
        base_filename (str): The base filename for the downloads.
    """
    st.subheader("⬇️ Download Results")

    col1, col2 = st.columns(2)

    # Excel
    with col1:
        excel_bytes = dataframe_to_excel_bytes(df)
        st.download_button(
            label="📥 Download as Excel",
            data=excel_bytes,
            file_name=base_filename if base_filename.lower().endswith(".xlsx") else f"{base_filename}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    # CSV
    with col2:
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        csv_name = base_filename.rsplit(".", 1)[0] + ".csv"
        st.download_button(
            label="📥 Download as CSV",
            data=csv_bytes,
            file_name=csv_name,
            mime="text/csv"
        )
def log_operation(operation: str, details: str = "", parameters: Dict = None) -> None:
    """Log performed operations to session state with parameters.
    
    This function logs the operation details, timestamp, and any parameters for auditing purposes.
    
    Args:
        operation (str): The name of the operation performed.
        details (str): Additional details about the operation.
        parameters (dict, optional): Parameters passed to the operation, defaults to None.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    op_entry = {
        "timestamp": timestamp,
        "operation": operation,
        "details": details,
        "parameters": parameters or {}
    }
    
    st.session_state["operation_log"].append(op_entry)
    st.session_state["current_operations"].append(op_entry)

def save_current_operations_as_template(template_name: str, description: str = "") -> bool:
    """Save current operation sequence as a reusable template.
    
    This function saves the sequence of transformations as a template that can be applied later.
    
    Args:
        template_name (str): The name of the template.
        description (str, optional): Description of the template, defaults to "".
    
    Returns:
        bool: True if template was saved successfully, False otherwise.
    """
    if not st.session_state["current_operations"]:
        st.error("No operations to save. Please perform some operations first.")
        return False
    
    templates = load_saved_templates()
    
    templates[template_name] = {
        "created_at": datetime.now().isoformat(),
        "description": description,
        "operations": st.session_state["current_operations"],
        "operation_count": len(st.session_state["current_operations"])
    }
    
    save_templates(templates)
    st.session_state["saved_templates"] = templates
    return True

# ============================================================================
# NEW: TEMPLATE COLUMN MAPPING HELPERS (FOR REUSABILITY ACROSS DIFFERENT FILES)
# ============================================================================

def get_template_required_columns(operations: List[Dict]) -> List[str]:
    """Extract all column names referenced in the template operations.
    
    This function extracts column names required for a given set of operations in the template. This is useful for mapping
    template columns to those in the current dataset.
    
    Args:
        operations (list): A list of operations defined in the template.
    
    Returns:
        list: A sorted list of column names referenced in the operations.
    """
    required = set()
    for op in operations:
        params = op.get("parameters", {})
        op_name = op.get("operation")
        
        if op_name == "remove_duplicates":
            cols = params.get("columns") or []
            if isinstance(cols, (list, tuple)):
                required.update(c for c in cols if c)
        elif op_name in ["filter_rows", "replace_values", "normalize_text"]:
            col = params.get("column")
            if col:
                required.add(col)
        elif op_name == "merge_columns":
            cols = params.get("columns", [])
            if isinstance(cols, (list, tuple)):
                required.update(c for c in cols if c)
        elif op_name == "arithmetic":
            if params.get("col1"): required.add(params.get("col1"))
            if params.get("col2"): required.add(params.get("col2"))
        elif op_name == "percentage_change":
            if params.get("old_col"): required.add(params.get("old_col"))
            if params.get("new_col"): required.add(params.get("new_col"))
        elif op_name == "weighted_average":
            cols = params.get("value_cols", [])
            if isinstance(cols, (list, tuple)):
                required.update(c for c in cols if c)
        # handle_missing, aggregate etc. do not require specific column mapping
    return sorted(list(required))

def remap_operations_columns(operations: List[Dict], column_mapping: Dict[str, str]) -> List[Dict]:
    """Return a deep copy of operations with column names replaced according to mapping.
    
    This function updates the operation sequence to reflect the column name changes based on the provided mapping.
    
    Args:
        operations (list): The list of operations to remap.
        column_mapping (dict): Mapping of old column names to new column names.
    
    Returns:
        list: The modified operations with remapped column names.
    """
    if not column_mapping:
        return operations
    remapped = copy.deepcopy(operations)
    for op in remapped:
        params = op.get("parameters", {})
        op_name = op.get("operation")
        
        if op_name == "remove_duplicates":
            cols = params.get("columns")
            if isinstance(cols, list):
                params["columns"] = [column_mapping.get(c, c) for c in cols]
        elif op_name in ["filter_rows", "replace_values", "normalize_text"]:
            col = params.get("column")
            if col:
                params["column"] = column_mapping.get(col, col)
        elif op_name == "merge_columns":
            cols = params.get("columns", [])
            if isinstance(cols, list):
                params["columns"] = [column_mapping.get(c, c) for c in cols]
        elif op_name == "arithmetic":
            if "col1" in params:
                params["col1"] = column_mapping.get(params["col1"], params["col1"])
            if "col2" in params:
                params["col2"] = column_mapping.get(params["col2"], params["col2"])
        elif op_name == "percentage_change":
            if "old_col" in params:
                params["old_col"] = column_mapping.get(params["old_col"], params["old_col"])
            if "new_col" in params:
                params["new_col"] = column_mapping.get(params["new_col"], params["new_col"])
        elif op_name == "weighted_average":
            cols = params.get("value_cols", [])
            if isinstance(cols, list):
                params["value_cols"] = [column_mapping.get(c, c) for c in cols]
    return remapped

def apply_template_to_dataframe(df: pd.DataFrame, template_name: str, 
                               column_mapping: Optional[Dict[str, str]] = None) -> Tuple[pd.DataFrame, List[str]]:
    """Apply a saved template to a DataFrame (with optional column mapping for different files).
    
    This function applies the operations defined in a saved template to a given DataFrame. 
    It also supports column mapping for reusability across different files with different column names.
    
    Args:
        df (pd.DataFrame): The DataFrame to apply the template on.
        template_name (str): The name of the template to apply.
        column_mapping (dict, optional): Mapping of old column names to new ones, defaults to None.
    
    Returns:
        Tuple: A tuple containing the transformed DataFrame and a list of messages about the transformations applied.
    """
    templates = load_saved_templates()
    
    if template_name not in templates:
        return df, [f"✗ Template '{template_name}' not found"]
    
    template = templates[template_name]
    operations = template.get("operations", [])
    
    # Apply column mapping if provided (makes templates reusable across files)
    if column_mapping:
        operations = remap_operations_columns(operations, column_mapping)
    
    messages = []
    
    for op in operations:
        op_name = op.get("operation")
        op_params = op.get("parameters", {})
        
        if op_name == "remove_duplicates":
            engine = DataCleaningEngine()
            df, msg = engine.remove_duplicates(
                df,
                op_params.get("columns"),
                op_params.get("keep", "first")
            )
            messages.append(msg)
        elif op_name == "filter_rows":
            engine = DataCleaningEngine()
            df, msg = engine.filter_rows(
                df,
                op_params.get("column"),
                op_params.get("operator"),
                op_params.get("value")
            )
            messages.append(msg)
        elif op_name == "replace_values":
            engine = DataCleaningEngine()
            df, msg = engine.replace_values(
                df,
                op_params.get("column"),
                op_params.get("old_value"),
                op_params.get("new_value")
            )
            messages.append(msg)
        elif op_name == "merge_columns":
            engine = DataCleaningEngine()
            df, msg = engine.merge_columns(
                df,
                op_params.get("columns", []),
                op_params.get("new_col_name", "merged"),
                op_params.get("separator", " ")
            )
            messages.append(msg)
        elif op_name == "normalize_text":
            engine = DataCleaningEngine()
            df, msg = engine.normalize_text(
                df,
                op_params.get("column"),
                op_params.get("to_lower", True),
                op_params.get("trim_spaces", True)
            )
            messages.append(msg)
        elif op_name == "handle_missing":
            engine = DataCleaningEngine()
            df, msg = engine.handle_missing_values(
                df,
                op_params.get("strategy", "drop"),
                op_params.get("fill_value")
            )
            messages.append(msg)
        elif op_name == "arithmetic":
            engine = MathOperationsEngine()
            df, msg = engine.arithmetic_operation(
                df,
                op_params.get("col1"),
                op_params.get("col2"),
                op_params.get("operation"),
                op_params.get("result_col", "result")
            )
            messages.append(msg)
        elif op_name == "percentage_change":
            engine = MathOperationsEngine()
            df, msg = engine.percentage_change(
                df,
                op_params.get("old_col"),
                op_params.get("new_col"),
                op_params.get("result_col", "pct_change")
            )
            messages.append(msg)
        elif op_name == "weighted_average":
            engine = MathOperationsEngine()
            df, msg = engine.weighted_average(
                df,
                op_params.get("value_cols", []),
                op_params.get("weights", []),
                op_params.get("result_col", "weighted_avg")
            )
            messages.append(msg)
    
    return df, messages

# ============================================================================
# DATA CLEANING OPERATIONS MODULE (unchanged)
# ============================================================================

class DataCleaningEngine:
    """Core data cleaning operations with validation and error handling."""
    
    @staticmethod
    def remove_duplicates(df: pd.DataFrame, columns: Optional[List[str]] = None,
                         keep: str = "first") -> Tuple[pd.DataFrame, str]:
        """Remove duplicate rows from the DataFrame based on the specified columns.
        
        Args:
            df (pd.DataFrame): The DataFrame from which duplicates will be removed.
            columns (list, optional): List of columns to check for duplicates, defaults to None.
            keep (str): Strategy to keep duplicates, can be 'first', 'last', or 'none'.
        
        Returns:
            Tuple: The cleaned DataFrame and a message about the operation performed.
        """
        try:
            initial_rows = len(df)
            subset = [c for c in columns if c in df.columns] if columns else None
            keep_val = False if keep == "none" else keep
            df_clean = df.drop_duplicates(subset=subset, keep=keep_val).reset_index(drop=True)
            removed = initial_rows - len(df_clean)
            msg = f"✓ Removed {removed} duplicate row(s). {len(df_clean)} rows remaining."
            return df_clean, msg
        except Exception as e:
            return df, f"✗ Error removing duplicates: {str(e)}"
    
    @staticmethod
    def filter_rows(df: pd.DataFrame, column: str, operator: str, value: str) -> Tuple[pd.DataFrame, str]:
        """Filter rows in the DataFrame based on a condition applied to a specified column.
        
        This function filters the rows in the DataFrame according to the given operator and value for a specified column.
        Supports various comparison operators like '==', '!=', '>', '<', and 'contains'.
        
        Args:
            df (pd.DataFrame): The DataFrame to be filtered.
            column (str): The column to apply the filter on.
            operator (str): The comparison operator ('==', '!=', '>', '<', '>=', '<=', or 'contains').
            value (str): The value to compare the column's data against.
            
        Returns:
            Tuple: The filtered DataFrame and a message about the operation performed.
        """
        try:
            if column not in df.columns:
                return df, f"✗ Column '{column}' not found"
            initial_rows = len(df)
            series = df[column]
            
            if operator == "contains":
                df_filtered = df[series.astype(str).str.contains(str(value), na=False, case=False)]
            elif operator == "==":
                df_filtered = df[series == value]
            elif operator == "!=":
                df_filtered = df[series != value]
            elif operator == ">":
                df_filtered = df[pd.to_numeric(series, errors="coerce") > float(value)]
            elif operator == "<":
                df_filtered = df[pd.to_numeric(series, errors="coerce") < float(value)]
            elif operator == ">=":
                df_filtered = df[pd.to_numeric(series, errors="coerce") >= float(value)]
            elif operator == "<=":
                df_filtered = df[pd.to_numeric(series, errors="coerce") <= float(value)]
            else:
                return df, f"✗ Unknown operator: {operator}"
            
            filtered = initial_rows - len(df_filtered)
            msg = f"✓ Filtered {filtered} row(s). {len(df_filtered)} rows remaining."
            return df_filtered, msg
        except Exception as e:
            return df, f"✗ Error filtering rows: {str(e)}"
    
    @staticmethod
    def replace_values(df: pd.DataFrame, column: str, old_value: str, new_value: str) -> Tuple[pd.DataFrame, str]:
        """Replace specific values in a DataFrame column with a new value.
        
        This function replaces occurrences of a specified old value in a given column with a new value.
        
        Args:
            df (pd.DataFrame): The DataFrame in which values will be replaced.
            column (str): The column in which the value replacement will occur.
            old_value (str): The value to be replaced.
            new_value (str): The value to replace the old value with.
            
        Returns:
            Tuple: The modified DataFrame and a message about the operation performed.
        """        
        try:
            if column not in df.columns:
                return df, f"✗ Column '{column}' not found"
            df_modified = df.copy()
            occurrences = (df_modified[column] == old_value).sum()
            df_modified[column] = df_modified[column].replace(old_value, new_value)
            msg = f"✓ Replaced {occurrences} occurrence(s) in '{column}'."
            return df_modified, msg
        except Exception as e:
            return df, f"✗ Error replacing values: {str(e)}"
    
    @staticmethod
    def merge_columns(df: pd.DataFrame, columns: List[str], new_col_name: str,
                     separator: str = " ") -> Tuple[pd.DataFrame, str]:
        """Merge multiple columns into a single new column in the DataFrame.
    
    This function combines the values of specified columns into a single column, with a separator between values.
    
    Args:
        df (pd.DataFrame): The DataFrame in which columns will be merged.
        columns (List[str]): List of columns to be merged.
        new_col_name (str): The name of the new column that will hold the merged values.
        separator (str): The separator to be used between the column values.
        
    Returns:
        Tuple: The modified DataFrame and a message about the operation performed.
    """
        try:
            valid_cols = [c for c in columns if c in df.columns]
            if not valid_cols:
                return df, f"✗ No valid columns found"
            df_modified = df.copy()
            df_modified[new_col_name] = df_modified[valid_cols].astype(str).agg(separator.join, axis=1)
            msg = f"✓ Merged {len(valid_cols)} column(s) into '{new_col_name}'."
            return df_modified, msg
        except Exception as e:
            return df, f"✗ Error merging columns: {str(e)}"
    
    @staticmethod
    def convert_date_format(df: pd.DataFrame, column: str, output_format: str = "%Y-%m-%d") -> Tuple[pd.DataFrame, str]:
        """Convert date values in a column to a specified format.
    
    This function converts the date format of a specified column to a new format.
    
    Args:
        df (pd.DataFrame): The DataFrame containing the column to be transformed.
        column (str): The column containing date values to be converted.
        output_format (str): The desired date format, default is "%Y-%m-%d".
        
    Returns:
        Tuple: The modified DataFrame and a message about the operation performed.
    """
        try:
            if column not in df.columns:
                return df, f"✗ Column '{column}' not found"
            df_modified = df.copy()
            df_modified[column] = pd.to_datetime(df_modified[column], errors="coerce").dt.strftime(output_format)
            msg = f"✓ Converted dates in '{column}' to format: {output_format}"
            return df_modified, msg
        except Exception as e:
            return df, f"✗ Error converting dates: {str(e)}"
    
    @staticmethod
    def normalize_text(df: pd.DataFrame, column: str, to_lower: bool = True,
                      trim_spaces: bool = True) -> Tuple[pd.DataFrame, str]:
        """Normalize text in a specified column by converting to lowercase and/or trimming spaces.
    
    This function allows normalizing text data by converting it to lowercase and trimming spaces.
    
    Args:
        df (pd.DataFrame): The DataFrame containing the column to be normalized.
        column (str): The column whose text needs to be normalized.
        to_lower (bool): Whether to convert text to lowercase, default is True.
        trim_spaces (bool): Whether to remove leading/trailing spaces, default is True.
        
    Returns:
        Tuple: The modified DataFrame and a message about the operation performed.
    """
        try:
            if column not in df.columns:
                return df, f"✗ Column '{column}' not found"
            df_modified = df.copy()
            s = df_modified[column].astype(str)
            if trim_spaces:
                s = s.str.strip()
            if to_lower:
                s = s.str.lower()
            df_modified[column] = s
            msg = f"✓ Normalized text in '{column}' (lower={to_lower}, trim={trim_spaces})."
            return df_modified, msg
        except Exception as e:
            return df, f"✗ Error normalizing text: {str(e)}"
    
    @staticmethod
    def handle_missing_values(df: pd.DataFrame, strategy: str = "drop",
                             fill_value: Optional[str] = None) -> Tuple[pd.DataFrame, str]:
        """Handle missing values in the DataFrame by dropping or filling them.
    
    This function handles missing values in the DataFrame by either dropping rows or filling missing values 
    using different strategies, such as mean, median, or a user-specified value.
    
    Args:
        df (pd.DataFrame): The DataFrame with missing values to be handled.
        strategy (str): The strategy to handle missing values ('drop', 'fill_mean', 'fill_median', 'fill_value').
        fill_value (str, optional): The value to fill missing values with, used if 'fill_value' strategy is selected.
        
    Returns:
        Tuple: The modified DataFrame and a message about the operation performed.
    """
        try:
            df_modified = df.copy()
            initial_missing = df_modified.isna().sum().sum()
            
            if strategy == "drop":
                df_modified = df_modified.dropna().reset_index(drop=True)
            elif strategy == "fill_mean":
                numeric_cols = df_modified.select_dtypes(include=[np.number]).columns
                for col in numeric_cols:
                    df_modified[col].fillna(df_modified[col].mean(), inplace=True)
            elif strategy == "fill_median":
                numeric_cols = df_modified.select_dtypes(include=[np.number]).columns
                for col in numeric_cols:
                    df_modified[col].fillna(df_modified[col].median(), inplace=True)
            elif strategy == "fill_value" and fill_value:
                df_modified = df_modified.fillna(fill_value)
            
            remaining_missing = df_modified.isna().sum().sum()
            msg = f"✓ Handled missing values ({initial_missing} → {remaining_missing})."
            return df_modified, msg
        except Exception as e:
            return df, f"✗ Error handling missing values: {str(e)}"

# ============================================================================
# MATHEMATICAL OPERATIONS MODULE (unchanged)
# ============================================================================

class MathOperationsEngine:
    """Mathematical operations for numerical data transformation."""
    
    @staticmethod
    def arithmetic_operation(df: pd.DataFrame, col1: str, col2: str,
                            operation: str, new_col: str) -> Tuple[pd.DataFrame, str]:
        """Perform arithmetic operations (addition, subtraction, multiplication, division) on two columns.
        
        This function applies the specified arithmetic operation (addition, subtraction, multiplication, or division) 
        on two columns of the DataFrame and stores the result in a new column.
        
        Args:
            df (pd.DataFrame): The DataFrame on which the operation will be performed.
            col1 (str): The first column to be used in the operation.
            col2 (str): The second column to be used in the operation.
            operation (str): The arithmetic operation to perform ('+', '-', '*', '/').
            new_col (str): The name of the new column to store the result.
        
        Returns:
            Tuple: A tuple containing the modified DataFrame and a message about the operation performed.
        
        Raises:
            ValueError: If columns are not found or the operation is unknown.
        """        
        try:
            if col1 not in df.columns or col2 not in df.columns:
                return df, f"✗ Column(s) not found"
            df_modified = df.copy()
            a = pd.to_numeric(df_modified[col1], errors="coerce")
            b = pd.to_numeric(df_modified[col2], errors="coerce")
            
            if operation == "+":
                df_modified[new_col] = a + b
            elif operation == "-":
                df_modified[new_col] = a - b
            elif operation == "*":
                df_modified[new_col] = a * b
            elif operation == "/":
                df_modified[new_col] = a / b.replace(0, np.nan)
            else:
                return df, f"✗ Unknown operation: {operation}"
            
            msg = f"✓ Created '{new_col}' as {col1} {operation} {col2}"
            return df_modified, msg
        except Exception as e:
            return df, f"✗ Error in arithmetic operation: {str(e)}"
    
    @staticmethod
    def percentage_change(df: pd.DataFrame, old_col: str, new_col: str,
                         result_col: str) -> Tuple[pd.DataFrame, str]:
        """Calculate percentage change between two columns.
    
    This function computes the percentage change between the values in two columns: 
    (new_col - old_col) / old_col * 100. The result is stored in a new column.
    
    Args:
        df (pd.DataFrame): The DataFrame on which the percentage change will be calculated.
        old_col (str): The column representing the old values.
        new_col (str): The column representing the new values.
        result_col (str): The name of the new column to store the percentage change.
    
    Returns:
        Tuple: A tuple containing the modified DataFrame and a message about the operation performed.
    
    Raises:
        ValueError: If the columns are not found in the DataFrame.
    """
        try:
            if old_col not in df.columns or new_col not in df.columns:
                return df, f"✗ Column(s) not found"
            df_modified = df.copy()
            old_vals = pd.to_numeric(df_modified[old_col], errors="coerce")
            new_vals = pd.to_numeric(df_modified[new_col], errors="coerce")
            df_modified[result_col] = ((new_vals - old_vals) / old_vals.replace(0, np.nan)) * 100
            msg = f"✓ Calculated percentage change: ({new_col} - {old_col}) / {old_col} * 100"
            return df_modified, msg
        except Exception as e:
            return df, f"✗ Error calculating percentage change: {str(e)}"
    
    @staticmethod
    def weighted_average(df: pd.DataFrame, value_cols: List[str],
                        weights: List[float], result_col: str) -> Tuple[pd.DataFrame, str]:
        """Calculate the weighted average of specified columns.
    
    This function computes the weighted average of multiple columns in the DataFrame 
    using the provided weights. The result is stored in a new column.
    
    Args:
        df (pd.DataFrame): The DataFrame on which the weighted average will be calculated.
        value_cols (List[str]): List of columns containing the values for which to calculate the weighted average.
        weights (List[float]): List of weights corresponding to each value column.
        result_col (str): The name of the new column to store the weighted average result.
    
    Returns:
        Tuple: A tuple containing the modified DataFrame and a message about the operation performed.
    
    Raises:
        ValueError: If the number of value columns and weights do not match or if the columns are not found.
    """
        try:
            if len(value_cols) != len(weights):
                return df, f"✗ Number of columns ({len(value_cols)}) != weights ({len(weights)})"
            valid_cols = [c for c in value_cols if c in df.columns]
            if not valid_cols:
                return df, f"✗ No valid columns found"
            df_modified = df.copy()
            weight_arr = np.array(weights[:len(valid_cols)])
            df_modified[result_col] = (df_modified[valid_cols] * weight_arr).sum(axis=1) / weight_arr.sum()
            msg = f"✓ Calculated weighted average of {len(valid_cols)} column(s)"
            return df_modified, msg
        except Exception as e:
            return df, f"✗ Error calculating weighted average: {str(e)}"
    
    @staticmethod
    def aggregate_function(df: pd.DataFrame, columns: List[str],
                          func: str) -> Tuple[pd.DataFrame, str]:
        """Apply an aggregate function (sum, mean, median, min, max) to specified columns.
    
    This function applies an aggregate function (sum, mean, median, min, or max) to the specified columns
    in the DataFrame and returns the result.
    
    Args:
        df (pd.DataFrame): The DataFrame on which the aggregation will be performed.
        columns (List[str]): List of columns to apply the aggregate function to.
        func (str): The aggregate function to apply ('sum', 'mean', 'median', 'min', 'max').
    
    Returns:
        Tuple: A tuple containing the result of the aggregation and a message about the operation performed.
    
    Raises:
        ValueError: If unknown function is passed or the columns are not found.
    """
        try:
            valid_cols = [c for c in columns if c in df.columns]
            if not valid_cols:
                return df, f"✗ No valid columns found"
            if func not in ["sum", "mean", "median", "min", "max"]:
                return df, f"✗ Unknown function: {func}"
            numeric_data = pd.to_numeric(df[valid_cols], errors="coerce")
            result = getattr(numeric_data, func)()
            msg = f"✓ Applied {func} aggregate function"
            return result, msg
        except Exception as e:
            return df, f"✗ Error in aggregate function: {str(e)}"

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
        st.title("📊 Excel Data Transformation Tool")
        st.markdown("*Professional data cleaning & mathematical operations for Excel files*")
    with col2:
        st.info(f"Session: {st.session_state.get('current_sheet', 'None')}")
    st.markdown("---")
    
    with st.sidebar:
        st.header("⚙️ Navigation")
        page = st.radio(
            "Select a page:",
            ["🏠 Home", "📤 Upload File", "🔄 Transform Data", "💾 Templates", "📋 Logs"],
            key="page_selector"
        )
    
    # ========================================================================
    # PAGE: HOME
    # ========================================================================
    if page == "🏠 Home":
        st.header("Welcome to the Excel Data Transformation Tool!")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("✨ Key Features")
            st.markdown("""
- **📤 File Upload & Validation**
- **🧹 Data Cleaning**
- **🧮 Math Operations**
- **📊 Real-time Preview**
- **💾 Templates** (now reusable across files with different columns!)
- **📈 Multi-sheet Support**
- **📥 Download Results**
            """)
        with col2:
            st.subheader("🚀 Quick Start")
            st.markdown("""
1. Upload your Excel file
2. Perform transformations
3. Save as template
4. On a new file → Templates → Apply → Map columns → Done!
            """)
        st.info("👉 Start by uploading a file in the '📤 Upload File' section!")
    
    # ========================================================================
    # PAGE: UPLOAD FILE (unchanged)
    # ========================================================================
    elif page == "📤 Upload File":
        st.header("📤 Upload Excel File")
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
                st.error(f"❌ File size {file_size_mb:.2f} MB exceeds limit of {max_file_size} MB")
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
                    st.session_state["current_sheet"] = selected_sheet
                    st.session_state["current_operations"] = []
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1: st.metric("Rows", len(df))
                    with col2: st.metric("Columns", len(df.columns))
                    with col3: st.metric("File Size", f"{file_size_mb:.2f} MB")
                    with col4: st.metric("Sheets", len(sheet_names))
                    
                    st.subheader("📋 Data Preview")
                    preview_rows = st.slider("Rows to preview:", 5, min(100, len(df)), 10)
                    st.dataframe(df.head(preview_rows), use_container_width=True)
                    

                    # Assuming df is the uploaded dataframe
                    st.subheader("📊 Choose Visualization Type")

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
                        if column:
                            fig = px.histogram(df, x=column, title=f"Distribution of {column}")
                            st.plotly_chart(fig)

                    elif visualization_type == "Scatter Plot":
                        # Allow user to select two numeric columns for scatter plot
                        numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()
                        x_column = st.selectbox("Select the X-axis column for the scatter plot", numeric_columns)
                        y_column = st.selectbox("Select the Y-axis column for the scatter plot", numeric_columns)
                        if x_column and y_column:
                            fig = px.scatter(df, x=x_column, y=y_column, title=f"Scatter Plot: {x_column} vs {y_column}")
                            st.plotly_chart(fig)
                    
                    st.subheader("📊 Column Information")
                    col_info = pd.DataFrame({
                        "Column": df.columns,
                        "Type": [str(dtype) for dtype in df.dtypes],
                        "Non-Null": [df[col].notna().sum() for col in df.columns],
                        "Missing": [df[col].isna().sum() for col in df.columns]
                    })
                    st.dataframe(col_info, use_container_width=True)
                    
                    st.success("✅ File loaded successfully! Ready for transformation.")
                except Exception as e:
                    st.error(f"❌ Error reading file: {str(e)}")
        else:
            st.info("👆 Upload an Excel file to get started")
    
    # ========================================================================
    # PAGE: TRANSFORM DATA (unchanged - operations already log parameters)
    # ========================================================================
    elif page == "🔄 Transform Data":
        if st.session_state["uploaded_df"] is None:
            st.warning("⚠️ Please upload a file first!")
            st.stop()
        
        st.header("🔄 Transform Data")
        df = st.session_state["uploaded_df"].copy()
        df_working = df.copy()
        
        st.subheader("Select Operations")
        tab1, tab2 = st.tabs(["🧹 Data Cleaning", "🧮 Math Operations"])
        
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
                        st.success(msg)
                        log_operation("normalize_text", msg, {"column": n_col, "to_lower": n_lower, "trim_spaces": n_trim})
                    st.divider()
                
                if st.checkbox("Handle Missing Values"):
                    st.write("**Missing Values Configuration**")
                    miss_strategy = st.selectbox("Strategy:", ["drop", "fill_mean", "fill_median"], key="miss_strategy")
                    if st.button("Apply Missing Handler", key="btn_missing"):
                        engine = DataCleaningEngine()
                        df_working, msg = engine.handle_missing_values(df_working, miss_strategy)
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
                        st.success(msg)
                        st.write(result)
                        log_operation("aggregate", msg, {"columns": agg_cols, "function": agg_func})
        
        st.divider()
        st.subheader("📊 Preview & Export")
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
        
        st.subheader("⬇️ Download Results")
        col1, col2 = st.columns(2)
        with col1:
            excel_bytes = dataframe_to_excel_bytes(df_working)
            st.download_button(
                label="📥 Download as Excel",
                data=excel_bytes,
                file_name=f"transformed_{st.session_state['uploaded_filename']}",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        with col2:
            csv_bytes = df_working.to_csv(index=False).encode()
            st.download_button(
                label="📥 Download as CSV",
                data=csv_bytes,
                file_name=f"transformed_{st.session_state['uploaded_filename'].rsplit('.', 1)[0]}.csv",
                mime="text/csv"
            )
        
        st.session_state["preview_df"] = df_working
    
    # ========================================================================
    # PAGE: TEMPLATES - NOW WITH COLUMN MAPPING
    # ========================================================================
    elif page == "💾 Templates":
        st.header("💾 Transformation Templates")
        st.markdown("Save and reuse your transformation workflows **across different Excel files**")
        
        templates = load_saved_templates()
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.subheader("📝 Save Current Operations")
            template_name = st.text_input("Template name:", key="tpl_name")
            template_desc = st.text_area("Description:", key="tpl_desc")
            
            if st.button("💾 Save Template", key="btn_save_tpl"):
                if template_name:
                    if save_current_operations_as_template(template_name, template_desc):
                        st.success(f"✅ Template '{template_name}' saved!")
                        st.rerun()
                    else:
                        st.error("Failed to save template")
                else:
                    st.error("Please enter a template name")
        
        with col2:
            st.subheader("📚 Saved Templates")
            
            if templates:
                for tpl_name, tpl in templates.items():
                    col_a, col_b, col_c = st.columns([2, 1, 1])
                    with col_a:
                        st.write(f"**{tpl_name}**")
                        st.caption(f"📝 {tpl.get('description', 'No description')}")
                        st.caption(f"✏️ {tpl.get('operation_count', 0)} operations")
                        st.caption(f"📅 {tpl.get('created_at', 'Unknown')[:10]}")
                    
                    with col_b:
                        if st.button("▶️ Apply", key=f"apply_{tpl_name}", use_container_width=True):
                            st.session_state["template_to_apply"] = tpl_name
                            st.rerun()
                    
                    with col_c:
                        if st.button("🗑️ Delete", key=f"delete_{tpl_name}", use_container_width=True):
                            del templates[tpl_name]
                            save_templates(templates)
                            st.success(f"✅ Template '{tpl_name}' deleted!")
                            st.rerun()
                    
                    st.divider()
            else:
                st.info("No templates saved yet. Perform some operations and save them as a template!")
        
        # ====================== COLUMN MAPPING INTERFACE ======================
        if st.session_state.get("template_to_apply"):
            tpl_name = st.session_state["template_to_apply"]
            if tpl_name in templates:
                st.subheader(f"🔗 Column Mapping for Template: **{tpl_name}**")
                st.info("Match the columns used when the template was created to the columns in your **current** Excel file.")
                
                operations = templates[tpl_name].get("operations", [])
                required_cols = get_template_required_columns(operations)
                
                current_df = st.session_state.get("uploaded_df")
                
                if current_df is None:
                    st.error("⚠️ Please upload a file first (Upload File page)")
                    if st.button("❌ Cancel"):
                        del st.session_state["template_to_apply"]
                        st.rerun()
                elif not required_cols:
                    st.info("This template has no column-specific operations.")
                    if st.button("✅ Apply Template", type="primary"):
                        df_result, messages = apply_template_to_dataframe(current_df.copy(), tpl_name)

                        st.session_state["preview_df"] = df_result
                        st.session_state["last_template_applied"] = tpl_name
                        st.session_state["show_template_download"] = True

                        st.success(f"✅ Template '{tpl_name}' applied!")
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
                        if st.button("✅ Confirm & Apply Template", type="primary", key="confirm_mapping"):
                            df_result, messages = apply_template_to_dataframe(
                                current_df.copy(), tpl_name, column_mapping=mapping
                            )

                            st.session_state["preview_df"] = df_result
                            st.session_state["last_template_applied"] = tpl_name
                            st.session_state["show_template_download"] = True

                            st.success(f"✅ Template '{tpl_name}' applied with column mapping!")
                            for msg in messages:
                                st.info(msg)

                            del st.session_state["template_to_apply"]
                            st.rerun()
                    with col_btn2:
                        if st.button("❌ Cancel Mapping", key="cancel_mapping"):
                            del st.session_state["template_to_apply"]
                            st.rerun()
            else:
                del st.session_state["template_to_apply"]
                st.rerun()
        # ---------------- TEMPLATE OUTPUT: PREVIEW + DOWNLOAD ----------------
        if st.session_state.get("show_template_download") and st.session_state.get("preview_df") is not None:
            st.divider()
            tpl_used = st.session_state.get("last_template_applied", "template")
            st.subheader(f"📦 Template Output ({tpl_used})")

            out_df = st.session_state["preview_df"]
            st.dataframe(out_df.head(20), use_container_width=True)

            original_name = st.session_state.get("uploaded_filename") or "uploaded.xlsx"
            safe_tpl = str(tpl_used).replace(" ", "_")
            base_filename = f"transformed_{safe_tpl}_{original_name}"

            render_download_buttons(out_df, base_filename)
    # ========================================================================
    # PAGE: LOGS (unchanged)
    # ========================================================================
    elif page == "📋 Logs":
        st.header("📋 Operation Logs")
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
