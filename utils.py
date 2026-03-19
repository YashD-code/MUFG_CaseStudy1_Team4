import streamlit as st
import json
import os
from io import BytesIO
from datetime import datetime
from typing import Dict
import pandas as pd

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
        "show_template_download": False,   
        "working_df": None,
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
