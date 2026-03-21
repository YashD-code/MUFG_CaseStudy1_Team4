from typing import List, Dict, Optional, Tuple
import pandas as pd
import copy

from data_cleaning import DataCleaningEngine
from math_operations import MathOperationsEngine
from utils import load_saved_templates


# ============================================================================
# TEMPLATE COLUMN MAPPING HELPERS
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
    
    # Apply column mapping if provided 
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
