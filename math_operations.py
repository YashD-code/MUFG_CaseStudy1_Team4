import pandas as pd
import numpy as np
from typing import List, Tuple

# ============================================================================
# MATHEMATICAL OPERATIONS MODULE
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
            numeric_data = df[valid_cols].apply(pd.to_numeric, errors="coerce")
            result = getattr(numeric_data, func)()
            msg = f"✓ Applied {func} aggregate function"
            return result, msg
        except Exception as e:
            return df, f"✗ Error in aggregate function: {str(e)}"
