import pandas as pd
import numpy as np
from typing import List, Optional, Tuple

# ============================================================================
# DATA CLEANING OPERATIONS MODULE
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
