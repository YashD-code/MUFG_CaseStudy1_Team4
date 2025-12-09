# Without GUI
# Author: Trivenee
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any
import pandas as pd
import numpy as np


# Exceptions --------------------------------------------------------

class MathOperationError(Exception):
    """Raised when a mathematical operation fails or invalid inputs are provided."""
    pass


# Data Models -------------------------------------------------------

@dataclass
class MathOperationResult:
    """Result container after performing mathematical operations."""
    df: pd.DataFrame
    message: str


# Utility / Internal Helper Functions ------------------------------

def _validate_numeric_columns(df: pd.DataFrame, cols: List[str]):
    """
    Ensure all selected columns exist and are numeric.
    """
    for col in cols:
        if col not in df.columns:
            raise MathOperationError(f"Column '{col}' not found in DataFrame.")
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise MathOperationError(f"Column '{col}' must be numeric for mathematical operations.")


# Core Mathematical Operations -------------------------------------

def add_columns(df: pd.DataFrame, col1: str, col2: str, new_col: str) -> MathOperationResult:
    """
    Add two numeric columns and create a new column.
    """
    _validate_numeric_columns(df, [col1, col2])
    df[new_col] = df[col1] + df[col2]
    return MathOperationResult(df, f"Created '{new_col}' as {col1} + {col2}")


def subtract_columns(df: pd.DataFrame, col1: str, col2: str, new_col: str) -> MathOperationResult:
    """
    Subtract one numeric column from another.
    """
    _validate_numeric_columns(df, [col1, col2])
    df[new_col] = df[col1] - df[col2]
    return MathOperationResult(df, f"Created '{new_col}' as {col1} - {col2}")


def multiply_columns(df: pd.DataFrame, col1: str, col2: str, new_col: str) -> MathOperationResult:
    """
    Multiply two numeric columns.
    """
    _validate_numeric_columns(df, [col1, col2])
    df[new_col] = df[col1] * df[col2]
    return MathOperationResult(df, f"Created '{new_col}' as {col1} * {col2}")


def divide_columns(df: pd.DataFrame, col1: str, col2: str, new_col: str) -> MathOperationResult:
    """
    Divide col1 by col2 safely (handles division by zero).
    """
    _validate_numeric_columns(df, [col1, col2])

    df[new_col] = df[col1] / df[col2].replace(0, np.nan)
    return MathOperationResult(df, f"Created '{new_col}' as {col1} / {col2}")


def apply_aggregate(df: pd.DataFrame, col: str, func: str, new_col: str) -> MathOperationResult:
    """
    Apply an aggregate function: sum, mean, median, min, max.
    """
    _validate_numeric_columns(df, [col])

    if func not in ["sum", "mean", "median", "min", "max"]:
        raise MathOperationError(f"Unsupported aggregate function '{func}'.")

    value = getattr(df[col], func)()
    df[new_col] = value

    return MathOperationResult(df, f"Applied '{func}' on '{col}' → created '{new_col}'")


def percentage_change(df: pd.DataFrame, old_col: str, new_col: str, out_col: str) -> MathOperationResult:
    """
    Compute percentage change between two numeric columns.
    Formula: ((new - old) / old) * 100
    """
    _validate_numeric_columns(df, [old_col, new_col])

    df[out_col] = ((df[new_col] - df[old_col]) / df[old_col].replace(0, np.nan)) * 100
    return MathOperationResult(df, f"Created '{out_col}' as % change between {old_col} → {new_col}")


def weighted_average(df: pd.DataFrame, columns: List[str], weights: List[float], out_col: str) -> MathOperationResult:
    """
    Compute row-wise weighted average of multiple columns.
    """
    _validate_numeric_columns(df, columns)

    if len(columns) != len(weights):
        raise MathOperationError("Number of columns and weights must match.")

    weight_arr = np.array(weights)

    df[out_col] = (df[columns] * weight_arr).sum(axis=1) / weight_arr.sum()
    return MathOperationResult(df, f"Created '{out_col}' as weighted average of {columns}")


def custom_formula(df: pd.DataFrame, formula: str, out_col: str) -> MathOperationResult:
    """
    Execute a custom formula using pandas eval().
    Example: "A + B * 0.18"
    """
    try:
        df[out_col] = df.eval(formula)
        return MathOperationResult(df, f"Applied custom formula → created '{out_col}'")
    except Exception as e:
        raise MathOperationError(f"Invalid formula '{formula}' → {str(e)}")



""" Inside Streamlit:
from math_operations import (
    add_columns, subtract_columns, multiply_columns, divide_columns,
    apply_aggregate, percentage_change, weighted_average, custom_formula
)"""