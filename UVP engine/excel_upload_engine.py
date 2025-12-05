from __future__ import annotations

import io
import os
from dataclasses import dataclass
from typing import List, Optional
import pandas as pd

# Configuration constants
ALLOWED_EXTENSIONS = {".xlsx"}
MAX_FILE_SIZE_MB = 200           # hard global limit, can still override via parameter
DEFAULT_PREVIEW_ROWS = 10
MAX_SHEETS = 50                  # safety cap
MAX_COLUMNS = 200                # safety cap
LARGE_FILE_THRESHOLD_MB = 50     # above this, we skip full row counting

# Exceptions
class FileValidationError(Exception):
    """Raised when the uploaded file fails basic validation (type/size)."""
    pass

class SheetNotFoundError(Exception):
    """Raised when the requested sheet does not exist."""
    pass

class ExcelReadError(Exception):
    """Raised when the Excel file cannot be parsed."""
    pass

# Data Models
@dataclass
class FileMetadata:
    filename: str
    size_bytes: int
    size_mb: float
    sheet_names: List[str]

@dataclass
class PreviewResult:
    metadata: FileMetadata
    selected_sheet: str
    total_rows: Optional[int]      # None for large files (skip counting)
    total_columns: int
    column_names: List[str]
    preview_df: pd.DataFrame

# Helper Functions
def _get_extension(filename: str) -> str:
    return os.path.splitext(filename)[1].lower()

def validate_file(
    filename: str,
    file_bytes: bytes,
    max_file_size_mb: int = MAX_FILE_SIZE_MB
) -> FileMetadata:
    """
    Validate:
    - Extension (.xlsx)
    - Size (in MB)
    Returns basic metadata with filename and size.
    """
    ext = _get_extension(filename)
    if ext not in ALLOWED_EXTENSIONS:
        raise FileValidationError(
            f"Invalid file type '{ext}'. Only {', '.join(ALLOWED_EXTENSIONS)} files are allowed."
        )

    size_bytes = len(file_bytes)
    size_mb = size_bytes / (1024 * 1024)

    if size_mb > max_file_size_mb:
        raise FileValidationError(
            f"File size {size_mb:.2f} MB exceeds the limit of {max_file_size_mb} MB."
        )

    return FileMetadata(
        filename=filename,
        size_bytes=size_bytes,
        size_mb=size_mb,
        sheet_names=[]
    )

def _choose_sheet(sheet_names: List[str], requested_sheet: Optional[str]) -> str:
    """
    Pick the sheet to use:
    - if requested_sheet is None → first sheet
    - else → check existence, else raise SheetNotFoundError
    """
    if not sheet_names:
        raise ExcelReadError("Workbook has no sheets.")

    if len(sheet_names) > MAX_SHEETS:
        sheet_names = sheet_names[:MAX_SHEETS]

    if requested_sheet is None:
        return sheet_names[0]

    if requested_sheet not in sheet_names:
        raise SheetNotFoundError(
            f"Sheet '{requested_sheet}' not found. Available sheets: {', '.join(sheet_names)}"
        )

    return requested_sheet

# Main API Function
def process_excel_upload(
    filename: str,
    file_bytes: bytes,
    requested_sheet: Optional[str] = None,
    preview_rows: int = DEFAULT_PREVIEW_ROWS,
    max_file_size_mb: int = MAX_FILE_SIZE_MB,
) -> PreviewResult:
    """
    High-level API for Part 2.

    Smart behavior:
    - Always reads only top N rows for preview.
    - For small/medium files (<= LARGE_FILE_THRESHOLD_MB):
        → also compute exact total row count.
    - For very large files (> LARGE_FILE_THRESHOLD_MB):
        → skip full row counting (total_rows=None) to stay fast.
    """

    # basic validation
    metadata = validate_file(filename, file_bytes, max_file_size_mb)

    try:
        with pd.ExcelFile(io.BytesIO(file_bytes)) as xls:
            sheet_names = xls.sheet_names or []
            if not sheet_names:
                raise ExcelReadError("The Excel file has no sheets.")

            if len(sheet_names) > MAX_SHEETS:
                sheet_names = sheet_names[:MAX_SHEETS]

            metadata.sheet_names = sheet_names

            # choose sheet
            target_sheet = _choose_sheet(sheet_names, requested_sheet)

            # read only top N rows for preview
            df_preview = pd.read_excel(
                xls,
                sheet_name=target_sheet,
                nrows=preview_rows
            )
            # limit columns if sheet is insanely wide
            if df_preview.shape[1] > MAX_COLUMNS:
                df_preview = df_preview.iloc[:, :MAX_COLUMNS]

            total_columns = df_preview.shape[1]
            column_names = list(df_preview.columns)

            # decide whether to compute total row count
            if metadata.size_mb > LARGE_FILE_THRESHOLD_MB:
                # Large file → skip heavy row counting
                total_rows: Optional[int] = None
            else:
                # Medium/small file → do a light scan on first column to count rows
                df_for_count = pd.read_excel(
                    xls,
                    sheet_name=target_sheet,
                    usecols=[0]
                )
                total_rows = len(df_for_count)

            # detect fully empty sheet
            if (total_rows == 0 or total_rows is None) and total_columns == 0:
                raise ExcelReadError(f"Selected sheet '{target_sheet}' appears to be empty.")

    except (FileValidationError, SheetNotFoundError):
        raise
    except ExcelReadError:
        raise
    except Exception as e:
        raise ExcelReadError(f"Failed to read Excel file: {str(e)}") from e

    return PreviewResult(
        metadata=metadata,
        selected_sheet=target_sheet,
        total_rows=total_rows,
        total_columns=total_columns,
        column_names=column_names,
        preview_df=df_preview,
    )