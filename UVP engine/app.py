import streamlit as st

from excel_upload_engine1 import (
    process_excel_upload,
    FileValidationError,
    SheetNotFoundError,
    ExcelReadError,
    DEFAULT_PREVIEW_ROWS,
)

st.set_page_config(page_title="Excel_Upload_Validation_&_Preview", layout="wide")

@st.cache_data
def get_preview_cached(
    file_bytes: bytes,
    filename: str,
    sheet: str | None,
    preview_rows: int,
    max_file_size_mb: int,
):
    """
    Cached wrapper around process_excel_upload.
    """
    return process_excel_upload(
        filename=filename,
        file_bytes=file_bytes,
        requested_sheet=sheet,
        preview_rows=preview_rows,
        max_file_size_mb=max_file_size_mb,
    )

def main():
    st.title("Excel_Upload_Engine")

    # User-configurable max file size
    max_file_size_mb = st.number_input(
        "Maximum allowed file size (MB):",
        min_value=10,
        max_value=1000,
        value=200,    
        step=10,
    )

    uploaded_file = st.file_uploader("Upload Excel file (.xlsx):", type=["xlsx"])

    if not uploaded_file:
        st.info("Please upload an Excel file to continue.")
        return

    # Read file bytes once
    file_bytes = uploaded_file.read()
    filename = uploaded_file.name

    approx_size_mb = len(file_bytes) / (1024 * 1024)
    st.write(f"**Uploaded file:** `{filename}`")
    st.write(f"Approx. size: `{approx_size_mb:.2f} MB`")
    st.write(f"Current configured max: `{max_file_size_mb} MB`")

    preview_rows = st.slider("Number of rows to preview:", 5, 100, DEFAULT_PREVIEW_ROWS)

    try:
        # Initial call with sheet=None: get sheet names + default sheet preview
        initial_result = get_preview_cached(
            file_bytes=file_bytes,
            filename=filename,
            sheet=None,
            preview_rows=preview_rows,
            max_file_size_mb=max_file_size_mb,
        )

        sheets = initial_result.metadata.sheet_names

        st.subheader("🧾 Workbook Sheets")
        st.write(sheets)

        default_index = sheets.index(initial_result.selected_sheet)
        selected_sheet = st.selectbox(
            "Select sheet to preview:",
            sheets,
            index=default_index
        )

        # Re-run if user changes sheet
        if selected_sheet != initial_result.selected_sheet:
            result = get_preview_cached(
                file_bytes=file_bytes,
                filename=filename,
                sheet=selected_sheet,
                preview_rows=preview_rows,
                max_file_size_mb=max_file_size_mb,
            )
        else:
            result = initial_result

        # Handle total_rows possibly being None
        if result.total_rows is None:
            total_rows_display = "Not computed for large file (to keep it fast)"
        else:
            total_rows_display = result.total_rows

        st.subheader("**File Metadata**")
        st.json(
            {
                "filename": result.metadata.filename,
                "size_mb": round(result.metadata.size_mb, 2),
                "sheet_names": result.metadata.sheet_names,
                "selected_sheet": result.selected_sheet,
                "total_rows": total_rows_display,
                "total_columns": result.total_columns,
                "column_names": result.column_names,
            }
        )

        st.subheader("**Data Preview**")
        st.dataframe(result.preview_df, use_container_width=True)

        st.success("Preview generated successfully")

    except FileValidationError as e:
        st.error(f"File validation error: {e}")
    except SheetNotFoundError as e:
        st.error(f"Sheet error: {e}")
    except ExcelReadError as e:
        st.error(f"Excel parsing error: {e}")
    except Exception as e:
        st.error(f"Unexpected error: {e}")

if __name__ == "__main__":
    main()