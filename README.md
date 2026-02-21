# Excel Data Transformation Tool

The **Excel Data Transformation Tool** is a powerful web-based tool designed to help you upload, transform, and clean Excel data effortlessly. This tool allows users to apply various data transformations (such as cleaning, mathematical operations) and save these transformations as reusable templates. Templates can be applied to different Excel files with different column names.

## Features

- **File Upload**: Upload Excel files for processing.
- **Data Cleaning**: Remove duplicates, filter rows, replace values, normalize text, and more.
- **Mathematical Operations**: Perform arithmetic, percentage change, weighted average, and other operations.
- **Template Management**: Save transformation workflows as templates for reuse across files.
- **Download Results**: Export the transformed data in Excel or CSV format.

## Installation

To run this project, you'll need Python installed. It's recommended to set up a virtual environment to avoid conflicts with other projects. Below are the steps to get your environment ready.

### Step 1: Clone the Repository

```bash
git clone https://github.com/YashD-code/MUFG_CaseStudy1_Team4.git
cd MUFG_CaseStudy1_Team4
```
### Step 2: Set Up a Virtual Environment

Create a virtual environment for the project:

```bash
python -m venv venv
```
Activate the virtual environment:

On macOS/Linux:
```
source venv/bin/activate
```
On Windows:
```
venv\Scripts\activate
```
### Step 3: Install the Required Dependencies

Install the required Python libraries using pip:
```
pip install -r requirements.txt
```
### Step 4: Run the Application

Run the Streamlit app:
```
streamlit run main.py
```
Once the app is running, open a browser and navigate to http://localhost:8501 to start using the tool.

## Usage

- **Upload an Excel File**: Use the "Upload File" section to upload your Excel file.

- **Data Transformations**: Apply various transformations such as removing duplicates, replacing values, or performing mathematical operations.

- **Save Transformation as Template**: After performing some transformations, you can save the entire operation as a reusable template.

- **Reapply Templates**: You can apply saved templates to new files. You may also remap columns to ensure compatibility across different files.

- **Download Transformed Data**: Once the transformations are applied, you can download the transformed file in Excel or CSV format.

## Template Management

- **Save Templates**: Save a set of operations as a template for future use.

- **Apply Templates**: Reapply saved templates to new files, with support for remapping column names.

- **Column Mapping**: If your template's column names don't match the current file's column names, you can map columns interactively.

## Example Operations
### Data Cleaning Operations

- **Remove Duplicates**: Remove duplicate rows based on selected columns.

- **Filter Rows**: Filter rows based on specific column values and operators.

- **Replace Values**: Replace specific values in a column with new ones.

- **Merge Columns**: Combine multiple columns into a single column with a separator.

- **Normalize Text**: Convert text to lowercase and/or trim leading/trailing spaces.

- **Handle Missing Values**: Drop missing values or fill them with strategies like mean or median.

## Mathematical Operations

- **Arithmetic**: Perform basic arithmetic operations like addition, subtraction, multiplication, and division on columns.

- **Percentage Change**: Calculate percentage change between two columns.

- **Weighted Average**: Calculate the weighted average of multiple columns with specified weights.

- **Aggregate Function**: Apply aggregation functions such as sum, mean, median, min, or max to selected columns.
