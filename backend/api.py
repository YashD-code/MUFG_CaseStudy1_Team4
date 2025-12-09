import nbformat
from nbclient import NotebookClient
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import tempfile, json, os, uuid, pandas as pd

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("processed_files", exist_ok=True)

def df_to_json_safe(df):
    return df.where(pd.notnull(df), None).to_dict(orient="records")

@app.post("/process")
async def process_file_api(
    file: UploadFile = File(...),
    operations: str = Form(...),
    priority_ops: str = Form(...)
):
    suffix = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp:
        temp.write(await file.read())
        temp.flush()
        temp_path = temp.name

    nb = nbformat.read("processor.ipynb", as_version=4)
    client = NotebookClient(nb)
    client.execute()

    ns = {"pd": pd}
    for cell in nb.cells:
        if cell.cell_type=="code":
            exec(cell.source, ns)

    result = ns["process_file"](temp_path, json.loads(operations), json.loads(priority_ops))
    first_sheet = list(result["data"].keys())[0]
    cleaned_df = result["data"][first_sheet]

    file_id = f"{uuid.uuid4()}.xlsx"
    output_path = f"processed_files/{file_id}"
    cleaned_df.to_excel(output_path, index=False)

    preview_safe = df_to_json_safe(cleaned_df)

    return {
        "status":"completed",
        "file_id":file_id,
        "sheet":first_sheet,
        "preview":preview_safe,
        "operations_performed": list(json.loads(operations).keys())
    }

@app.get("/history")
def get_history():
    files = []
    for f in os.listdir("processed_files"):
        if f.endswith(".xlsx"):
            files.append({
                "id": f,
                "name": f,
                "created_at": os.path.getmtime(f"processed_files/"+f)
            })
    return files

@app.get("/download/{file_id}")
def download(file_id: str):
    return FileResponse(f"processed_files/{file_id}")
