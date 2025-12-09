import React, { useState, useEffect } from "react";
import OperationsPanel from "./components/OperationsPanel";

function App() {
  const [file, setFile] = useState(null);
  const [history, setHistory] = useState([]);
  const [operations, setOperations] = useState({
    handle_missing: {},
    remove_duplicates: {},
    replace_values: {},
    normalize_text: {},
    filter_rows: {},
    merge_columns: {},
    convert_data_types: {},
    remove_outliers: {},
  });
  const [priorityOps, setPriorityOps] = useState([]);

  const toggleOp = (op) => setOperations(prev => ({ ...prev, [op]: prev[op] !== null ? null : {} }));
  const addToPriority = (op) => setPriorityOps(prev => prev.includes(op) ? prev.filter(x => x !== op) : [...prev, op]);

  const processFile = async () => {
    if (!file) { alert("Please upload a file!"); return; }

    const selectedOps = Object.keys(operations).filter(key => operations[key] !== null);
    const activeOpsConfig = {};
    selectedOps.forEach(op => activeOpsConfig[op] = operations[op] || {});

    const formData = new FormData();
    formData.append("file", file);
    formData.append("operations", JSON.stringify(activeOpsConfig));
    formData.append("priority_ops", JSON.stringify(priorityOps));

    try {
      const res = await fetch("http://localhost:8000/process", { method: "POST", body: formData });
      if (!res.ok) throw new Error("Processing failed!");

      const data = await res.json();
      alert("File processed successfully!");

      const lastProcess = { filename: file.name, operations: selectedOps, priority: priorityOps, timestamp: new Date().toISOString() };
      localStorage.setItem("last_process", JSON.stringify(lastProcess));

      loadHistory();
    } catch (err) { console.error(err); alert("ERROR: Could not process file."); }
  };

  const loadHistory = async () => {
    try {
      const res = await fetch("http://localhost:8000/history");
      const data = await res.json();
      const oneWeekAgo = new Date();
      oneWeekAgo.setDate(oneWeekAgo.getDate() - 7);
      setHistory(data.filter(h => new Date(h.created_at*1000) >= oneWeekAgo));
    } catch (err) { console.error("Failed to load history", err); }
  };

  useEffect(() => { loadHistory(); }, []);

  return (
    <div className="max-w-6xl mx-auto p-10">
      <h1 className="text-3xl font-bold mb-6">Data Cleaning Process</h1>

      <input type="file" onChange={e => setFile(e.target.files[0])} className="mb-6" />

      <OperationsPanel
        file={file}
        operations={operations}
        toggleOp={toggleOp}
        priorityOps={priorityOps}
        addToPriority={addToPriority}
        processFile={processFile}
        history={history}
      />

      <div className="mt-10 p-6 bg-white shadow-lg rounded-xl">
        <h2 className="text-xl font-bold mb-3">Last Process Summary</h2>
        {localStorage.getItem("last_process") ? (() => {
          const last = JSON.parse(localStorage.getItem("last_process"));
          return (
            <div>
              <p><b>File:</b> {last.filename}</p>
              <p><b>Operations Applied:</b> {last.operations.join(", ") || "None"}</p>
              <p><b>Priority Order:</b> {last.priority.join(" → ") || "None"}</p>
              <p><b>Time:</b> {new Date(last.timestamp).toLocaleString()}</p>
            </div>
          );
        })() : <p className="text-gray-500">No processing done yet.</p>}
      </div>
    </div>
  );
}

export default App;
