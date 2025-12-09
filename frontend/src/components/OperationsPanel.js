import React from "react";

export default function OperationsPanel({
  operations,
  toggleOp,
  priorityOps,
  addToPriority,
  processFile,
  history,
  file,
}) {
  const opList = Object.keys(operations);

  return (
    <div className="p-6 bg-gray-100 rounded-xl shadow-md">
      <h2 className="text-xl font-bold mb-4">Select Operations</h2>

      {opList.map((op) => {
        const isEnabled = operations[op] !== null;

        return (
          <div
            key={op}
            className="flex items-center justify-between p-3 bg-white rounded-lg shadow mb-3"
          >
            <div>
              <h3 className="font-semibold capitalize">{op.replace(/_/g, " ")}</h3>
              <p className="text-xs text-gray-500">{isEnabled ? "Selected" : "Not Selected"}</p>
            </div>

            <div className="flex gap-2">
              <button
                onClick={() => toggleOp(op)}
                className={`px-3 py-1 rounded-lg text-white ${isEnabled ? "bg-blue-600" : "bg-gray-400"}`}
              >
                Perform
              </button>

              <button
                onClick={() => addToPriority(op)}
                className={`px-3 py-1 rounded-lg text-white ${priorityOps.includes(op) ? "bg-green-600" : "bg-gray-400"}`}
              >
                Priority
              </button>
            </div>
          </div>
        );
      })}

      <button
        onClick={processFile}
        className="mt-6 w-full bg-black text-white py-3 rounded-xl hover:opacity-80"
      >
        {file ? "PROCESS FILE" : "UPLOAD FILE FIRST"}
      </button>

      <h2 className="text-xl font-bold mt-10 mb-3">Processing History </h2>

      {history.length === 0 ? (
        <p className="text-gray-500">No files processed yet.</p>
      ) : (
        history.map((h) => (
          <div
            key={h.id}
            className="border p-3 rounded-lg bg-white shadow-sm mb-2 flex justify-between"
          >
            <div>
              <p className="font-semibold">{h.name}</p>
              <p className="text-xs text-gray-500">{new Date(h.created_at * 1000).toLocaleString()}</p>
              <p className="text-xs text-gray-400">Operations: {h.operations?.join(", ")}</p>
            </div>

            <a href={`http://localhost:8000/download/${h.id}`} className="text-blue-600 underline">
              Download
            </a>
          </div>
        ))
      )}
    </div>
  );
}
