import React from "react";

export default function FileUpload({ setFile }) {
  return (
    <div>
      <input
        type="file"
        onChange={(e) => setFile(e.target.files[0])}
        className="p-2 border rounded"
      />
    </div>
  );
}
