export default function FooterHistory({ history }) {
    return (
      <div className="mt-10 p-4 bg-gray-100 rounded-xl">
        <h3 className="font-semibold mb-3">Past Processed Files</h3>
  
        {history.map((h) => (
          <div key={h.id} className="flex justify-between mb-2">
            <span>{h.name}</span>
            <a
              className="px-3 py-1 bg-green-600 text-white rounded"
              href={`http://localhost:8000/download/${h.id}`}
            >
              Download
            </a>
          </div>
        ))}
      </div>
    );
  }
  