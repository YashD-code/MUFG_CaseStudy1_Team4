export default function ProgressBar({ progress }) {
    return (
      <div className="w-full bg-gray-200 rounded mt-4">
        <div
          className="bg-indigo-600 text-xs text-white p-1 rounded"
          style={{ width: progress + "%" }}
        >
          {progress}%
        </div>
      </div>
    );
  }
  