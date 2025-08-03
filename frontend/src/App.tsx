import React from "react";
import FileUploader from "./components/FileUploader";
import ChatInterface from "./components/ChatInterface";

export default function App() {
  const [csvInfo, setCsvInfo] = React.useState<any>(null);

  return (
    <div className="p-4 max-w-2xl mx-auto">
      <h1 className="text-2xl font-bold mb-4">Petro Agent</h1>
      <FileUploader onDetect={setCsvInfo} />
      {csvInfo && (
        <pre className="my-4 p-2 bg-gray-100 rounded">
          {JSON.stringify(csvInfo, null, 2)}
        </pre>
      )}
      <ChatInterface />
    </div>
  );
}
