import React from 'react';
import Chatbot from './components/Chatbot';
import FileUploader from './components/FileUploader';

export default function App() {
  return (
    <div className="h-screen flex">
      <div className="w-1/3 p-4 border-r">
        <FileUploader />
      </div>
      <div className="flex-1 p-4">
        <Chatbot />
      </div>
    </div>
  );
}
