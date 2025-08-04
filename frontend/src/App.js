import React from "react";
import Chat from "./Chat";
import UploadAndTrain from "./UploadAndTrain";

export default function App() {
  return (
    <div className="min-h-screen flex flex-col">
      <header className="bg-gradient-to-r from-blue-600 to-indigo-600 p-6 text-white flex items-center space-x-4 shadow-lg">
        <img src="/logo.png" alt="Logo" className="h-12" />
        <h1 className="text-3xl font-bold">PetroAgent</h1>
      </header>
      <img src="/banner.png" alt="Banner" className="w-full object-cover h-48" />
      <main className="flex-1 container mx-auto p-6 space-y-12">
        <Chat />
        <UploadAndTrain />
      </main>
      <footer className="bg-gray-200 text-center p-4">
        © 2025 PetroAgent Inc.
      </footer>
    </div>
  );
}
