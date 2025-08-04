import React from 'react';
import Chat from './Chat';

export default function App() {
  return (
    <div className="min-h-screen flex flex-col items-center p-6">
      <header className="mb-6 text-center">
        <img src="/banner.png" alt="Banner" className="w-full max-w-xl rounded"/>
        <h1 className="text-4xl font-bold mt-4">PetroAgent</h1>
      </header>
      <Chat />
    </div>
  );
}
