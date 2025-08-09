import React from 'react'
import { createRoot } from 'react-dom/client'
import App from './App'
import './styles.css' // tailwind via @layer

createRoot(document.getElementById('root')!).render(<App />)
