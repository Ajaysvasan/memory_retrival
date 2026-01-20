import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.tsx'

console.log('main.tsx loaded');

const rootElement = document.getElementById('root');
if (!rootElement) {
  console.error('Root element not found!');
  document.body.innerHTML = '<div style="padding: 2rem; color: red;"><h1>Error: Root element not found</h1></div>';
} else {
  console.log('Root element found, rendering app...');
  try {
    createRoot(rootElement).render(
      <StrictMode>
        <App />
      </StrictMode>,
    );
    console.log('App rendered successfully');
  } catch (error) {
    console.error('Error rendering app:', error);
    rootElement.innerHTML = `<div style="padding: 2rem; color: red;"><h1>Render Error</h1><p>${String(error)}</p></div>`;
  }
}
