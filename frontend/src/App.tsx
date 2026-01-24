import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Home from './components/Home';
import Chat from './components/Chat';
import ErrorBoundary from './components/ErrorBoundary';

function App() {
  console.log('App component rendering...');

  try {
    return (
      <ErrorBoundary>
        <Router>
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/chat" element={<Chat />} />
          </Routes>
        </Router>
      </ErrorBoundary>
    );
  } catch (error) {
    console.error('Error in App component:', error);
    return (
      <div style={{ padding: '2rem', color: 'red' }}>
        <h1>App Error</h1>
        <p>{String(error)}</p>
      </div>
    );
  }
}

export default App;
