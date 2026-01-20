import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { AuthProvider } from './context/AuthContext';
import Home from './components/Home';
import Auth from './components/Auth';
import Chat from './components/Chat';
import ProtectedRoute from './components/ProtectedRoute';
import ErrorBoundary from './components/ErrorBoundary';

function App() {
  console.log('App component rendering...');

  try {
    return (
      <ErrorBoundary>
        <AuthProvider>
          <Router>
            <Routes>
              <Route path="/" element={<Home />} />
              <Route path="/auth/login" element={<Auth />} />
              <Route path="/auth/register" element={<Auth />} />
              <Route
                path="/chat"
                element={
                  <ProtectedRoute>
                    <Chat />
                  </ProtectedRoute>
                }
              />
            </Routes>
          </Router>
        </AuthProvider>
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
