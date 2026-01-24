import { Navigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';

interface ProtectedRouteProps {
  children: React.ReactNode;
}

function ProtectedRoute({ children }: ProtectedRouteProps) {
  const { user, isLoading } = useAuth();

  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-slate-950 via-blue-950 to-sky-600 p-6">
        <div className="w-full max-w-md rounded-3xl bg-white/95 backdrop-blur-lg shadow-2xl p-8">
          <div className="flex items-center gap-4">
            <div className="h-10 w-10 rounded-2xl bg-sky-100 flex items-center justify-center">
              <div className="h-5 w-5 rounded-full border-2 border-sky-600 border-t-transparent animate-spin" />
            </div>
            <div>
              <div className="text-lg font-bold text-gray-900">Loading</div>
              <div className="text-sm text-gray-600">Preparing your session…</div>
            </div>
          </div>
        </div>
      </div>
    );
  }

  if (!user) {
    return <Navigate to="/auth/login" replace />;
  }

  return <>{children}</>;
}

export default ProtectedRoute;
