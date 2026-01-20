import { useState, useEffect } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';

function Auth() {
  const navigate = useNavigate();
  const location = useLocation();
  const { login, register } = useAuth();

  const [isLogin, setIsLogin] = useState(true);
  const [formData, setFormData] = useState({
    email: '',
    password: '',
    name: '',
  });
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    // Check if we're on login or register route
    const path = location.pathname;
    setIsLogin(path.includes('/login'));
  }, [location]);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value
    });
    setError('');
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setLoading(true);

    try {
      if (isLogin) {
        await login(formData.email, formData.password);
      } else {
        if (!formData.name.trim()) {
          setError('Name is required');
          setLoading(false);
          return;
        }
        await register(formData.email, formData.password, formData.name);
      }

      // Redirect to chat page after successful auth
      navigate('/chat');
    } catch (err: any) {
      setError(err.message || 'An error occurred. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const switchMode = () => {
    setIsLogin(!isLogin);
    setError('');
    setFormData({ email: '', password: '', name: '' });
    navigate(isLogin ? '/auth/register' : '/auth/login');
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-fuchsia-500 to-pink-500 p-8">
      <div className="bg-white/95 backdrop-blur-lg rounded-3xl p-12 w-full max-w-md shadow-2xl">
        <div className="text-center mb-8">
          <h1 className="text-3xl font-bold mb-2 text-gray-800">
            {isLogin ? 'Welcome Back' : 'Create Account'}
          </h1>
          <p className="text-gray-600 text-sm">
            {isLogin ? 'Sign in to continue' : 'Sign up to get started'}
          </p>
        </div>

        {error && (
          <div className="bg-red-50 text-red-700 p-4 rounded-lg mb-6 border border-red-200 text-sm">
            {error}
          </div>
        )}

        <form onSubmit={handleSubmit} className="flex flex-col gap-6">
          {!isLogin && (
            <div className="flex flex-col">
              <label htmlFor="name" className="font-semibold mb-2 text-gray-700 text-sm">
                Full Name
              </label>
              <input
                type="text"
                id="name"
                name="name"
                value={formData.name}
                onChange={handleChange}
                required
                placeholder="Enter your full name"
                disabled={loading}
                className="px-4 py-3.5 border-2 border-gray-200 rounded-lg text-base transition-all focus:outline-none focus:border-fuchsia-500 focus:ring-3 focus:ring-fuchsia-500/10 disabled:bg-gray-100 disabled:cursor-not-allowed"
              />
            </div>
          )}

          <div className="flex flex-col">
            <label htmlFor="email" className="font-semibold mb-2 text-gray-700 text-sm">
              Email
            </label>
            <input
              type="email"
              id="email"
              name="email"
              value={formData.email}
              onChange={handleChange}
              required
              placeholder="Enter your email"
              disabled={loading}
              className="px-4 py-3.5 border-2 border-gray-200 rounded-lg text-base transition-all focus:outline-none focus:border-fuchsia-500 focus:ring-3 focus:ring-fuchsia-500/10 disabled:bg-gray-100 disabled:cursor-not-allowed"
            />
          </div>

          <div className="flex flex-col">
            <label htmlFor="password" className="font-semibold mb-2 text-gray-700 text-sm">
              Password
            </label>
            <input
              type="password"
              id="password"
              name="password"
              value={formData.password}
              onChange={handleChange}
              required
              placeholder="Enter your password"
              disabled={loading}
              minLength={6}
              className="px-4 py-3.5 border-2 border-gray-200 rounded-lg text-base transition-all focus:outline-none focus:border-fuchsia-500 focus:ring-3 focus:ring-fuchsia-500/10 disabled:bg-gray-100 disabled:cursor-not-allowed"
            />
          </div>

          <button
            type="submit"
            className="px-4 py-4 bg-gradient-to-r from-fuchsia-500 to-pink-500 text-white rounded-lg text-base font-semibold uppercase tracking-wide mt-2 hover:-translate-y-0.5 hover:shadow-lg hover:shadow-fuchsia-500/40 transition-all disabled:opacity-60 disabled:cursor-not-allowed"
            disabled={loading}
          >
            {loading ? 'Processing...' : (isLogin ? 'Sign In' : 'Sign Up')}
          </button>
        </form>

        <div className="text-center mt-8 pt-8 border-t border-gray-200">
          <p className="text-gray-600 text-sm">
            {isLogin ? "Don't have an account? " : 'Already have an account? '}
            <button
              type="button"
              className="bg-none border-none text-fuchsia-500 cursor-pointer font-semibold text-sm p-0 underline transition-colors hover:text-pink-500 disabled:opacity-60 disabled:cursor-not-allowed"
              onClick={switchMode}
              disabled={loading}
            >
              {isLogin ? 'Sign Up' : 'Sign In'}
            </button>
          </p>
        </div>

        <div className="mt-6 text-center">
          <button
            type="button"
            className="bg-none border-none text-fuchsia-500 cursor-pointer text-sm py-2 transition-colors hover:text-pink-500 hover:underline disabled:opacity-60 disabled:cursor-not-allowed"
            onClick={() => navigate('/')}
            disabled={loading}
          >
            ← Back to Home
          </button>
        </div>
      </div>
    </div>
  );
}

export default Auth;
