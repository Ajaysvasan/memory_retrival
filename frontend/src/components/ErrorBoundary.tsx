import { Component, type ReactNode } from 'react';

interface Props {
  children: ReactNode;
}

interface State {
  hasError: boolean;
  error?: Error;
}

class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    console.error('Error caught by boundary:', error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-slate-950 via-blue-950 to-sky-600 p-6">
          <div className="w-full max-w-xl rounded-3xl bg-white/95 backdrop-blur-lg shadow-2xl p-8 md:p-10">
            <div className="flex items-start gap-4">
              <div className="h-12 w-12 shrink-0 rounded-2xl bg-red-50 flex items-center justify-center text-red-600 text-xl">
                !
              </div>
              <div className="min-w-0">
                <h1 className="text-2xl md:text-3xl font-bold text-gray-900">Something went wrong</h1>
                <p className="mt-2 text-sm text-gray-600 break-words">
                  {this.state.error?.message || 'Unexpected error'}
                </p>
              </div>
            </div>

            <div className="mt-8 flex flex-col sm:flex-row gap-3">
              <button
                className="inline-flex items-center justify-center rounded-xl bg-gradient-to-r from-slate-900 via-blue-800 to-sky-500 px-5 py-3 text-sm font-semibold text-white shadow-md transition-all hover:-translate-y-0.5 hover:shadow-lg hover:shadow-sky-500/30"
                onClick={() => window.location.reload()}
              >
                Reload Page
              </button>
              <button
                className="inline-flex items-center justify-center rounded-xl border border-gray-200 bg-white px-5 py-3 text-sm font-semibold text-gray-700 transition-colors hover:bg-gray-50"
                onClick={() => (window.location.href = '/')}
              >
                Back to Home
              </button>
            </div>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}

export default ErrorBoundary;
