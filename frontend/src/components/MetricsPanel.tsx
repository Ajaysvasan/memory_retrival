import type { ChatMetrics } from '../services/chatService';

interface MetricsPanelProps {
  currentMetrics: ChatMetrics | null;
  averageMetrics: ChatMetrics;
  totalMessages: number;
}

function MetricsPanel({ currentMetrics, averageMetrics, totalMessages }: MetricsPanelProps) {
  const formatLatency = (ms: number) => {
    if (ms < 1000) {
      return `${Math.round(ms)}ms`;
    }
    return `${(ms / 1000).toFixed(2)}s`;
  };

  const getScoreColorClass = (score: number) => {
    if (score >= 0.8) return 'text-green-500';
    if (score >= 0.6) return 'text-yellow-500';
    return 'text-red-500';
  };

  const getLatencyColorClass = (ms: number) => {
    if (ms < 500) return 'text-green-500';
    if (ms < 1000) return 'text-yellow-500';
    return 'text-red-500';
  };

  const getBarColorClass = (score: number) => {
    if (score >= 0.8) return 'bg-green-500';
    if (score >= 0.6) return 'bg-yellow-500';
    return 'bg-red-500';
  };

  return (
    <div className="flex flex-col h-full bg-white">
      {/* Sticky Header */}
      <div className="sticky top-0 p-6 border-b border-gray-200 flex justify-between items-center bg-gradient-to-r from-slate-900 via-blue-800 to-sky-500 text-white shrink-0 z-10">
        <h2 className="text-xl font-bold m-0">Performance Metrics</h2>
        <span className="bg-white/20 px-3 py-1 rounded-xl text-sm font-semibold">
          {totalMessages} queries
        </span>
      </div>

      {/* Scrollable Content */}
      <div className="flex-1 overflow-y-auto p-6">
        {currentMetrics ? (
          <>
            <div className="mb-8">
              <h3 className="m-0 mb-4 text-sm font-semibold text-gray-700 uppercase tracking-wide text-xs">
                Current Query
              </h3>

              <div className="bg-gray-50 border border-gray-200 rounded-xl p-5 mb-4 transition-all hover:shadow-md hover:-translate-y-0.5">
                <div className="flex items-center gap-2 mb-3 text-sm font-semibold text-gray-600">
                  <span className="text-xl">🎯</span>
                  Confidence Score
                </div>
                <div className="mb-2">
                  <span className={`text-3xl font-bold leading-none ${getScoreColorClass(currentMetrics.confidenceScore)}`}>
                    {(currentMetrics.confidenceScore * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="w-full h-2 bg-gray-200 rounded-full overflow-hidden mt-3">
                  <div
                    className={`h-full rounded-full transition-all ${getBarColorClass(currentMetrics.confidenceScore)}`}
                    style={{ width: `${currentMetrics.confidenceScore * 100}%` }}
                  />
                </div>
              </div>

              <div className="bg-gray-50 border border-gray-200 rounded-xl p-5 mb-4 transition-all hover:shadow-md hover:-translate-y-0.5">
                <div className="flex items-center gap-2 mb-3 text-sm font-semibold text-gray-600">
                  <span className="text-xl">⚡</span>
                  Latency
                </div>
                <div className="mb-2">
                  <span className={`text-3xl font-bold leading-none ${getLatencyColorClass(currentMetrics.latency)}`}>
                    {formatLatency(currentMetrics.latency)}
                  </span>
                </div>
                <div className="flex gap-2 flex-wrap mt-2">
                  {currentMetrics.cacheHit && (
                    <span className="px-3 py-1 rounded-md text-xs font-semibold bg-blue-100 text-blue-800">
                      Cache Hit
                    </span>
                  )}
                  {currentMetrics.retrievedChunks && (
                    <span className="px-3 py-1 rounded-md text-xs font-semibold bg-purple-100 text-purple-800">
                      {currentMetrics.retrievedChunks} chunks
                    </span>
                  )}
                </div>
              </div>

              <div className="bg-gray-50 border border-gray-200 rounded-xl p-5 mb-4 transition-all hover:shadow-md hover:-translate-y-0.5">
                <div className="flex items-center gap-2 mb-3 text-sm font-semibold text-gray-600">
                  <span className="text-xl">✓</span>
                  Accuracy
                </div>
                <div className="mb-2">
                  <span className={`text-3xl font-bold leading-none ${getScoreColorClass(currentMetrics.accuracy)}`}>
                    {(currentMetrics.accuracy * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="w-full h-2 bg-gray-200 rounded-full overflow-hidden mt-3">
                  <div
                    className={`h-full rounded-full transition-all ${getBarColorClass(currentMetrics.accuracy)}`}
                    style={{ width: `${currentMetrics.accuracy * 100}%` }}
                  />
                </div>
              </div>
            </div>

            {totalMessages > 0 && (
              <div className="mb-8">
                <h3 className="m-0 mb-4 text-sm font-semibold text-gray-700 uppercase tracking-wide text-xs">
                  Average Performance
                </h3>

                <div className="bg-gray-50 border border-gray-200 rounded-xl p-5 mb-4 transition-all hover:shadow-md hover:-translate-y-0.5">
                  <div className="flex items-center gap-2 mb-3 text-sm font-semibold text-gray-600">
                    <span className="text-xl">📊</span>
                    Avg Confidence
                  </div>
                  <div>
                    <span className={`text-3xl font-bold leading-none ${getScoreColorClass(averageMetrics.confidenceScore)}`}>
                      {(averageMetrics.confidenceScore * 100).toFixed(1)}%
                    </span>
                  </div>
                </div>

                <div className="bg-gray-50 border border-gray-200 rounded-xl p-5 mb-4 transition-all hover:shadow-md hover:-translate-y-0.5">
                  <div className="flex items-center gap-2 mb-3 text-sm font-semibold text-gray-600">
                    <span className="text-xl">⏱️</span>
                    Avg Latency
                  </div>
                  <div>
                    <span className={`text-3xl font-bold leading-none ${getLatencyColorClass(averageMetrics.latency)}`}>
                      {formatLatency(averageMetrics.latency)}
                    </span>
                  </div>
                </div>

                <div className="bg-gray-50 border border-gray-200 rounded-xl p-5 mb-4 transition-all hover:shadow-md hover:-translate-y-0.5">
                  <div className="flex items-center gap-2 mb-3 text-sm font-semibold text-gray-600">
                    <span className="text-xl">📈</span>
                    Avg Accuracy
                  </div>
                  <div>
                    <span className={`text-3xl font-bold leading-none ${getScoreColorClass(averageMetrics.accuracy)}`}>
                      {(averageMetrics.accuracy * 100).toFixed(1)}%
                    </span>
                  </div>
                </div>
              </div>
            )}
          </>
        ) : (
          <div className="flex flex-col items-center justify-center h-full text-gray-500 text-center p-8">
            <div className="text-5xl mb-4">📊</div>
            <p className="m-0 text-sm">Metrics will appear here after you send your first message</p>
          </div>
        )}
      </div>
    </div>
  );
}

export default MetricsPanel;