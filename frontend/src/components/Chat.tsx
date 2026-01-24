import { useState, useRef, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { chatService, type ChatMessage, type ChatMetrics } from '../services/chatService';
import ChatInterface from './ChatInterface';
import MetricsPanel from './MetricsPanel';

function Chat() {
  const navigate = useNavigate();
  const { user, logout } = useAuth();
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [currentMetrics, setCurrentMetrics] = useState<ChatMetrics | null>(null);
  const [averageMetrics, setAverageMetrics] = useState<ChatMetrics>({
    confidenceScore: 0,
    latency: 0,
    accuracy: 0,
  });
  const messagesEndRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    // Redirect to login if not authenticated
    if (!user) {
      navigate('/auth/login');
    }
  }, [user, navigate]);

  useEffect(() => {
    // Scroll to bottom when new message arrives
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  useEffect(() => {
    // Calculate average metrics
    if (currentMetrics) {
      const totalMessages = messages.filter(m => m.role === 'assistant').length;
      if (totalMessages > 1) {
        setAverageMetrics((prev) => ({
          confidenceScore: (prev.confidenceScore * (totalMessages - 1) + currentMetrics.confidenceScore) / totalMessages,
          latency: (prev.latency * (totalMessages - 1) + currentMetrics.latency) / totalMessages,
          accuracy: (prev.accuracy * (totalMessages - 1) + currentMetrics.accuracy) / totalMessages,
        }));
      } else {
        setAverageMetrics(currentMetrics);
      }
    }
  }, [currentMetrics, messages]);

  const handleSendMessage = async (content: string) => {
    if (!content.trim() || isLoading) return;

    // Add user message
    const userMessage: ChatMessage = {
      id: Date.now().toString(),
      role: 'user',
      content: content.trim(),
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setIsLoading(true);

    try {
      const response = await chatService.sendMessage(content.trim());

      // Add assistant message
      const assistantMessage: ChatMessage = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: response.message,
        timestamp: new Date(),
      };

      setMessages((prev) => [...prev, assistantMessage]);
      setCurrentMetrics(response.metrics);
    } catch (error: any) {
      // Add error message
      const errorMessage: ChatMessage = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: `Error: ${error.message || 'Failed to get response'}`,
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleClearChat = () => {
    setMessages([]);
    setCurrentMetrics(null);
    setAverageMetrics({
      confidenceScore: 0,
      latency: 0,
      accuracy: 0,
    });
  };

  const handleLogout = () => {
    logout();
    navigate('/');
  };

  if (!user) {
    return null;
  }

  return (
    <div className="h-screen flex flex-col bg-slate-50 overflow-hidden">
      <header className="bg-gradient-to-r from-slate-900 via-blue-800 to-sky-500 text-white py-4 shadow-md shrink-0">
        <div className="max-w-full mx-auto px-4 lg:px-8 flex flex-col lg:flex-row justify-between items-start lg:items-center gap-4 lg:gap-0">
          <div className="flex flex-col gap-1">
            <h1 className="text-xl lg:text-2xl font-bold">RAG System Chat</h1>
            <span className="text-xs lg:text-sm opacity-90">Welcome, {user.name}</span>
          </div>
          <div className="flex gap-2 lg:gap-3 items-center w-full lg:w-auto justify-end">
            <button
              className="px-3 lg:px-4 py-1.5 lg:py-2 bg-white/20 text-white border border-white/30 rounded-md text-xs lg:text-sm font-medium transition-all hover:bg-white/30 hover:-translate-y-0.5"
              onClick={handleClearChat}
            >
              Clear Chat
            </button>
            <button
              className="px-3 lg:px-4 py-1.5 lg:py-2 bg-white/20 text-white border border-white/30 rounded-md text-xs lg:text-sm font-medium transition-all hover:bg-white/30 hover:-translate-y-0.5"
              onClick={() => navigate('/')}
            >
              Home
            </button>
            <button
              className="px-3 lg:px-4 py-1.5 lg:py-2 bg-red-500/30 text-white border border-red-500/50 rounded-md text-xs lg:text-sm font-medium transition-all hover:bg-red-500/50 hover:-translate-y-0.5"
              onClick={handleLogout}
            >
              Logout
            </button>
          </div>
        </div>
      </header>

      <div className="flex-1 flex flex-col lg:flex-row overflow-hidden">
        {/* Chat Area */}
        <div className="flex-1 flex flex-col bg-white lg:border-r border-gray-200 overflow-hidden">
          <ChatInterface
            messages={messages}
            isLoading={isLoading}
            onSendMessage={handleSendMessage}
            messagesEndRef={messagesEndRef}
          />
        </div>

        {/* Metrics Panel */}
        <div className="w-full lg:w-[400px] flex flex-col bg-white lg:border-l border-t lg:border-t-0 border-gray-200 overflow-hidden">
          <MetricsPanel
            currentMetrics={currentMetrics}
            averageMetrics={averageMetrics}
            totalMessages={messages.filter(m => m.role === 'assistant').length}
          />
        </div>
      </div>
    </div>
  );
}

export default Chat;