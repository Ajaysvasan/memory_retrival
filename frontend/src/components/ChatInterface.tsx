import { useState } from 'react';
import type { FormEvent } from 'react';
import type { ChatMessage } from '../services/chatService';

interface ChatInterfaceProps {
  messages: ChatMessage[];
  isLoading: boolean;
  onSendMessage: (message: string) => void;
  messagesEndRef: React.RefObject<HTMLDivElement | null>;
}

function ChatInterface({ messages, isLoading, onSendMessage, messagesEndRef }: ChatInterfaceProps) {
  const [input, setInput] = useState('');

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault();
    if (input.trim() && !isLoading) {
      onSendMessage(input);
      setInput('');
    }
  };

  const formatTime = (date: Date) => {
    return new Date(date).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  return (
    <div className="flex flex-col h-full min-h-0">
      <div className="flex-1 overflow-y-auto p-8 flex flex-col gap-6 bg-gray-50 min-h-0">
        {messages.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full text-gray-500 text-center p-8">
            <div className="text-6xl mb-4">💬</div>
            <h3 className="text-2xl mb-2 text-gray-700">Start a conversation</h3>
            <p className="text-base">Ask questions about RAG, or anything you'd like to know!</p>
          </div>
        ) : (
          messages.map((message) => (
            <div
              key={message.id}
              className={`flex mb-4 animate-[fadeIn_0.3s_ease-in] ${message.role === 'user' ? 'justify-end' : 'justify-start'
                }`}
            >
              <div
                className={`max-w-[70%] px-5 py-4 rounded-xl shadow-sm ${message.role === 'user'
                  ? 'bg-gradient-to-r from-slate-900 via-blue-800 to-sky-500 text-white rounded-br-sm'
                  : 'bg-white text-gray-700 border border-gray-200 rounded-bl-sm'
                  }`}
              >
                <div className="flex justify-between items-center mb-2 text-xs opacity-80">
                  <span className="font-semibold uppercase tracking-wide">
                    {message.role === 'user' ? 'You' : 'RAG Assistant'}
                  </span>
                  <span className="ml-4">{formatTime(message.timestamp)}</span>
                </div>
                <div className="leading-relaxed break-words whitespace-pre-wrap">{message.content}</div>
              </div>
            </div>
          ))
        )}
        {isLoading && (
          <div className="flex mb-4 justify-start animate-[fadeIn_0.3s_ease-in]">
            <div className="max-w-[70%] px-5 py-4 rounded-xl shadow-sm bg-white text-gray-700 border border-gray-200 rounded-bl-sm">
              <div className="flex justify-between items-center mb-2 text-xs opacity-80">
                <span className="font-semibold uppercase tracking-wide">RAG Assistant</span>
              </div>
              <div className="flex gap-2 py-2">
                <span className="w-2 h-2 rounded-full bg-sky-500 animate-[typing_1.4s_infinite_ease-in-out]"></span>
                <span className="w-2 h-2 rounded-full bg-sky-500 animate-[typing_1.4s_infinite_ease-in-out] [animation-delay:0.2s]"></span>
                <span className="w-2 h-2 rounded-full bg-sky-500 animate-[typing_1.4s_infinite_ease-in-out] [animation-delay:0.4s]"></span>
              </div>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      <form className="p-6 bg-white border-t border-gray-200" onSubmit={handleSubmit}>
        <div className="flex gap-3 items-center max-w-full">
          <input
            type="text"
            className="flex-1 px-5 py-3.5 border-2 border-gray-200 rounded-xl text-base transition-all focus:outline-none focus:border-sky-500 focus:ring-3 focus:ring-sky-500/10 disabled:bg-gray-100 disabled:cursor-not-allowed"
            placeholder="Type your message..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
            disabled={isLoading}
          />
          <button
            type="submit"
            className="px-5 py-3.5 bg-gradient-to-r from-slate-900 via-blue-800 to-sky-500 text-white rounded-xl flex items-center justify-center transition-all min-w-[50px] hover:-translate-y-0.5 hover:shadow-lg hover:shadow-sky-500/30 disabled:opacity-50 disabled:cursor-not-allowed"
            disabled={!input.trim() || isLoading}
          >
            <svg
              width="20"
              height="20"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <line x1="22" y1="2" x2="11" y2="13"></line>
              <polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>
            </svg>
          </button>
        </div>
      </form>
    </div>
  );
}

export default ChatInterface;
