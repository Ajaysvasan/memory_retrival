// yea I am going to hard code some values
// They left me with no option
import { useState, useRef, useEffect } from "react";
import type { FormEvent } from "react";
import type { ChatMessage } from "../services/chatService";
// If the response is string , why on earth here it is an array
// Halluciation at it's peak
interface ChatInterfaceProps {
  messages: ChatMessage[];
  isLoading: boolean;
  onSendMessage: (message: string) => void;
  messagesEndRef: React.RefObject<HTMLDivElement | null>;
}

function ChatInterface({
  messages,
  isLoading,
  onSendMessage,
  messagesEndRef,
}: ChatInterfaceProps) {
  const [input, setInput] = useState("");
  const panel2EndRef = useRef<HTMLDivElement | null>(null);
  const panel3EndRef = useRef<HTMLDivElement | null>(null);
  const panel4EndRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    // Auto-scroll only panel 2, 3, and 4 when new messages arrive
    panel2EndRef.current?.scrollIntoView({ behavior: "smooth" });
    panel3EndRef.current?.scrollIntoView({ behavior: "smooth" });
    panel4EndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault();
    if (input.trim() && !isLoading) {
      onSendMessage(input);
      setInput("");
    }
  };

  const formatTime = (date: Date) => {
    return new Date(date).toLocaleTimeString([], {
      hour: "2-digit",
      minute: "2-digit",
    });
  };

  const scrollbarHideStyle = {
    msOverflowStyle: "none" as const,
    scrollbarWidth: "none" as const,
    WebkitOverflowScrolling: "touch" as const,
  };

  const renderMessages = (
    panelNumber: number,
    panelName: string,
    endRef?: React.RefObject<HTMLDivElement | null>
  ) => (
    <>
      {messages.length === 0 ? (
        <div className="flex flex-col items-center justify-center h-full text-gray-500 text-center p-4">
          <div className="text-5xl mb-4">💬</div>
          <h3 className="text-xl mb-2 text-gray-700 font-semibold">
            {panelName}
          </h3>
          <p className="text-base">Messages will appear here</p>
        </div>
      ) : (
        <>
          {messages.map((message) => (
            <div
              key={`${message.id}-${panelNumber}`}
              className={`flex mb-4 animate-[fadeIn_0.3s_ease-in] ${
                message.role === "user" ? "justify-end" : "justify-start"
              }`}
            >
              <div
                className={`max-w-[85%] px-4 py-3 rounded-lg shadow-sm ${
                  message.role === "user"
                    ? "bg-gradient-to-r from-slate-900 via-blue-800 to-sky-500 text-white rounded-br-sm"
                    : "bg-white text-gray-700 border border-gray-200 rounded-bl-sm"
                }`}
              >
                <div className="flex justify-between items-center mb-2 text-xs opacity-80">
                  <span className="font-semibold uppercase tracking-wide">
                    {message.role === "user" ? "You" : "RAG Assistant"}
                  </span>
                  <span className="ml-3">{formatTime(message.timestamp)}</span>
                </div>
                <div className="leading-relaxed break-words whitespace-pre-wrap text-sm">
                  {message.content}
                </div>
              </div>
            </div>
          ))}
          {isLoading && (
            <div className="flex mb-4 justify-start animate-[fadeIn_0.3s_ease-in]">
              <div className="max-w-[85%] px-4 py-3 rounded-lg shadow-sm bg-white text-gray-700 border border-gray-200 rounded-bl-sm">
                <div className="flex justify-between items-center mb-2 text-xs opacity-80">
                  <span className="font-semibold uppercase tracking-wide">
                    RAG Assistant
                  </span>
                </div>
                <div className="flex gap-2 py-2">
                  <span className="w-2 h-2 rounded-full bg-sky-500 animate-[typing_1.4s_infinite_ease-in-out]"></span>
                  <span className="w-2 h-2 rounded-full bg-sky-500 animate-[typing_1.4s_infinite_ease-in-out] [animation-delay:0.2s]"></span>
                  <span className="w-2 h-2 rounded-full bg-sky-500 animate-[typing_1.4s_infinite_ease-in-out] [animation-delay:0.4s]"></span>
                </div>
              </div>
            </div>
          )}
          {endRef && <div ref={endRef} />}
        </>
      )}
    </>
  );

  return (
    <div className="flex flex-col h-full">
      <style>{`
        .hide-scrollbar::-webkit-scrollbar {
          display: none;
        }
      `}</style>

      {/* Four Equal Panels Grid */}
      <div className="flex-1 grid grid-cols-2 grid-rows-2 gap-0 overflow-hidden">
        {/* Panel 1 - Top Left - Our RAG */}
        <div className="border-r border-b border-gray-300 overflow-hidden flex flex-col">
          <div className="sticky top-0 bg-gradient-to-r from-slate-900 via-blue-800 to-sky-500 text-white px-5 py-3 font-bold text-base shadow-md z-10 shrink-0">
            Our RAG
          </div>
          <div
            className="flex-1 overflow-y-auto p-5 bg-gray-50 hide-scrollbar"
            style={scrollbarHideStyle}
          >
            {renderMessages(1, "Our RAG")}
            <div ref={messagesEndRef} />
          </div>
        </div>

        {/* Panel 2 - Top Right - Hybrid RAG */}
        <div className="border-b border-gray-300 overflow-hidden flex flex-col">
          <div className="sticky top-0 bg-gradient-to-r from-slate-900 via-blue-800 to-sky-500 text-white px-5 py-3 font-bold text-base shadow-md z-10 shrink-0">
            Hybrid RAG
          </div>
          <div
            className="flex-1 overflow-y-auto p-5 bg-gray-50 hide-scrollbar"
            style={scrollbarHideStyle}
          >
            {renderMessages(2, "Hybrid RAG", panel2EndRef)}
          </div>
        </div>

        {/* Panel 3 - Bottom Left - FID RAG */}
        <div className="border-r border-gray-300 overflow-hidden flex flex-col">
          <div className="sticky top-0 bg-gradient-to-r from-slate-900 via-blue-800 to-sky-500 text-white px-5 py-3 font-bold text-base shadow-md z-10 shrink-0">
            FID RAG
          </div>
          <div
            className="flex-1 overflow-y-auto p-5 bg-gray-50 hide-scrollbar"
            style={scrollbarHideStyle}
          >
            {renderMessages(3, "FID RAG", panel3EndRef)}
          </div>
        </div>

        {/* Panel 4 - Bottom Right - Agentic RAG */}
        <div className="overflow-hidden flex flex-col">
          <div className="sticky top-0 bg-gradient-to-r from-slate-900 via-blue-800 to-sky-500 text-white px-5 py-3 font-bold text-base shadow-md z-10 shrink-0">
            Agentic RAG
          </div>
          <div
            className="flex-1 overflow-y-auto p-5 bg-gray-50 hide-scrollbar"
            style={scrollbarHideStyle}
          >
            {renderMessages(4, "Agentic RAG", panel4EndRef)}
          </div>
        </div>
      </div>

      {/* Single Sticky Input Form */}
      <form
        className="sticky bottom-0 p-6 bg-white border-t-2 border-gray-300 shadow-lg shrink-0"
        onSubmit={handleSubmit}
      >
        <div className="flex gap-3 items-center max-w-full">
          <input
            type="text"
            className="flex-1 px-5 py-3.5 border-2 border-gray-200 rounded-xl text-base transition-all focus:outline-none focus:border-sky-500 focus:ring-3 focus:ring-sky-500/10 disabled:bg-gray-100 disabled:cursor-not-allowed"
            placeholder="Ask to our RAG Assistant...."
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

