import { useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';

function Home() {
  const navigate = useNavigate();
  const containerRef = useRef<HTMLDivElement | null>(null);
  const [cursor, setCursor] = useState({ x: 0, y: 0 });

  const sectionRef = useRef<HTMLDivElement | null>(null);
  const sectionRafRef = useRef<number | null>(null);
  const [sectionCursor, setSectionCursor] = useState({ x: 0, y: 0 });

  const handleContainerMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
    const el = containerRef.current;
    if (!el) return;
    const rect = el.getBoundingClientRect();
    setCursor({
      x: e.clientX - rect.left,
      y: e.clientY - rect.top,
    });
  };

  return (
    <div
      ref={containerRef}
      onMouseMove={handleContainerMouseMove}
      className="relative min-h-screen flex flex-col overflow-hidden bg-gradient-to-br from-slate-950 via-blue-950 to-sky-600 text-white"
    >
      <div
        className="pointer-events-none absolute inset-0 opacity-80 z-0"
        style={{
          background: `radial-gradient(150px circle at ${cursor.x}px ${cursor.y}px, rgba(56, 189, 248, 0.28), rgba(15, 23, 42, 0) 55%)`,
        }}
      />
      <div className="relative z-10">
        <nav className="sticky top-0 z-[1000] bg-white/95 backdrop-blur-lg shadow-md py-4">
          <div className="max-w-7xl mx-auto px-4 md:px-8 flex flex-col md:flex-row justify-between items-center gap-4 md:gap-0">
            <div className="flex items-center">
              <h2 className="text-xl md:text-2xl font-bold bg-gradient-to-r from-slate-900 via-blue-800 to-sky-500 bg-clip-text text-transparent">
                RAG System
              </h2>
            </div>
            <div className="flex gap-2 md:gap-3 items-center w-full md:w-auto justify-center md:justify-end">
              <button
                className="px-5 md:px-7 py-2.5 md:py-3 text-xs md:text-sm font-semibold uppercase tracking-wide rounded-lg bg-gradient-to-r from-slate-900 via-blue-800 to-sky-500 text-white hover:-translate-y-0.5 hover:shadow-lg hover:shadow-sky-500/30 transition-all w-full md:w-auto"
                onClick={() => navigate('/chat')}
              >
                Get Started
              </button>
            </div>
          </div>
        </nav>

        <header className="py-6 md:py-8 text-center bg-white/10 backdrop-blur-lg">
          <h1 className="text-3xl md:text-5xl font-bold mb-2 drop-shadow-lg">RAG System</h1>
          <p className="text-base md:text-xl opacity-90">Retrieval-Augmented Generation</p>
        </header>

        <main className="flex-1 max-w-7xl w-full mx-auto px-4 md:px-8 py-8 md:py-12">
          <section
            ref={sectionRef}
            className="relative overflow-hidden bg-white/10 backdrop-blur-xl text-white rounded-2xl md:rounded-3xl p-6 md:p-12 mb-8 shadow-2xl border border-white/20"
            onMouseMove={(e) => {
              const el = sectionRef.current;
              if (!el) return;
              if (sectionRafRef.current) return;
              sectionRafRef.current = window.requestAnimationFrame(() => {
                sectionRafRef.current = null;
                const rect = el.getBoundingClientRect();
                setSectionCursor({
                  x: e.clientX - rect.left,
                  y: e.clientY - rect.top,
                });
              });
            }}
            onMouseLeave={() => {
              if (sectionRafRef.current) {
                window.cancelAnimationFrame(sectionRafRef.current);
                sectionRafRef.current = null;
              }
            }}
          >
            <div
              className="pointer-events-none absolute inset-0 opacity-50"
              style={{
                background: `radial-gradient(100px circle at ${sectionCursor.x}px ${sectionCursor.y}px, rgba(56, 189, 248, 0.4), transparent 70%)`,
              }}
            />
            <div className="relative">
              <h2 className="text-2xl md:text-4xl mb-6 text-center text-white font-bold drop-shadow-lg">What is RAG?</h2>
              <div className="leading-relaxed">
                <p className="text-lg mb-8 text-center">
                  <strong>Retrieval-Augmented Generation (RAG)</strong> is an advanced AI technique that combines
                  the power of information retrieval with generative language models. RAG enhances the capabilities
                  of Large Language Models (LLMs) by allowing them to access and incorporate external knowledge
                  sources in real-time.
                </p>

                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 my-8">
                  <div className="bg-white/15 backdrop-blur-md text-white p-6 rounded-2xl shadow-lg hover:-translate-y-1 hover:shadow-xl hover:bg-white/25 transition-all border border-white/30">
                    <h3 className="text-xl mb-4 font-semibold">🔍 Intelligent Retrieval</h3>
                    <p className="text-sm leading-relaxed">
                      RAG systems retrieve relevant information from knowledge bases, documents, or databases
                      before generating responses, ensuring accuracy and up-to-date information.
                    </p>
                  </div>

                  <div className="bg-white/15 backdrop-blur-md text-white p-6 rounded-2xl shadow-lg hover:-translate-y-1 hover:shadow-xl hover:bg-white/25 transition-all border border-white/30">
                    <h3 className="text-xl mb-4 font-semibold">🧠 Context-Aware Generation</h3>
                    <p className="text-sm leading-relaxed">
                      By augmenting prompts with retrieved context, RAG enables LLMs to produce more accurate,
                      relevant, and factually grounded responses.
                    </p>
                  </div>

                  <div className="bg-white/15 backdrop-blur-md text-white p-6 rounded-2xl shadow-lg hover:-translate-y-1 hover:shadow-xl hover:bg-white/25 transition-all border border-white/30">
                    <h3 className="text-xl mb-4 font-semibold">📚 Knowledge Integration</h3>
                    <p className="text-sm leading-relaxed">
                      RAG seamlessly integrates external knowledge sources, allowing AI systems to access
                      domain-specific information without retraining the model.
                    </p>
                  </div>

                  <div className="bg-white/15 backdrop-blur-md text-white p-6 rounded-2xl shadow-lg hover:-translate-y-1 hover:shadow-xl hover:bg-white/25 transition-all border border-white/30">
                    <h3 className="text-xl mb-4 font-semibold">🎯 Enhanced Accuracy</h3>
                    <p className="text-sm leading-relaxed">
                      By grounding responses in retrieved documents, RAG reduces hallucinations and improves
                      the reliability of AI-generated content.
                    </p>
                  </div>
                </div>

                <div className="mt-8 md:mt-12 pt-6 md:pt-8 border-t-2 border-white/30">
                  <h3 className="text-2xl md:text-3xl mb-6 text-center text-white font-bold drop-shadow-lg">How RAG Works</h3>
                  <ol className="max-w-2xl mx-auto pl-6 md:pl-8 space-y-4 text-base md:text-lg leading-relaxed">
                    <li><strong>Query Processing:</strong> Your question is analyzed and processed</li>
                    <li><strong>Information Retrieval:</strong> Relevant documents and data are retrieved from knowledge bases</li>
                    <li><strong>Context Augmentation:</strong> Retrieved information is added to your query as context</li>
                    <li><strong>Response Generation:</strong> The LLM generates an answer based on both your query and the retrieved context</li>
                  </ol>
                </div>
              </div>
            </div>
          </section>
        </main>

        <footer className="py-8 text-center bg-black/20 backdrop-blur-lg">
          <p className="opacity-80">&copy; 2024 RAG System. Powered by advanced AI technology.</p>
        </footer>
      </div>
    </div>
  );
}

export default Home;