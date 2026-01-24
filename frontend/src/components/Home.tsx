import { useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';

function Home() {
  const navigate = useNavigate();
  const containerRef = useRef<HTMLDivElement | null>(null);
  const [cursor, setCursor] = useState({ x: 0, y: 0 });
  const [hoveredAlgo, setHoveredAlgo] = useState<string | null>(null);
  const [hoveredFlow, setHoveredFlow] = useState<string | null>(null);

  const handleContainerMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
    const el = containerRef.current;
    if (!el) return;
    const rect = el.getBoundingClientRect();
    setCursor({
      x: e.clientX - rect.left,
      y: e.clientY - rect.top,
    });
  };

  const algorithms = [
    {
      id: 'hnsw',
      name: 'HNSW (Hierarchical Navigable Small World)',
      icon: '🔍',
      description: 'A graph-based algorithm for approximate nearest neighbor search with O(log n) complexity. Creates a multi-layer graph structure where each layer contains increasingly sparse connections, enabling efficient similarity search in high-dimensional vector spaces.'
    },
    {
      id: 'bm25',
      name: 'BM25 Lexical Retrieval',
      icon: '📝',
      description: 'A probabilistic ranking function based on term frequency and inverse document frequency. Excels at exact keyword matching and complements dense retrieval by capturing lexical signals that embeddings might miss.'
    },
    {
      id: 'cross-encoder',
      name: 'Cross-Encoder Reranking',
      icon: '🎯',
      description: 'A neural reranking model that scores query-document pairs jointly. Processes both inputs together to capture fine-grained semantic relationships, significantly improving retrieval precision at the cost of higher latency.'
    },
    {
      id: 'rl-policy',
      name: 'RL-Based Retrieval Policy',
      icon: '🤖',
      description: 'A reinforcement learning agent that decides when and how to retrieve. Learns to balance evidence quality, latency, consistency, and cost by optimizing a reward function that penalizes hallucinations and unnecessary retrievals.'
    },
    {
      id: 'belief-store',
      name: 'Belief Store with Versioning',
      icon: '💾',
      description: 'A persistent knowledge base storing verified factual claims with supporting evidence, confidence scores, and version history. Enables long-term consistency by preventing contradictory answers across conversations.'
    },
    {
      id: 'memory',
      name: 'Hierarchical Memory System',
      icon: '🧠',
      description: 'Multi-tier memory architecture: Short-Term Memory for recent turns, Working Memory for active facts, Episodic Memory for summarized conversations, and a Belief Store for persistent verified knowledge.'
    }
  ];

  const flowSteps = [
    {
      id: 'query',
      name: 'User Query',
      description: 'The system receives a user question and preprocesses it for intent classification and topic extraction.'
    },
    {
      id: 'state',
      name: 'Conversation State & Topic Tracking',
      description: 'Maintains conversation context, tracks active topics, and identifies relevant beliefs from previous interactions.'
    },
    {
      id: 'rl',
      name: 'RL-Based Retrieval Policy',
      description: 'The RL agent decides whether to skip retrieval, retrieve small/large k chunks, or refuse to answer based on confidence and cache state.'
    },
    {
      id: 'retrieval',
      name: 'Hybrid Retrieval (HNSW + BM25 + Cache)',
      description: 'Performs dense vector search (HNSW), lexical search (BM25), checks GPU cache, and applies cross-encoder reranking.'
    },
    {
      id: 'evidence',
      name: 'Evidence Alignment & Contradiction Detection',
      description: 'Extracts factual claims from retrieved evidence, verifies alignment with query, and checks for contradictions with existing beliefs.'
    },
    {
      id: 'belief',
      name: 'Belief Store Update',
      description: 'Updates the persistent belief store with new verified facts, confidence scores, and version history for future consistency.'
    },
    {
      id: 'answer',
      name: 'Answer / Refusal',
      description: 'Generates an evidence-grounded answer or explicitly refuses if evidence is insufficient or contradictory, prioritizing trust over fluency.'
    }
  ];

  return (
    <div
      ref={containerRef}
      onMouseMove={handleContainerMouseMove}
      className="relative min-h-screen flex flex-col overflow-x-hidden bg-gradient-to-br from-slate-950 via-blue-950 to-sky-900 text-white"
    >
      <style>{`
        @keyframes flipIn {
          0% {
            opacity: 0;
            transform: rotateY(-90deg) scale(0.8);
          }
          100% {
            opacity: 1;
            transform: rotateY(0) scale(1);
          }
        }
        @keyframes flipOut {
          0% {
            opacity: 1;
            transform: rotateY(0) scale(1);
          }
          100% {
            opacity: 0;
            transform: rotateY(90deg) scale(0.8);
          }
        }
        @keyframes slideInLeft {
          0% {
            opacity: 0;
            transform: translateX(-30px) scale(0.95);
          }
          100% {
            opacity: 1;
            transform: translateX(0) scale(1);
          }
        }
        @keyframes slideInRight {
          0% {
            opacity: 0;
            transform: translateX(30px) scale(0.95);
          }
          100% {
            opacity: 1;
            transform: translateX(0) scale(1);
          }
        }
        @keyframes fadeIn {
          0% { opacity: 0; }
          100% { opacity: 1; }
        }
        .flip-in {
          animation: flipIn 0.4s ease-out forwards;
        }
        .flip-out {
          animation: flipOut 0.4s ease-out forwards;
        }
        .slide-in-left {
          animation: slideInLeft 0.4s ease-out forwards;
        }
        .slide-in-right {
          animation: slideInRight 0.4s ease-out forwards;
        }
      `}</style>
      <div
        className="pointer-events-none absolute inset-0 opacity-80 z-0"
        style={{
          background: `radial-gradient(200px circle at ${cursor.x}px ${cursor.y}px, rgba(56, 189, 248, 0.28), rgba(15, 23, 42, 0) 55%)`,
        }}
      />
      
      <div className="relative z-10">
        {/* Navigation */}
        <nav className="sticky top-0 z-[1000] bg-white/95 backdrop-blur-lg shadow-md py-4">
          <div className="max-w-7xl mx-auto px-4 md:px-8 flex flex-col md:flex-row justify-between items-center gap-4 md:gap-0">
            <div className="flex items-center">
              <h2 className="text-xl md:text-2xl font-bold bg-gradient-to-r from-slate-900 via-blue-800 to-sky-500 bg-clip-text text-transparent">
                Memory-Aware RAG System
              </h2>
            </div>
            <div className="flex gap-2 md:gap-3 items-center w-full md:w-auto justify-center md:justify-end">
              <button
                className="px-5 md:px-7 py-2.5 md:py-3 text-xs md:text-sm font-semibold uppercase tracking-wide rounded-lg bg-sky-500 text-white hover:-translate-y-0.5 hover:shadow-lg hover:shadow-sky-500/30 transition-all w-full md:w-auto"
                onClick={() => navigate('/chat')}
              >
                Get Started
              </button>
            </div>
          </div>
        </nav>

        {/* Hero Section */}
        <header className="py-12 md:py-16 text-center px-4">
          <h1 className="text-4xl md:text-6xl font-bold mb-4 drop-shadow-lg">
            Memory-Aware, Consistency-Preserving RAG
          </h1>
          <p className="text-lg md:text-2xl opacity-90 mb-2">with RL-Controlled Retrieval</p>
          <p className="text-base md:text-lg opacity-75 max-w-3xl mx-auto mt-4">
            A next-generation RAG system that eliminates hallucinations, maintains consistency across conversations,
            and optimizes retrieval through reinforcement learning
          </p>
        </header>

        <main className="max-w-7xl w-full mx-auto px-4 md:px-8 py-8 md:py-12 space-y-16">
          
          {/* Section 1: What is RAG */}
          <section className="relative overflow-hidden bg-white/10 backdrop-blur-xl rounded-3xl p-8 md:p-12 shadow-2xl border border-white/20">
            <h2 className="text-3xl md:text-4xl font-bold mb-6 text-center">What is RAG?</h2>
            <div className="space-y-6 text-base md:text-lg leading-relaxed">
              <p>
                <strong>Retrieval-Augmented Generation (RAG)</strong> combines information retrieval with generative language models.
                Instead of relying solely on a model's parametric knowledge, RAG systems retrieve relevant external documents
                and incorporate them into the generation process.
              </p>
              <p>
                Traditional RAG systems face critical challenges: hallucinations despite retrieval, degrading accuracy over long
                conversations, inconsistent answers to repeated questions, and stateless retrieval that ignores prior verified knowledge.
              </p>
              <p>
                Our system addresses these limitations through <strong>memory-aware architecture</strong>, <strong>RL-based retrieval control</strong>,
                and <strong>belief persistence</strong> to ensure consistent, evidence-grounded answers across extended conversations.
              </p>
            </div>
          </section>

          {/* Section 2 & 3: Architecture + Flow */}
          
          {/* Section 2: Our Architecture */}
          <section className="relative overflow-hidden bg-white/10 backdrop-blur-xl rounded-3xl p-8 md:p-12 shadow-2xl border border-white/20">
            <h2 className="text-2xl md:text-3xl font-bold mb-8 text-center">Our Architecture</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {algorithms.map((algo) => (
                <div
                  key={algo.id}
                  className="relative group"
                  onMouseEnter={() => setHoveredAlgo(algo.id)}
                  onMouseLeave={() => setHoveredAlgo(null)}
                >
                  <div className="bg-white/15 backdrop-blur-md p-6 rounded-xl border border-white/30 cursor-pointer hover:bg-sky-500 hover:border-sky-400 hover:shadow-2xl hover:scale-105 transition-all duration-300 shadow-lg h-full min-h-[200px] flex flex-col items-center justify-center overflow-hidden">
                    {hoveredAlgo === algo.id ? (
                      <div className="flip-in text-center px-2">
                        <p className="text-sm leading-relaxed">{algo.description}</p>
                      </div>
                    ) : (
                      <div className="flip-in flex flex-col items-center text-center gap-3">
                        <span className="text-4xl transform group-hover:scale-110 transition-transform duration-300">{algo.icon}</span>
                        <h3 className="text-base font-semibold leading-tight">{algo.name}</h3>
                      </div>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </section>

          {/* Section 3: System Flow */}
          <section className="relative overflow-hidden bg-white/10 backdrop-blur-xl rounded-3xl p-8 md:p-12 shadow-2xl border border-white/20">
            <h2 className="text-2xl md:text-3xl font-bold mb-12 text-center">System Flow</h2>
            <div className="relative max-w-4xl mx-auto">
              {/* Vertical Line */}
              <div className="absolute left-1/2 top-0 bottom-0 w-1 bg-gradient-to-b from-sky-400 via-blue-500 to-sky-600 transform -translate-x-1/2 z-0"></div>
              
              <div className="space-y-12 relative z-10">
                {flowSteps.map((step, index) => (
                  <div
                    key={step.id}
                    className={`flex items-center ${index % 2 === 0 ? 'justify-start' : 'justify-end'}`}
                  >
                    <div
                      className={`relative w-5/12 ${index % 2 === 0 ? 'pr-8' : 'pl-8'}`}
                      onMouseEnter={() => setHoveredFlow(step.id)}
                      onMouseLeave={() => setHoveredFlow(null)}
                    >
                      {/* Connection dot */}
                      <div className={`absolute top-1/2 transform -translate-y-1/2 w-4 h-4 bg-sky-400 rounded-full border-4 border-white shadow-lg transition-all duration-300 ${hoveredFlow === step.id ? 'scale-150 bg-sky-300' : ''} ${index % 2 === 0 ? 'right-0' : 'left-0'}`}></div>
                      
                      <div className={`bg-gradient-to-r from-sky-500 to-blue-600 p-5 rounded-xl shadow-lg cursor-pointer hover:scale-110 hover:shadow-2xl hover:from-sky-400 hover:to-blue-500 transition-all duration-300 ${hoveredFlow === step.id ? index % 2 === 0 ? 'slide-in-left' : 'slide-in-right' : ''}`}>
                        <div className="flex flex-col gap-2">
                          <span className="text-xs font-bold opacity-80">STEP {index + 1}</span>
                          <span className="font-bold text-base">{step.name}</span>
                        </div>
                      </div>
                      
                      {hoveredFlow === step.id && (
                        <div className={`absolute ${index === flowSteps.length - 1 ? 'bottom-full mb-4' : 'top-full mt-4'} ${index % 2 === 0 ? 'left-0' : 'right-0'} w-full bg-white text-gray-800 p-4 rounded-xl shadow-2xl z-30 border-2 border-sky-500 ${index % 2 === 0 ? 'slide-in-left' : 'slide-in-right'}`}>
                          <p className="text-sm leading-relaxed font-medium">{step.description}</p>
                        </div>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </section>

          {/* Key Features */}
          <section className="relative overflow-hidden bg-white/10 backdrop-blur-xl rounded-3xl p-8 md:p-12 shadow-2xl border border-white/20">
            <h2 className="text-3xl md:text-4xl font-bold mb-8 text-center">Key Innovations</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              <div className="bg-white/15 backdrop-blur-md p-6 rounded-xl border border-white/30 hover:bg-white/25 transition-all">
                <h3 className="text-xl font-bold mb-3">🚫 Hallucination Prevention</h3>
                <p className="text-sm">Evidence alignment verification ensures every claim is grounded in retrieved documents.</p>
              </div>
              <div className="bg-white/15 backdrop-blur-md p-6 rounded-xl border border-white/30 hover:bg-white/25 transition-all">
                <h3 className="text-xl font-bold mb-3">🔄 Consistency Across Time</h3>
                <p className="text-sm">Belief store maintains verified facts to prevent contradictory answers in long conversations.</p>
              </div>
              <div className="bg-white/15 backdrop-blur-md p-6 rounded-xl border border-white/30 hover:bg-white/25 transition-all">
                <h3 className="text-xl font-bold mb-3">⚡ Optimized Retrieval</h3>
                <p className="text-sm">RL policy learns to skip unnecessary retrievals while maintaining answer quality.</p>
              </div>
              <div className="bg-white/15 backdrop-blur-md p-6 rounded-xl border border-white/30 hover:bg-white/25 transition-all">
                <h3 className="text-xl font-bold mb-3">💬 Honest Refusals</h3>
                <p className="text-sm">System explicitly refuses to answer when evidence is insufficient, prioritizing trust over fluency.</p>
              </div>
              <div className="bg-white/15 backdrop-blur-md p-6 rounded-xl border border-white/30 hover:bg-white/25 transition-all">
                <h3 className="text-xl font-bold mb-3">🧠 Multi-Tier Memory</h3>
                <p className="text-sm">Hierarchical memory system from short-term to episodic to persistent beliefs.</p>
              </div>
              <div className="bg-white/15 backdrop-blur-md p-6 rounded-xl border border-white/30 hover:bg-white/25 transition-all">
                <h3 className="text-xl font-bold mb-3">📊 Explainable Decisions</h3>
                <p className="text-sm">UI exposes evidence used, belief updates, and retrieval decisions for full transparency.</p>
              </div>
            </div>
          </section>

        </main>

        <footer className="py-8 text-center bg-black/20 backdrop-blur-lg mt-16">
          <p className="opacity-80">&copy; 2024 Memory-Aware RAG System. Research Project.</p>
        </footer>
      </div>
    </div>
  );
}

export default Home;