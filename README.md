# RAG-TCRL-X 
[Visit our site](https://memory-retrival.vercel.app/?_vercel_share=amK7VJ0GaMYbVEBuuJkw85Xj5OGF1p39)

<div align="center">

**Topic-Conditioned, Validation-First Retrieval-Augmented Generation**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-success.svg)]()

_A deterministic, explainable, and context-aware RAG system for real-world AI applications_

[Features](#-key-features) • [Architecture](#-architecture) • [Quick Start](#-quick-start) • [Documentation](#-documentation) • [Contributing](#-contributing)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Problem Statement](#-problem-statement)
- [Key Features](#-key-features)
- [Architecture](#-architecture)
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
- [Usage](#-usage)
- [Technical Challenges](#-technical-challenges)
- [How It Works](#-how-it-works)
- [Performance](#-performance)
- [Roadmap](#-roadmap)
- [Contributing](#-contributing)
- [Team](#-team)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)

---

## 🎯 Overview

RAG-TCRL-X is an advanced Retrieval-Augmented Generation (RAG) system designed to overcome the fundamental limitations of traditional RAG architectures. Unlike conventional systems that struggle with long conversations and context management, RAG-TCRL-X delivers **consistent**, **context-aware**, and **evidence-grounded** responses through innovative topic-conditioned retrieval and validation-first generation.

### What Makes RAG-TCRL-X Different?

- **Validation-First Architecture**: Evidence is verified before generation, not after
- **Topic-Aware Context Management**: Intelligent topic tracking prevents context rot
- **Zero Hallucination Tolerance**: Refuses to answer when evidence is insufficient
- **Fully Local**: No external APIs, ensuring privacy and control
- **Production-Ready**: Designed for real-world deployment in critical domains

---

## 🚨 Problem Statement

Modern RAG systems face critical challenges that limit their reliability in production environments:

### Core Issues

| Challenge                | Impact                                                  | Traditional Approach       | Our Solution                                 |
| ------------------------ | ------------------------------------------------------- | -------------------------- | -------------------------------------------- |
| **Context Rotting**      | Loss of earlier discussion points in long conversations | Limited context windows    | Topic-conditioned Context State Object (CSO) |
| **Inconsistent Answers** | Different responses to the same question                | Simple embedding retrieval | Multi-layer topic-aware caching              |
| **Hallucinations**       | Confident but incorrect responses                       | Post-generation filtering  | Pre-generation validation engine             |
| **Latency vs Depth**     | Trade-off between speed and quality                     | Single-strategy retrieval  | Adaptive multi-tier caching                  |

### Real-World Impact

These failures make traditional RAG systems unreliable for:

- 🏥 Healthcare knowledge bases
- 📚 Educational platforms
- 🔧 Technical documentation systems
- 🔬 Research assistance tools
- 💼 Enterprise knowledge management

---

## ✨ Key Features

### 🧠 Intelligent Context Management

- **Context State Object (CSO)**: Maintains long-term conversational coherence using topic-based decay mechanisms
- **Smart Context Preservation**: No need to pass entire chat history to the model
- **Temporal Awareness**: Tracks topic relevance over conversation lifetime

### 🎯 Topic-Conditioned Retrieval

- **Multi-Topic Routing**: Decomposes queries and routes by semantic topics, not just embeddings
- **Domain Agnostic**: Single architecture supports medical, financial, technical, and educational domains
- **Efficient Indexing**: HNSW + FAISS for fast approximate nearest-neighbor search

### ✅ Validation-First Generation

- **Evidence Verification**: Every chunk is validated before being used in generation
- **Confidence Scoring**: Explicit relevance and coverage metrics for each response
- **Hallucination Prevention**: System refuses to answer when evidence is insufficient

### ⚡ Performance Optimized

- **Multi-Layer Caching**: Reduces latency while preserving consistency
- **Topic-Aware Keys**: Prevents context drift in cached responses
- **Adaptive Retrieval**: Balances speed and answer depth dynamically

### 🔒 Privacy & Control

- **Fully Local**: Runs entirely on your infrastructure
- **No External APIs**: Complete data privacy and control
- **Transparent**: Explainable decisions with evidence trails

---

## 🏗️ Architecture

RAG-TCRL-X implements a sophisticated multi-layer architecture designed for reliability and performance:

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Query                                │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│         1. Query Intake & Intent Classification                  │
│    (Factual / Analytical / Comparative / Procedural)            │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│         2. Topic Decomposition & Routing                         │
│        (Map to semantic topics using centroids)                  │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│         3. Retrieval Layer (HNSW + FAISS)                       │
│      (Fast ANN search within topic-specific indexes)            │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│         4. Validation Engine                                     │
│   (Verify relevance, coverage, and alignment)                   │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│         5. Generation Layer (Local LLM)                         │
│    (Evidence-constrained response generation)                   │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│         6. Cache & Memory Control                               │
│      (Topic-aware caching prevents drift)                       │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│         7. RL Control (Optional Extension)                      │
│     (Adaptive retrieval and caching policies)                   │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
                    Generated Response
```

### Component Details

#### 1️⃣ Query Intake & Intent Classification

Analyzes incoming queries to determine their nature and required processing strategy.

#### 2️⃣ Topic Decomposition & Routing

Breaks down complex queries into semantic topics and routes them to appropriate knowledge domains.

#### 3️⃣ Retrieval Layer

High-performance vector search using HNSW (Hierarchical Navigable Small World) graphs and FAISS indexing.

#### 4️⃣ Validation Engine

Multi-stage verification process ensuring retrieved content is relevant, complete, and aligned with the query.

#### 5️⃣ Generation Layer

Local LLM generates responses strictly bounded by validated evidence, preventing hallucinations.

#### 6️⃣ Cache & Memory Control

Intelligent caching system maintains consistency across conversations using topic-aware keys.

#### 7️⃣ RL Control (Optional)

Reinforcement learning module for personalized retrieval and caching optimization.

---

## 📦 Prerequisites

### Hardware Requirements

| Component   | Minimum                     | Recommended                        |
| ----------- | --------------------------- | ---------------------------------- |
| **CPU**     | Intel 8th Gen or equivalent | Intel 10th Gen+ or AMD Ryzen 5000+ |
| **RAM**     | 8 GB                        | 16 GB or higher                    |
| **GPU**     | Not required                | CUDA-enabled GPU (4GB+ VRAM)       |
| **Storage** | 10 GB free space            | 20 GB+ SSD                         |
| **Network** | Required for initial setup  | -                                  |

### Software Requirements

#### Backend

- **Python**: 3.10, 3.11, 3.12, or 3.13
- **Operating System**: Linux, macOS, or Windows 10/11
- **CUDA Toolkit** (optional): For GPU acceleration

#### Frontend

- **Node.js**: 16.x or higher
- **npm**: 8.x or higher

---

## 🚀 Installation

### Option 1: Quick Start (Recommended)

#### Linux / macOS

```bash
# Clone the repository
git clone https://github.com/Ajaysvasan/RAG-TCRL-X.git
cd RAG-TCRL-X

# Make setup script executable
chmod +x run.sh

# Run setup and start backend
./run.sh
```

#### Windows

```batch
# Clone the repository
git clone https://github.com/Ajaysvasan/RAG-TCRL-X.git
cd RAG-TCRL-X

# Run setup and start backend
run.bat
```

### Option 2: Manual Setup

#### Backend Setup

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Linux/macOS:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start the backend server
uvicorn main:app --reload
```

The backend will be available at: `http://localhost:8000`

#### Frontend Setup

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

The frontend will be available at: `http://localhost:5173` (or the port specified by your framework)

---

## 💻 Usage

### Starting the System

#### First Time Setup

Use the automated setup scripts (`run.sh` or `run.bat`) which will:

1. Install all dependencies
2. Download required models
3. Initialize the database
4. Start both backend and frontend servers

#### Subsequent Runs

**Backend:**

```bash
uvicorn main:app --reload
```

**Frontend:**

```bash
npm run dev
```

---

## 🔧 Technical Challenges

### Challenge 1: Context Preservation

**Problem**: Maintaining long-term conversational coherence without exponentially increasing memory usage.

**Solution**: Context State Object (CSO) with topic-based decay mechanisms that intelligently prune less relevant information while preserving critical context.

### Challenge 2: Evidence Grounding

**Problem**: Ensuring generated answers are strictly supported by retrieved data without post-hoc verification.

**Solution**: Validation-first architecture where evidence is verified before generation, with explicit relevance scoring and coverage metrics.

### Challenge 3: Latency vs Detail Trade-off

**Problem**: Producing detailed, nuanced answers without sacrificing response time.

**Solution**: Multi-layer caching system with topic-aware keys and adaptive retrieval strategies that balance speed and comprehensiveness.

### Challenge 4: Scalability Across Domains

**Problem**: Supporting diverse knowledge domains without domain-specific fine-tuning.

**Solution**: Topic-conditioned routing with semantic centroids that automatically adapt to different domains using the same core architecture.

### Challenge 5: Hallucination Control

**Problem**: Preventing confident but incorrect responses when data is insufficient.

**Solution**: Explicit confidence thresholds and a "refuse to answer" mechanism when evidence coverage falls below acceptable levels.

---

## ⚙️ How It Works

### Topic-Conditioned Retrieval

Instead of simple embedding similarity, RAG-TCRL-X uses **topic centroids** to:

1. Decompose complex queries into semantic topics
2. Route each topic to specialized knowledge subspaces
3. Retrieve relevant chunks within topic-specific contexts
4. Merge results while maintaining topic coherence

### Validation Engine

Every retrieved chunk undergoes multi-stage validation:

```python
# Pseudocode for validation process
def validate_chunk(chunk, query, context):
    relevance_score = compute_semantic_relevance(chunk, query)
    coverage_score = assess_query_coverage(chunk, query)
    alignment_score = check_context_alignment(chunk, context)

    if relevance_score < THRESHOLD:
        return REJECT
    if coverage_score < MIN_COVERAGE:
        return PARTIAL
    if alignment_score < ALIGNMENT_MIN:
        return REJECT

    return ACCEPT, (relevance_score, coverage_score, alignment_score)
```

### Context State Object (CSO)

The CSO maintains conversational state using:

- **Topic Weights**: Decay over time but can be reinforced by related queries
- **Entity Tracking**: Monitors mentioned entities and their relationships
- **Temporal Markers**: Tracks when topics were discussed
- **Relevance Pruning**: Removes stale context intelligently

---

---

## 🗺️ Roadmap

### ✅ Completed

- [x] Core RAG architecture with topic-conditioned retrieval
- [x] Validation-first generation pipeline
- [x] Context State Object (CSO) implementation
- [x] Multi-layer caching system
- [x] Basic web interface

### 🚧 In Progress

- [ ] Reinforcement Learning optimization module
- [ ] Multi-modal support (images, tables, diagrams)
- [ ] Advanced analytics dashboard
- [ ] Performance profiling tools

### 📅 Planned

- [ ] Distributed deployment support
- [ ] Multi-language support
- [ ] Fine-tuning interface for domain adaptation
- [ ] Enterprise SSO integration
- [ ] Plugin system for custom validators
- [ ] Mobile applications (iOS/Android)

---

## 🤝 Contributing

We welcome contributions from the community! Here's how you can help:

### Ways to Contribute

1. **Report Bugs**: Open an issue with detailed reproduction steps
2. **Suggest Features**: Share your ideas for improvements
3. **Submit Pull Requests**: Fix bugs or add new features
4. **Improve Documentation**: Help make our docs clearer
5. **Share Use Cases**: Tell us how you're using RAG-TCRL-X

### Development Setup

```bash
# Fork and clone the repository
git clone https://github.com/YOUR_USERNAME/RAG-TCRL-X.git
cd RAG-TCRL-X

# Create a feature branch
git checkout -b feature/amazing-feature

# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Make your changes and commit
git commit -m "Add amazing feature"

# Push to your fork
git push origin feature/amazing-feature

# Open a Pull Request
```

### Code Standards

- Follow PEP 8 for Python code
- Write unit tests for new features
- Update documentation for API changes
- Keep commit messages clear and descriptive

---

## 👥 Team

RAG-TCRL-X is built and maintained by:

<table>
  <tr>
    <td align="center">
      <a href="https://github.com/Ajaysvasan">
        <img src="https://github.com/Ajaysvasan.png" width="100px;" alt="Ajay S Vasan"/>
        <br />
        <sub><b>Ajay S Vasan</b></sub>
      </a>
      <br />
      <sub>Core Architecture & ML</sub>
    </td>
    <td align="center">
      <a href="https://github.com/don1502">
        <img src="https://github.com/don1502.png" width="100px;" alt="Don Christ"/>
        <br />
        <sub><b>Don Christ</b></sub>
      </a>
      <br />
      <sub>Backend & Infrastructure</sub>
    </td>
  </tr>
</table>

### Get in Touch

- 📧 Email: [Contact Form]
- 💬 Discussions: [GitHub Discussions](https://github.com/Ajaysvasan/RAG-TCRL-X/discussions)
- 🐛 Issues: [GitHub Issues](https://github.com/Ajaysvasan/RAG-TCRL-X/issues)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Research Inspiration**: Based on advances in neural retrieval and constrained generation
- **Open Source Libraries**: Built with Transformers, FAISS, FastAPI, and React
- **Community**: Thanks to all contributors and users who provide feedback

---

## 📚 Documentation

For detailed documentation, please visit:

- [Installation Guide](docs/installation.md)
- [API Reference](docs/api-reference.md)
- [Architecture Deep Dive](docs/architecture.md)
- [Configuration Guide](docs/configuration.md)
- [Troubleshooting](docs/troubleshooting.md)

---

## 🌟 Star History

If you find RAG-TCRL-X useful, please consider giving it a star! ⭐

[![Star History Chart](https://api.star-history.com/svg?repos=Ajaysvasan/RAG-TCRL-X&type=Date)](https://star-history.com/#Ajaysvasan/RAG-TCRL-X&Date)

---

<div align="center">

**[⬆ Back to Top](#rag-tcrl-x)**

Made with ❤️ by the RAG-TCRL-X Team

</div>
