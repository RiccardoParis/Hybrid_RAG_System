Markdown
# Hybrid Multi-Source RAG System 🧠🧬

An advanced Retrieval-Augmented Generation (RAG) architecture tailored for the medical domain. This system intelligently routes user queries across three distinct databases (Vector, Graph, and Relational) to synthesize highly accurate, hallucination-free answers using Large Language Models.

## 🏗️ System Architecture

The system utilizes an agentic workflow powered by **LangGraph** to process, route, and fuse information:
1. **Intelligent Router:** Analyzes the structural intent of the query (semantic, topological, or quantitative) to select the appropriate data source.
2. **Qdrant (Vector DB):** Handles semantic searches across medical abstracts and scientific literature using cross-lingual embeddings (`multilingual-e5-base`).
3. **Neo4j (Knowledge Graph):** Manages multi-hop reasoning and ontological relationships (e.g., diseases, side effects, drugs) using Dynamic Schema-Augmented Cypher Generation.
4. **PostgreSQL (Relational DB):** Processes quantitative queries and exact data filtering (e.g., clinical trial enrollments) via the Table-Augmented Generation (TAG) paradigm.
5. **Late Fusion Node:** An LLM synthesizes the raw extracted data from all sources into a single, coherent natural language response.

## ⚙️ Prerequisites

- Python 3.10+
- Docker & Docker Compose
- Groq API Key (for LLM inference)

## 🚀 Installation & Setup

**1. Clone the repository and install dependencies:**

```bash
git clone [https://github.com/your-username/hybrid-rag-system.git](https://github.com/your-username/hybrid-rag-system.git)
cd hybrid-rag-system
pip install -r requirements.txt
2. Configure Environment Variables:
Create a .env file in the root directory (refer to .env.example):

Snippet di codice
GROQ_API_KEY=your_api_key_here
POSTGRES_URI=postgresql://postgres:Password@127.0.0.1:5433/medical_rag_db
QDRANT_URL=http://localhost:6333
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=Password
3. Launch the Databases (Docker):

Bash
docker compose up -d
💉 Medical Data Ingestion
The system comes with a massive ingestion pipeline to populate the three databases with medical literature, neurology graphs, and clinical trial records.
To initialize and populate the databases:

Bash
python src/medical_bulk_ingestion.py
(Note: This process may take several minutes depending on your hardware).

💻 Usage
Launch the interactive Streamlit dashboard:

Bash
streamlit run src/app.py
🗺️ Roadmap & Future Work
[ ] Implement Metadata-based Dynamic Routing.

[ ] Integrate Reinforcement Learning from User Feedback (RLUF) to continuously optimize the router.

[ ] Evaluate System Energy Footprint and optimization strategies.
```

### Architettura di Sistema

Di seguito l'architettura del nostro RAG Ibrido Multi-Sorgente con Router Neurale:
![Architettura](Hybrid_RAG_architecture.svg)
