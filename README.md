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
graph TD
    %% Definizioni Stili
    classDef user fill:#e1bee7,stroke:#8e24aa,stroke-width:2px,color:#000
    classDef orchestrator fill:#ffecb3,stroke:#fbc02d,stroke-width:2px,color:#000
    classDef router fill:#ffe0b2,stroke:#f57c00,stroke-width:2px,color:#000
    classDef db fill:#b3e5fc,stroke:#0288d1,stroke-width:2px,color:#000
    classDef llm fill:#c8e6c9,stroke:#388e3c,stroke-width:2px,color:#000
    classDef process fill:#f5f5f5,stroke:#9e9e9e,stroke-width:1px,color:#000

    %% Nodi Principali
    User((Utente / Streamlit UI)):::user
    ParseQuery[Parse Query Node<br/>LangGraph]:::orchestrator
    
    %% Router e Log
    Router{DistilBERT Router<br/>Intent Recognition}:::router
    RLLogger[(PostgreSQL<br/>RL Logs & Energy)]:::db
    
    %% Flusso Iniziale
    User -->|Query Utente| ParseQuery
    ParseQuery -->|Estrae Metadati| Router
    Router -->|Salva Scelta e Token| RLLogger
    
    %% Routing Paths
    Router -->|no_retrieval| DirectAnswer[Direct Answer Node<br/>Llama-3]:::llm
    Router -->|vector| VectorSearch[Vector Search Node]:::process
    Router -->|sql| SQLSearch[SQL Search Node]:::process
    Router -->|graph| GraphSearch[Graph Search Node]:::process
    Router -->|multi| ResolveMulti{Resolve Multi-Source<br/>LLM Judge}:::llm
    
    %% Multi-Source Split
    ResolveMulti -->|Assegna| VectorSearch
    ResolveMulti -->|Assegna| SQLSearch
    ResolveMulti -->|Assegna| GraphSearch

    %% Database Interactions
    VectorSearch -->|Similarity Search| Qdrant[(Qdrant<br/>Abstract PubMed)]:::db
    SQLSearch -->|Text-to-SQL| Postgres[(PostgreSQL<br/>Trial Clinici)]:::db
    GraphSearch -->|Cypher QA| Neo4j[(Neo4j<br/>Knowledge Graph)]:::db

    %% Graph Logic (Lookup)
    GraphSearch -->|Estrae Nodi/Relazioni| CondGraph{Trovati ID 'ns/'?}:::process
    CondGraph -->|Sì| LookupNode[Lookup Node<br/>Cypher ID-to-Name]:::process
    LookupNode -->|Traduzione Entità| Neo4j
    CondGraph -->|No| LateFusion
    LookupNode --> LateFusion

    %% Late Fusion
    Qdrant -->|Testo Recuperato| LateFusion
    Postgres -->|Dati Tabellari| LateFusion
    
    LateFusion[Late Fusion Node<br/>Sintesi Dati (Llama-3)]:::llm
    
    %% Risposta Finale
    LateFusion --> FinalAnswer([Risposta Finale Unificata]):::process
    DirectAnswer --> FinalAnswer
    FinalAnswer --> User
    
    %% Feedback Loop (RLUF)
    User -.->|Feedback Utente 👍/👎| RLLogger
    RLLogger -.->|Aggiornamento Pesi| Router
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


