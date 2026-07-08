Markdown
# Hybrid Multi-Source RAG System 🧠🧬

An advanced Retrieval-Augmented Generation (RAG) architecture tailored for the medical domain. This system intelligently routes user queries across three distinct databases (Vector, Graph, and Relational) to synthesize highly accurate, hallucination-free answers using Large Language Models.

## 🏗️ System Architecture

The system utilizes an agentic workflow powered by **LangGraph** to process, route, and fuse information:
1. **Neural Router (DistilBERT):** Replaces rigid rules with an intent-recognition classification head. Trained via Curriculum Learning (SFT + RL) to map the semantic intent to the appropriate database using Compact Schemas.
2. **Qdrant (Vector DB):** Handles semantic searches across medical abstracts and scientific literature using cross-lingual embeddings (`multilingual-e5-base`).
3. **Neo4j (Knowledge Graph):** Manages multi-hop reasoning and ontological relationships (e.g., diseases, side effects, drugs) using Dynamic Schema-Augmented Cypher Generation.
4. **PostgreSQL (Relational DB):** Processes quantitative queries and exact data filtering (e.g., clinical trial enrollments) via the Table-Augmented Generation (TAG) paradigm. Tracks RL logs and Energy Footprint.
5. **Late Fusion Node:** An LLM (Llama-3) synthesizes the raw extracted data from all sources into a single, coherent natural language response.
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
Modify the .env file in the root directory (refer to .env.example):

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

co come riscrivere la sezione del README.md per renderla cristallina, includendo i riferimenti a Hetionet e valorizzando il lavoro di estrazione che hai fatto.

Sostituisci la vecchia sezione ## 💉 Medical Data Ingestion con questa versione dettagliata:

Markdown
## 💉 Pipeline ETL e Data Ingestion

Il sistema non utilizza un dataset statico preconfezionato, ma costruisce la sua base di conoscenza attraverso una vera e propria pipeline ETL in due fasi:

### Fase 1: Data Collection & Subgraph Extraction
Prima di popolare i database, è necessario raccogliere ed estrarre i dati grezzi dalle fonti ufficiali:

1. **Il Grafo Medico Base (Hetionet):** Scarica il file JSON originale di Hetionet v1.0 dal repository ufficiale e posizionalo nella cartella `data/raw/`:
   [Link per il download di Hetionet (hetionet-v1.0.json.bz2)](https://github.com/hetio/hetionet/tree/master/hetnet/json)
   
2. **Estrazione del Sottografo:** Estrai solo i nodi e gli archi rilevanti per il dominio del progetto (malattie, geni, farmaci, sintomi):
   ```bash
   python scripts/data_collection/extract_hetionet.py
Arricchimento Dati dal Web (API Fetching):
Utilizza gli script dedicati per interrogare le API pubbliche (es. PubMed, ClinicalTrials.gov) e raccogliere gli abstract vettoriali e i dati tabellari corrispondenti alle entità del grafo estrattto:

Bash
python scripts/data_collection/fetch_pubmed_api.py
python scripts/data_collection/fetch_medical_api.py
Fase 2: Bulk Ingestion nei 3 Database
Una volta generati i dataset elaborati nella cartella data/processed/, puoi avviare l'ingestione parallela. Questo script popolerà Qdrant (generando gli embeddings), Neo4j (creando nodi e relazioni Cypher) e PostgreSQL (creando le tabelle relazionali):

Bash
python src/medical_bulk_ingestion.py

🧠 Neural Router Training Pipeline
The router learns through Curriculum Learning. To initialize the brain of the system:

Adversarial Warmup: Generate a synthetic, highly deceptive dataset.

Bash
python src/auto_warmup.py
Supervised Fine-Tuning (SFT): Teach the model semantic accuracy.

Bash
python src/sft_trainer.py
Reinforcement Learning (RLUF): Optimize live weights based on User Feedback and real token costs (Energy Footprint).

Bash
python src/rl_trainer.py

💻 Usage
Launch the interactive Streamlit dashboard:

Bash
streamlit run src/app.py

🗺️ Roadmap & Future Work
[x] Implement Metadata-based Dynamic Routing.

[x] Integrate Reinforcement Learning from User Feedback (RLUF).

[x] Evaluate System Energy Footprint and token optimization.

[ ] Evaluate system on OTT-QA Benchmark using LLM-as-a-Judge (Relevance, Faithfulness, Fluency).
```

### Architettura di Sistema

Di seguito l'architettura del nostro RAG Ibrido Multi-Sorgente con Router Neurale:
![Architettura](Hybrid_RAG_Architecture.drawio.svg)
