# Hybrid Multi-Source RAG Agent 🧠🔗

Questo progetto implementa un'architettura **Retrieval-Augmented Generation (RAG) Ibrida e Multi-Sorgente**. Il sistema supera i limiti dei tradizionali RAG basati esclusivamente su vettori (VectorRAG), integrando in parallelo e dinamicamente l'accesso a tre diverse tipologie di database:
1. **Vector DB (Qdrant):** Per la ricerca semantica su documenti testuali non strutturati (es. paper accademici, manuali in PDF/TXT).
2. **Graph DB (Neo4j):** Per il ragionamento topologico e relazionale su entità fortemente connesse (Knowledge Graphs).
3. **Relational DB (PostgreSQL):** Per calcoli deterministici, aggregazioni matematiche e dati strutturati tabellari (es. cataloghi, prezzi).

Il cuore del sistema è un **Intelligent Router** basato su LangGraph e Groq (Llama-3.1-8b / 3.3-70b), capace di analizzare strutturalmente la domanda dell'utente, interrogare in parallelo i database necessari, risolvere eventuali ID grezzi (Graph Lookup) e fondere i contesti in una singola risposta naturale tramite una strategia di **Late Fusion**.

---

## 🛠 Prerequisiti

Per eseguire il progetto localmente, assicurati di avere installato:
- **Python 3.10+**
- **Docker e Docker Compose** (per sollevare i database)
- Una chiave API valida per [Groq](https://console.groq.com/) (modelli Llama 3)

---

## 🚀 Installazione

**1. Clona la repository**
```bash
git clone [https://github.com/TuoUsername/Hybrid_RAG_System.git](https://github.com/TuoUsername/Hybrid_RAG_System.git)
cd Hybrid_RAG_System/graph_rag_project
2. Crea e attiva un ambiente virtuale (consigliato)

Bash
python -m venv venv
# Su Windows:
venv\Scripts\activate
# Su macOS/Linux:
source venv/bin/activate
3. Installa le dipendenze Python
Il sistema utilizza i modelli di embedding di Hugging Face BAAI/bge-base-en-v1.5 per una vettorizzazione locale senza bisogno di server esterni (es. Ollama).

Bash
pip install -r requirements.txt
4. Configurazione Variabili d'Ambiente
Modifica il file .env nella root del progetto (dove si trova docker-compose.yml) e inserisci le seguenti configurazioni:

Snippet di codice
# Qdrant Configuration
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=your_qdrant_api_key_here

# Neo4j Configuration
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=Password

# PostgreSQL Configuration
POSTGRES_URI=postgresql://postgres:Password@localhost:5432/used_cars_db

# Groq Configuration
GROQ_API_KEY=inserisci_qui_la_tua_chiave_api_groq
🐳 Avvio dell'Infrastruttura Database
L'intera infrastruttura di persistenza è containerizzata. Per avviare contemporaneamente Qdrant, Neo4j, PostgreSQL e PgAdmin, apri il terminale nella root del progetto ed esegui:

Bash
docker compose up -d
Accesso ai servizi in background:

Neo4j Browser: http://localhost:7474 (Login: neo4j / Password)

PgAdmin (per Postgres): http://localhost:5050 (Login: admin@admin.com / Password)

Qdrant API: http://localhost:6333

📥 Ingestione dei Dati
Prima di interrogare il sistema, i database devono essere popolati. Puoi farlo in due modi:

Opzione 1: Tramite Interfaccia Web (Dashboard)
Avvia l'app Streamlit e utilizza la sidebar laterale per caricare file .pdf, .txt o .json. Il sistema li indirizzerà automaticamente al Vector DB o al Graph DB.

Opzione 2: Tramite Script Bulk
Inserisci i tuoi documenti nella cartella data/texts/ e i tuoi grafi JSON in data/graphs/. Quindi esegui:

Bash
python src/bulk_ingest.py
Opzione 3: Dati SQL di Esempio
Per popolare il database relazionale con il dataset di test sulle automobili, esegui:

Bash
python src/ingest_sql_sample.py
🖥 Avvio dell'Applicazione
Una volta che l'infrastruttura è avviata e i dati sono stati ingeriti, avvia l'interfaccia utente basata su Streamlit:

Bash
streamlit run src/app.py
