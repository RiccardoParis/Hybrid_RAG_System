Markdown
# 🧠 Hybrid Multi-Agent RAG System for Medical Big Data

An advanced, multi-source Retrieval-Augmented Generation (RAG) architecture designed to query, route, and synthesize complex medical, biological, and clinical trial data. 

Unlike standard linear RAG pipelines, this system implements a **Reinforcement Learning-trained Semantic Router** to dynamically direct user queries across Vector, Relational (SQL), and Graph databases, concluding with a **Late Fusion** node for unified natural language generation.

![Architecture Diagram](Hybrid_RAG_Architecture.drawio.svg)

## ✨ Key Features

*   **Dynamic Semantic Routing (RL Bandit):** A DistilBERT-based router trained via Supervised Fine-Tuning (SFT) and Reinforcement Learning ($\epsilon$-greedy) to classify queries. It achieves a 60% baseline accuracy on complex domain disambiguation, optimizing API costs and latency.
*   **Multi-Source Retrieval:**
    *   **Vector Search (Qdrant):** For unstructured medical literature (PubMed).
    *   **Graph Search (Neo4j):** For topological queries and biomedical entity relationships (Hetionet).
    *   **SQL Search (PostgreSQL):** For exact aggregations and clinical trial metrics.
*   **Late Fusion Node:** Bypasses intermediate LLM summarization by injecting raw data tuples and JSONs directly into the final inference node, drastically reducing token costs and preventing "double-summarization" hallucinations.
*   **LLM-as-a-Judge Evaluation:** A custom, robust evaluation pipeline inspired by the *Ragas* framework, scoring the architecture on *Relevance*, *Fluency*, and *Faithfulness*.

---

## 📂 Repository Structure
```text

├── Hybrid_RAG_Architecture.drawio.svg
├── hybrid_rag_project/
│   ├── data/                           # (User-created) Raw datasets 
│   ├── docker-compose.yml              # Container orchestration for databases
│   ├── holdout_test_set.json           # Holdout data for Router Evaluation
│   ├── requirements.txt                # Python dependencies
│   ├── evaluation/                     # LLM-as-a-Judge and Benchmark scripts
│   │   ├── evaluate_spider.py          # Text-to-SQL specific evaluation
│   │   ├── migrate_spider.py           # Spider dataset ingestion tool
│   │   ├── rag_evaluator.py            # Robust LLM evaluator with backoff
│   │   └── run_evaluation_cycle.py     # Router accuracy testing
│   ├── scripts/data_collection/        # Automated ETL pipelines
│   │   ├── extract_hetionet.py         # Hetionet parsing
│   │   └── fetch_*.py                  # APIs for PubMed and WikiGraphs
│   └── src/                            # Core Application Code
│       ├── app.py                      # Streamlit Dashboard
│       ├── router.py                   # LangGraph state machine & Late Fusion
│       ├── rl_router.py                # Reinforcement Learning Bandit logic
│       ├── *_retriever.py              # Isolated modules for Qdrant, Neo4j, SQL
│       ├── medical_bulk_ingestion.py   # Bulk database populator
│       └── *_trainer.py                # SFT and RL training scripts

```

🚀 Setup & Installation
1. Clone the repository and install dependencies:
```Bash
git clone [https://github.com/yourusername/Hybrid_RAG_System.git](https://github.com/yourusername/Hybrid_RAG_System.git)
cd Hybrid_RAG_System/hybrid_rag_project
pip install -r requirements.txt
```
2. Environment Variables:
Create a .env file in the hybrid_rag_project directory and configure your API keys and database URIs:
```Ini, TOML
GROQ_API_KEY=your_groq_api_key
POSTGRES_URI=postgresql://user:password@localhost:5432/medical_db
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=your_qdrant_api_key
```

3. Launch Databases:
Spin up the local PostgreSQL, Neo4j, and Qdrant instances using Docker:
```Bash
docker-compose up -d
```
🗄️ Data Collection & Database Population
To replicate the academic environment, you must download the benchmark datasets and populate the three databases.

Step 1: Download External Datasets
Hetionet (For Neo4j): Download the JSON version of the Hetionet biological network from the official repository: [Hetionet v1.0 JSON](https://github.com/hetio/hetionet/tree/master/hetnet/json). Save it in data/.
Spider Dataset (For PostgreSQL/Evaluation): Download the Spider dataset (a large-scale complex text-to-SQL dataset) from Yale Lily: [Spider Dataset](https://yale-lily.github.io/spider). Extract it in data/spider/ only the directory database and the file dev.json.

Step 2: Fetch Unstructured Data
Run the data collection scripts to fetch medical literature and contextual graphs in this order:
```Bash
python scripts/data_collection/extract_hetionet.py
python scripts/data_collection/fetch_pubmed_api.py
python scripts/data_collection/fetch_medical_api.py
```

Step 3: Populate the Databases
Once the raw data is collected, run the bulk ingestion script to populate Qdrant, Neo4j, and PostgreSQL simultaneously:
```Bash
python src/medical_bulk_ingestion.py
```
🧠 Training the Semantic Router
The DistilBERT router must be trained before running the application:
Supervised Fine-Tuning (SFT): Bootstraps the router's semantic understanding.
```Bash
python src/sft_trainer.py
```
Reinforcement Learning (RL): Refines the policy using an $\epsilon$-greedy bandit approach based on simulated query rewards.
```Bash
python src/rl_trainer.py
```
🖥️ Running the Application
Start the main interactive RAG dashboard using Streamlit:
```Bash
streamlit run src/app.py
```
The dashboard allows you to query the system in natural language, view the chosen route (Vector, Graph, SQL, Multi), inspect the raw extracted contexts, and read the final synthesized answer.
📊 Evaluation & Reproducibility (Ablation Study)
The repository includes a comprehensive evaluation suite to measure routing accuracy, text-to-SQL performance, and generation quality.
1. Router Accuracy Testing:Evaluate the RL Router against the holdout test set to measure routing precision, latency, and token cost:
```Bash
python evaluation/run_evaluation_cycle.py --phase RL
```
2. Text-to-SQL Evaluation (Spider):Test the SQL retriever's accuracy on the complex cross-domain Spider benchmark:
```Bash
python evaluation/migrate_spider.py    # Migrates Spider schemas to PostgreSQL
python evaluation/evaluate_spider.py   # Runs the text-to-SQL evaluation
```
3. RAG Quality Metrics (LLM-as-a-Judge):Computes Relevance, Fluency, and Faithfulness using a strict, customized LLM judge (Llama-3.3-70B) to penalize hallucinations and code artifacts:
```Bash
python evaluation/rag_evaluator.py
```
Results will be saved as CSV files in the evaluation/ directory.
