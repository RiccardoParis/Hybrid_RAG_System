import os
import re
import json
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()
from config import GROQ_API_KEY
from schema_extractor import get_detailed_schemas, get_compact_schemas
from rl_router import RLBanditRouter
from rl_logger import log_interaction

def generate_synthetic_queries():
    """
    Fase 1: Genera 250 query simulate invocando LLama-3.3-70B a batch e salva l'output JSON.
    """
    print("[Auto Warmup] Avvio generazione query sintetiche a batch. Acquisizione schemi...")
    
    schemas = get_detailed_schemas()
    
    llm = ChatGroq(
        temperature=0.7, 
        groq_api_key=GROQ_API_KEY, 
        model_name="llama-3.3-70b-versatile"
    )
    
    categories = ['no_retrieval', 'vector', 'graph', 'sql', 'multi']
    all_queries = []
    
    category_instructions = {
        "no_retrieval": "General conversation questions, greetings, purely ethical questions, or completely out-of-domain issues.",
        "vector": "Discursive questions, rich in complex medical terms, asking for explanations, mechanisms of action, or clinical summaries (e.g., 'What is the exact pharmacokinetics of...', 'Illustrate how it is metabolized...'). NEVER mention tables, graphs, or counts.",
        "graph": "Questions that explicitly explore links, pathways, and direct interactions between concepts (e.g., 'What are all the known interactions between X and Y?', 'Show me the chain of side effects linked to...'). Disguise the question in natural language, without using the word 'graph' or 'nodes'.",
        "sql": "Purely quantitative and extractive questions (e.g., 'Quantify the total number of subjects undergoing the NCT test...', 'Provide the exact list of companies involved in the study...'). It is STRICTLY FORBIDDEN to use the word 'table', 'database', or 'column'. The question must sound like it was asked by a medical doctor or researcher.",
        "multi": "Highly multifaceted questions where the user explicitly asks to calculate/extract numbers AND simultaneously explain the underlying biological mechanisms (e.g., 'How many phase 3 trials exist for Aspirin, and what are the molecular mechanisms by which it acts on receptors?')."
    }
    
    for category in categories:
        print(f"[Auto Warmup] Generazione di 50 domande per la categoria '{category}' in corso...")
        
        instruction = category_instructions.get(category, "")
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a dataset creator for training a medical Information Retrieval system.
Your task is to generate 50 simulated questions IN ENGLISH strictly based on the provided schemas.

You must generate the questions EXCLUSIVELY for the category: '{category}'.
Category description: {instruction}

MANDATORY OUTPUT FORMAT:
You must return EXCLUSIVELY a single JSON array containing the 50 questions. Do not use markdown (forbidden ```json). Start with [ and end with ].
Example:
[
  {{"query": "Question text 1", "expected_route": "{category}"}},
  {{"query": "Question text 2", "expected_route": "{category}"}}
]

AVAILABLE SCHEMAS:
[VECTOR DB]
{vector_meta}

[GRAPH DB]
{graph_meta}

[SQL DB]
{sql_meta}
"""),
            ("user", "Generate the JSON with 50 questions IN ENGLISH for the category '{category}'.")
        ])
        
        chain = prompt | llm
        
        try:
            response = chain.invoke({
                "category": category,
                "instruction": instruction,
                "vector_meta": schemas['vector'],
                "graph_meta": schemas['graph'],
                "sql_meta": schemas['sql']
            })
            
            content = response.content.strip()
            match = re.search(r'\[.*\]', content, re.DOTALL)
            if match:
                json_str = match.group(0)
            else:
                json_str = content
                
            queries_list = json.loads(json_str)
            all_queries.extend(queries_list)
            print(f"[Auto Warmup] Successo per '{category}': estratte {len(queries_list)} domande.")
            
        except json.JSONDecodeError as e:
            print(f"[Auto Warmup] Errore critico nel parsing JSON per la categoria '{category}': {e}")
        except Exception as e:
            print(f"[Auto Warmup] Si è verificato un errore per la categoria '{category}': {e}")

    # Salvataggio
    project_root = os.path.dirname(os.path.dirname(__file__))
    data_dir = os.path.join(project_root, "data")
    os.makedirs(data_dir, exist_ok=True)
    
    file_path = os.path.join(data_dir, "synthetic_warmup_queries.json")
    
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(all_queries, f, indent=4, ensure_ascii=False)
        
    print(f"[Auto Warmup] Generazione completata con successo! Salvate {len(all_queries)} query in: {file_path}")

def simulate_routing_interactions():
    """
    Fase 2: Simula il traffico sul router e popola la tabella rl_logs in PostgreSQL.
    """
    project_root = os.path.dirname(os.path.dirname(__file__))
    file_path = os.path.join(project_root, "data", "synthetic_warmup_queries.json")
    
    if not os.path.exists(file_path):
        print("[Auto Warmup] Errore: File JSON non trovato per la simulazione.")
        return
        
    print("[Auto Warmup] Caricamento query dal file JSON...")
    with open(file_path, "r", encoding="utf-8") as f:
        queries_list = json.load(f)
        
    from rl_logger import log_interaction, update_reward
    
    print("[Auto Warmup] Inizializzazione RL Router e recupero schemi compatti...")
    bandit_router = RLBanditRouter()
    schemas = get_compact_schemas()
    
    print(f"[Auto Warmup] Inizio Simulazione su {len(queries_list)} query (Esplorazione Casuale Epsilon=1.0)...")
    for i, item in enumerate(queries_list, start=1):
        query = item.get("query", "")
        # Dentro il ciclo in auto_warmup.py:
        expected_route = item.get("expected_route")
        
        # Definizione dei pesi energetici per classe
        cost_map = {
            "no_retrieval": 35,
            "vector": 400,
            "graph": 600,
            "sql": 800,
            "multi": 1500
        }
        
        # Sceglie l'azione forzando l'esplorazione totalmente casuale (epsilon=1.0)
        # in modo che il dataset iniziale copra uniformemente tutte le braccia.
        action_name, is_exploration = bandit_router.choose_arm(
            query=query, 
            epsilon=1.0, 
            vector_meta=schemas['vector'], 
            graph_meta=schemas['graph'], 
            sql_meta=schemas['sql']
        )
        
        # Calcola il costo simulato basato sul Ground Truth per condizionare pesantemente il Reinforcement Learning
        simulated_cost = cost_map.get(action_name, 150)
        
        # Registra l'interazione nel DB Postgres (user_reward = NULL inizialmente).
        log_id = log_interaction(query=query, chosen_arm=action_name, token_cost=simulated_cost)
        
        # Logica ORACLE: Valutazione immediata usando la expected_route come Ground Truth
        if action_name == expected_route:
            update_reward(log_id, 1)
            eval_str = "CORRETTO (+1)"
        else:
            update_reward(log_id, 0)
            eval_str = "ERRATO (0)"
        
        # Tronca la query per pulizia nei log
        short_query = (query[:40] + '...') if len(query) > 40 else query
        print(f"[Warmup {i:03d}/{len(queries_list)}] Inserita query: '{short_query}' -> Braccio esplorato: {action_name} | Truth: {expected_route} | Esito: {eval_str} (Log ID: {log_id})")
        
    print("[Auto Warmup] Fase 2 completata! Tabella rl_logs popolata con valutazioni oracolo e pronta per l'addestramento.")

if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(__file__))
    file_path = os.path.join(project_root, "data", "synthetic_warmup_queries.json")
    
    # Esegue la pipeline automatica end-to-end
    if not os.path.exists(file_path):
        generate_synthetic_queries()
        
    if os.path.exists(file_path):
        simulate_routing_interactions()
