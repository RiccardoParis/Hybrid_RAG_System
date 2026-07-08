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
        "no_retrieval": "domande di conversazione generale, saluti, questioni puramente etiche o completamente fuori dominio.",
        "vector": "Domande discorsive, ricche di termini medici complessi, che chiedono spiegazioni, meccanismi d'azione o riassunti clinici (es. 'Qual è la farmacocinetica esatta di...', 'Illustra come viene metabolizzato...'). NON menzionare mai tabelle, grafi o conteggi.",
        "graph": "Domande che esplorano esplicitamente i legami, i percorsi e le interazioni dirette tra concetti (es. 'Quali sono tutte le interazioni note tra X e Y?', 'Mostrami la catena di effetti collaterali collegati a...'). Mimetizza la domanda in un linguaggio naturale, senza usare la parola 'grafo' o 'nodi'.",
        "sql": "Domande puramente quantitative ed estrattive (es. 'Quantifica il totale dei soggetti sottoposti al test NCT...', 'Fornisci l'elenco esatto delle aziende coinvolte nello studio...'). È SEVERAMENTE VIETATO usare la parola 'tabella', 'database' o 'colonna'. La domanda deve sembrare posta da un medico.",
        "multi": "Domande molto sfaccettate in cui l'utente richiede esplicitamente di calcolare/estrarre dei numeri E contemporaneamente spiegare i meccanismi biologici sottostanti (es. 'Quanti trial di fase 3 esistono per l'Aspirina, e quali sono i meccanismi molecolari con cui agisce sui recettori?')."
    }
    
    for category in categories:
        print(f"[Auto Warmup] Generazione di 50 domande per la categoria '{category}' in corso...")
        
        instruction = category_instructions.get(category, "")
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Sei un creatore di dataset per l'addestramento di un sistema di Information Retrieval medico.
Il tuo compito è generare 50 domande simulate in lingua italiana basandoti rigorosamente sugli schemi forniti.

Devi generare le domande ESCLUSIVAMENTE per la categoria: '{category}'.
Descrizione della categoria: {instruction}

OUTPUT FORMAT OBLIGATORIO:
Devi restituire ESCLUSIVAMENTE un singolo array JSON contenente le 50 domande. Non usare markdown (vietato ```json). Inizia con [ e finisci con ].
Esempio:
[
  {{"query": "Testo della domanda 1", "expected_route": "{category}"}},
  {{"query": "Testo della domanda 2", "expected_route": "{category}"}}
]

SCHEMI A DISPOSIZIONE:
[VECTOR DB]
{vector_meta}

[GRAPH DB]
{graph_meta}

[SQL DB]
{sql_meta}
"""),
            ("user", "Genera il JSON con 50 domande per la categoria '{category}'.")
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
