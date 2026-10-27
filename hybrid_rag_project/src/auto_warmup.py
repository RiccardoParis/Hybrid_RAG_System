import os
import re
import json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()
from schema_extractor import get_detailed_schemas, get_compact_schemas
from rl_router import RLBanditRouter
from rl_logger import log_interaction

def generate_synthetic_queries():
    """
    Phase 1: Generates simulated queries via single-shot generation to maximize contextual 
    diversity and prevent repetitive batch outputs.
    """
    print("[Auto Warmup] Starting single-shot synthetic query generation. Fetching schemas...")
    
    schemas = get_detailed_schemas()
    
    # --- SANITY CHECK SCHEMI ---
    print("\n[Sanity Check] Verifico la lunghezza degli schemi estratti dai DB:")
    print(f" -> Vector Schema: {len(schemas['vector'])} caratteri")
    print(f" -> Graph Schema: {len(schemas['graph'])} caratteri")
    print(f" -> SQL Schema: {len(schemas['sql'])} caratteri")
    if len(schemas['sql']) < 50 or len(schemas['graph']) < 50:
        print("⚠️ ATTENZIONE: Uno degli schemi sembra sospettosamente corto. Verifica le connessioni ai DB.\n")
    else:
        print("✅ Schemi caricati correttamente.\n")
    # ---------------------------
    
    aws_api_key = os.getenv("AWS_MANTLE_API_KEY")
    aws_region = os.getenv("AWS_MANTLE_REGION", "us-east-1")
    
    if not aws_api_key:
        raise ValueError("ERRORE: AWS_MANTLE_API_KEY mancante nel file .env")
        
    # max_tokens aumentato per permettere l'output di 50 domande in un singolo JSON
    llm = ChatOpenAI(
        model="qwen.qwen3-coder-480b-a35b-instruct",
        temperature=0.8, # Temperatura leggermente alzata per favorire la creatività
        api_key=aws_api_key,
        base_url=f"https://bedrock-mantle.{aws_region}.api.aws/v1",
        max_tokens=4500 
    )
    
    categories = ['no_retrieval', 'vector', 'graph', 'sql', 'multi']
    all_queries = []
    
    category_instructions = {
        "no_retrieval": "CRITICAL: DO NOT USE ANY MEDICAL TERMS, DRUGS, OR CANCER REFERENCES. Generate strictly everyday small talk, weather inquiries, sports trivia, or Python programming questions.",
        "vector": "Discursive questions, rich in complex medical terms, asking for explanations or mechanisms of action. CRITICAL: NEVER mention tables, graphs, databases, or counts.",
        "graph": "Questions that explicitly explore biological links, pathways, and direct interactions between concepts. CRITICAL: NEVER use the words 'graph', 'node', 'edge', or 'network'. Disguise the question naturally.",
        "sql": "Purely quantitative and extractive questions (e.g., totals, exact lists, dates). CRITICAL: NEVER use the words 'table', 'database', 'column', or 'row'.",
        "multi": "Highly multifaceted questions where the user explicitly asks to calculate/extract numbers AND simultaneously explain the underlying biological mechanisms."
    }
    
    QUESTIONS_PER_CATEGORY = 50
    
    for category in categories:
        print(f"\n[Auto Warmup] Generating {QUESTIONS_PER_CATEGORY} questions for category '{category}' in a single shot...")
        instruction = category_instructions.get(category, "")
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a dataset creator for training a medical Information Retrieval system.
Your task is to generate EXACTLY {num_questions} simulated questions IN ENGLISH strictly based on the provided schemas.

You must generate the questions EXCLUSIVELY for the category: '{category}'.
Category description: {instruction}

CRITICAL DIVERSITY RULE:
Ensure maximum variance across the {num_questions} questions. Do not repeat the same drug names, diseases, or phrasing structures. 
At least 30% of the generated questions MUST be "trick questions" (Hard Negatives), using misleading vocabulary while still belonging strictly to '{category}'.

MANDATORY OUTPUT FORMAT:
Return EXCLUSIVELY a single JSON array containing the {num_questions} questions. Do not use markdown (no ```json).
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
            ("user", "Generate the JSON array with exactly {num_questions} completely unique questions IN ENGLISH for the category '{category}'.")
        ])
        
        chain = prompt | llm
        
        try:
            response = chain.invoke({
                "category": category,
                "instruction": instruction,
                "num_questions": QUESTIONS_PER_CATEGORY,
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
            print(f"  ✅ Success: {len(queries_list)} questions generated for '{category}'.")
            
        except json.JSONDecodeError as e:
            print(f"  ❌ JSON parsing error for '{category}': {e}.")
        except Exception as e:
            print(f"  ❌ General error for '{category}': {e}.")

    project_root = os.path.dirname(os.path.dirname(__file__))
    data_dir = os.path.join(project_root, "data")
    os.makedirs(data_dir, exist_ok=True)
    file_path = os.path.join(data_dir, "synthetic_warmup_queries.json")
    
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(all_queries, f, indent=4, ensure_ascii=False)
        
    print(f"\n[Auto Warmup] Generation successfully completed! Saved {len(all_queries)} queries in: {file_path}")

def simulate_routing_interactions():
    """
    Phase 2: Simulates router traffic and populates the rl_logs table in PostgreSQL.
    """
    project_root = os.path.dirname(os.path.dirname(__file__))
    file_path = os.path.join(project_root, "data", "synthetic_warmup_queries.json")
    
    if not os.path.exists(file_path):
        print("[Auto Warmup] Error: JSON file not found for simulation.")
        return
        
    print("[Auto Warmup] Loading queries from JSON file...")
    with open(file_path, "r", encoding="utf-8") as f:
        queries_list = json.load(f)
        
    from rl_logger import log_interaction, update_reward
    
    print("[Auto Warmup] Initializing RL Router and fetching compact schemas...")
    bandit_router = RLBanditRouter()
    schemas = get_compact_schemas()
    
    print(f"[Auto Warmup] Starting Simulation on {len(queries_list)} queries (Random Exploration Epsilon=1.0)...")
    for i, item in enumerate(queries_list, start=1):
        query = item.get("query", "")
        expected_route = item.get("expected_route")
        
        action_name, is_exploration = bandit_router.choose_arm(
            query=query, 
            epsilon=1.0, 
            vector_meta=schemas['vector'], 
            graph_meta=schemas['graph'], 
            sql_meta=schemas['sql']
        )
        
        log_id = log_interaction(query=query, chosen_arm=action_name)
        
        if action_name == expected_route:
            update_reward(log_id, 1)
            eval_str = "CORRECT (+1)"
        else:
            update_reward(log_id, 0)
            eval_str = "INCORRECT (0)"
        
        short_query = (query[:40] + '...') if len(query) > 40 else query
        print(f"[Warmup {i:03d}/{len(queries_list)}] Inserted query: '{short_query}' -> Explored arm: {action_name} | Truth: {expected_route} | Outcome: {eval_str} (Log ID: {log_id})")
        
    print("[Auto Warmup] Phase 2 completed! The rl_logs table is populated with oracle evaluations and ready for training.")

if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(__file__))
    file_path = os.path.join(project_root, "data", "synthetic_warmup_queries.json")
    
    # Se il file esiste, lo eliminiamo per forzare una rigenerazione pulita
    if os.path.exists(file_path):
        os.remove(file_path)
        
    generate_synthetic_queries()
    simulate_routing_interactions()