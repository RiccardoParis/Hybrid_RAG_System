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
    Phase 1: Generates 250 simulated queries via batch invocation of LLama-3.3-70B and saves the JSON output.
    """
    print("[Auto Warmup] Starting batch synthetic query generation. Fetching schemas...")
    
    schemas = get_detailed_schemas()
    
    llm = ChatGroq(
        temperature=0.7, 
        groq_api_key=GROQ_API_KEY, 
        model_name="openai/gpt-oss-120b"
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
        print(f"[Auto Warmup] Generating 50 questions for category '{category}'...")
        
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
            print(f"[Auto Warmup] Success for '{category}': {len(queries_list)} questions extracted.")
            
        except json.JSONDecodeError as e:
            print(f"[Auto Warmup] Critical JSON parsing error for category '{category}': {e}")
        except Exception as e:
            print(f"[Auto Warmup] An error occurred for category '{category}': {e}")

    # Save to disk
    project_root = os.path.dirname(os.path.dirname(__file__))
    data_dir = os.path.join(project_root, "data")
    os.makedirs(data_dir, exist_ok=True)
    
    file_path = os.path.join(data_dir, "synthetic_warmup_queries.json")
    
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(all_queries, f, indent=4, ensure_ascii=False)
        
    print(f"[Auto Warmup] Generation successfully completed! Saved {len(all_queries)} queries in: {file_path}")

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
        
        # Energy weight definitions per class
        cost_map = {
            "no_retrieval": 35,
            "vector": 400,
            "graph": 600,
            "sql": 800,
            "multi": 1500
        }
        
        # Select arm forcing entirely random exploration (epsilon=1.0)
        # ensuring the initial dataset uniformly covers all arms.
        action_name, is_exploration = bandit_router.choose_arm(
            query=query, 
            epsilon=1.0, 
            vector_meta=schemas['vector'], 
            graph_meta=schemas['graph'], 
            sql_meta=schemas['sql']
        )
        
        # Calculate simulated cost based on Ground Truth to heavily condition Reinforcement Learning
        simulated_cost = cost_map.get(action_name, 150)
        
        # Log the interaction in the Postgres DB (user_reward = NULL initially).
        log_id = log_interaction(query=query, chosen_arm=action_name, token_cost=simulated_cost)
        
        # ORACLE Logic: Immediate evaluation using expected_route as Ground Truth
        if action_name == expected_route:
            update_reward(log_id, 1)
            eval_str = "CORRECT (+1)"
        else:
            update_reward(log_id, 0)
            eval_str = "INCORRECT (0)"
        
        # Truncate query for clean logs
        short_query = (query[:40] + '...') if len(query) > 40 else query
        print(f"[Warmup {i:03d}/{len(queries_list)}] Inserted query: '{short_query}' -> Explored arm: {action_name} | Truth: {expected_route} | Outcome: {eval_str} (Log ID: {log_id})")
        
    print("[Auto Warmup] Phase 2 completed! The rl_logs table is populated with oracle evaluations and ready for training.")

if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(__file__))
    file_path = os.path.join(project_root, "data", "synthetic_warmup_queries.json")
    
    # Run the automatic end-to-end pipeline
    if not os.path.exists(file_path):
        generate_synthetic_queries()
        
    if os.path.exists(file_path):
        simulate_routing_interactions()