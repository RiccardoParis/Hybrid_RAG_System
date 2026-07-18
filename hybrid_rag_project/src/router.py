import re
import json
from typing import TypedDict, List, Any, Annotated
import operator
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.callbacks import BaseCallbackHandler
from config import GROQ_API_KEY

LLM_PRICING = {
    "llama-3.1-8b-instant": {"input": 0.05 / 1e6, "output": 0.08 / 1e6},
    "llama-3.3-70b-versatile": {"input": 0.59 / 1e6, "output": 0.79 / 1e6},
    "qwen/qwen3-32b": {"input": 0.29 / 1e6, "output": 0.59 / 1e6},
    "meta-llama/llama-4-scout-17b-16e-instruct": {"input": 0.11 / 1e6, "output": 0.34 / 1e6}
}

class GroqTokenCallback(BaseCallbackHandler):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.input_tokens = 0
        self.output_tokens = 0
        self.total_cost = 0.0

    def on_llm_end(self, response, **kwargs):
        if response.generations:
            for gen in response.generations[0]:
                if hasattr(gen, 'message') and hasattr(gen.message, 'response_metadata'):
                    usage = gen.message.response_metadata.get('token_usage', {})
                    inp = usage.get('prompt_tokens', 0)
                    out = usage.get('completion_tokens', 0)
                    self.input_tokens += inp
                    self.output_tokens += out
                    
                    pricing = LLM_PRICING.get(self.model_name, {"input": 0.0, "output": 0.0})
                    self.total_cost += (inp * pricing["input"]) + (out * pricing["output"])

# Import dei retriever e moduli di sistema
from vector_retriever import VectorRetriever
from graph_retriever import GraphRetriever
from sql_retriever import SQLRetriever
from rl_router import RLBanditRouter
from schema_extractor import get_compact_schemas, get_detailed_schemas
from rl_logger import log_interaction

# 1. Definizione dello stato aggiornata
class AgentState(TypedDict):
    query: str
    vector_query: str
    sql_query: str
    graph_query: str
    chosen_arm: str
    log_id: int
    routing_decision: List[str]
    ns_ids: List[str]
    vector_results: Any
    graph_result: Any
    lookup_results: dict
    sql_context: str
    final_answer: str
    input_tokens: Annotated[int, operator.add]
    output_tokens: Annotated[int, operator.add]
    total_cost: Annotated[float, operator.add]

# 2. Inizializzazioni
vector_retriever = VectorRetriever(collection_name="hybrid_rag", model_name="intfloat/multilingual-e5-base")
graph_retriever = GraphRetriever()
sql_retriever=SQLRetriever()
bandit_router = RLBanditRouter()

import json

# 3. Risoluzione Multi-Source usando gli Schemi Dettagliati
def resolve_multi_source(query: str, schemas: str):
    llm = ChatGroq(temperature=0, groq_api_key=GROQ_API_KEY, model_name="llama-3.1-8b-instant")
    prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a Query Decomposition Master for a Multi-Source RAG system. 
The user's query requires information from different databases. Your task is to break down the query into specific SUB-QUESTIONS, assigning each to the correct database based on the provided schemas.

STRICT RULES:
1. Output EXCLUSIVELY a flat JSON dictionary. DO NOT create nested dictionaries.
2. The allowed keys are strictly: "vector", "sql", "graph".
3. The value for each key MUST BE A SIMPLE TEXT STRING (e.g., "How many patients are enrolled?"). 
4. DO NOT write code (no SQL, no Cypher) in the values, write only NATURAL LANGUAGE questions.

Database Schemas:
{schemas}"""),
      ("user", "{query}")
    ])
    try:
        res = (prompt | llm).invoke({"schemas": schemas, "query": query})
        usage = res.response_metadata.get('token_usage', {})
        inp = usage.get('prompt_tokens', 0)
        out = usage.get('completion_tokens', 0)
        pricing = LLM_PRICING.get("llama-3.1-8b-instant", {"input": 0.0, "output": 0.0})
        cost = (inp * pricing["input"]) + (out * pricing["output"])
        print(f"[Router - DEBUG] Costo per resolve_multi_source: ${cost:.5f}")
        content = res.content.strip()
        
        match = re.search(r'\{.*\}', content, re.DOTALL)
        if match:
            try:
                parsed_dict = json.loads(match.group(0))
                return parsed_dict, inp, out, cost
            except json.JSONDecodeError:
                pass
        
        # Fallback di sicurezza: se fallisce, restituisce la query originale per entrambi
        return {"vector": query, "graph": query}, inp, out, cost
        
    except Exception as e:
        print(f"[Router] Errore in resolve_multi_source: {e}")
        return {"vector": query, "graph": query}, 0, 0, 0.0

# 4. Il Nodo Decisionale (Cuore del RL)
def parse_query_node(state: AgentState):
    query = state["query"]
    
    # Estrazione schemi compatti dalla cache
    compact_schemas = get_compact_schemas()
    
    # Scelta del braccio tramite DistilBERT
    action_name, is_expl = bandit_router.choose_arm(
        query, epsilon=0.1, 
        vector_meta=compact_schemas["vector"], 
        graph_meta=compact_schemas["graph"], 
        sql_meta=compact_schemas["sql"]
    )
    
    print(f"[Router] Braccio scelto: {action_name} (Exploration: {is_expl})")
    
    # Registrazione su PostgreSQL con token_cost iniziale a 0
    log_id = log_interaction(query, action_name)
    
    routes = []
    inp, out, cost = 0, 0, 0.0
    state_updates = {}
    
    if action_name == "multi":
        detailed_schemas = get_detailed_schemas()
        schemas_str = f"Vector: {detailed_schemas['vector']}\nGraph: {detailed_schemas['graph']}\nSQL: {detailed_schemas['sql']}"
        
        # multi_decision ora è un DIZIONARIO (es. {"vector": "sotto-domanda", "sql": "sotto-domanda"})
        multi_decision, inp, out, cost = resolve_multi_source(query, schemas_str)
        
        for db_name, sub_query in multi_decision.items():
            if db_name in ["vector", "graph", "sql"]:
                routes.append(f"{db_name}_search")
                state_updates[f"{db_name}_query"] = sub_query
                
        if not routes: 
            routes.append("vector_search")
            state_updates["vector_query"] = query
    elif action_name == "no_retrieval":
        routes = ["direct_answer"]
    else:
        routes = [f"{action_name}_search"]
        
    return {"chosen_arm": action_name, "log_id": log_id, "routing_decision": routes, "input_tokens": inp, "output_tokens": out, "total_cost": cost, **state_updates}

def vector_search_node(state: AgentState):
    """Esegue la ricerca sul vector store."""
    original_query = state.get("vector_query") or state["query"]
    
    # FIX PER IL MODELLO E5: Aggiungiamo il prefisso obbligatorio per le domande
    e5_query = f"query: {original_query}"
    
    print(f"[Router] Esecuzione Vector Search per: {e5_query}")
    try:
        # Passiamo la query formattata a Qdrant
        docs = vector_retriever.search(e5_query)
        # Estraiamo il testo dai documenti per passarlo all'LLM
        results = [doc.page_content for doc in docs]
    except Exception as e:
        print(f"[Router] Errore Vector Search: {e}")
        results = [f"Errore Vector Search: {e}"]
    return {"vector_results": results}

def graph_search_node(state: AgentState):
    """Esegue la ricerca sul knowledge graph."""
    cb = GroqTokenCallback(model_name="llama-3.1-8b-instant")
    query = state.get("graph_query") or state["query"]
    print(f"[Router] Esecuzione Graph Search per: {query}")
    try:
        # Esegue una query con LangChain GraphCypherQAChain
        result = graph_retriever.ask(query, callbacks=[cb])
        inp = cb.input_tokens
        out = cb.output_tokens
        cost = cb.total_cost
        # Per assicurarci che gli ID entrino correttamente nello stato in base ai pattern LangGraph
        found_ids = list(set(re.findall(r'\bns/[\w-]+\b', str(result))))
    except Exception as e:
        print(f"[Router] Errore Graph Search: {e}")
        result = {"error": f"Errore Graph Search: {e}"}
        found_ids = []
        inp, out, cost = 0, 0, 0.0
        
    return {"graph_result": result, "ns_ids": found_ids, "input_tokens": inp, "output_tokens": out, "total_cost": cost}

def sql_search_node(state: AgentState):
    """Executes the search on the SQL database using the isolated module."""
    cb = GroqTokenCallback(model_name="llama-3.1-8b-instant")
    query = state.get("sql_query") or state["query"]
    print(f"[Router] Executing SQL Search for: {query}")
    
    try:
        result = sql_retriever.ask(query, callbacks=[cb])
        inp = cb.input_tokens
        out = cb.output_tokens
        cost = cb.total_cost
        
    except Exception as e:
        print(f"[Router] SQL Search Error: {e}")
        result = f"SQL Search Error: {e}"
        inp, out, cost = 0, 0, 0.0
        
    return {"sql_context": result, "input_tokens": inp, "output_tokens": out, "total_cost": cost}

def route_after_graph(state: AgentState):
    """
    Analizza la stringa dentro state['graph_result']. 
    Se trova ID, li salva e va a lookup, altrimenti va a late_fusion.
    """
    graph_res_str = str(state.get("graph_result", ""))
    found_ids = list(set(re.findall(r'\bns/[\w-]+\b', graph_res_str)))
    
    if found_ids:
        state['ns_ids'] = found_ids
        return "lookup"
        
    return "late_fusion"

def lookup_node(state: AgentState):
    """Esegue la funzione di lookup se sono stati trovati degli ID."""
    ns_ids = state.get("ns_ids", [])
    print(f"[Router] Esecuzione Lookup per gli IDs: {ns_ids}")
    
    lookup_results = {}
    if ns_ids:
        try:
            # Esegue la query Cypher usando la connessione di GraphRetriever
            cypher_query = "MATCH (n) WHERE n.center IN $ids RETURN n.center AS id, n.title AS title, labels(n) AS type"
            res = graph_retriever.graph.query(cypher_query, {"ids": ns_ids})
            
            for r in res:
                lookup_results[r['id']] = r['title']
                
        except Exception as e:
            print(f"[Router] Errore Lookup: {e}")
            
    return {"lookup_results": lookup_results}

def late_fusion_node(state: AgentState):
    """
    Raccoglie i risultati di Vector, Graph e Lookup.
    Usa un LLM per fonderli in una risposta unificata.
    """
    print("[Router] Esecuzione Late Fusion Node")
    
    # Inizializza LLM
    llm = ChatGroq(
        temperature=0, 
        groq_api_key=GROQ_API_KEY, 
        model_name="llama-3.3-70b-versatile"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an analytical medical assistant. You will receive text snippets (from PubMed) and/or records extracted from a relational database (e.g., clinical trials).

CRITICAL INSTRUCTIONS FOR SQL CONTEXT:
- If the user asked for a calculation, count, sum, or average, the SQL context will directly provide the exact computed number. Do NOT attempt to recount or recalculate; simply trust the provided number and formulate a clear, natural language answer.
- If the user asked for a semantic summary, the SQL context will provide raw text rows. Read them carefully and summarize them as requested.
- TRUST THE FILTERS: If the user's question contains specific filters (e.g., "with exactly 300 patients" or "sponsored by Pfizer"), assume the SQL database has already applied these filters perfectly. Do NOT refuse to answer just because the filtering values (like the number 300) are not explicitly printed in the SQL context.
- If the SQL context provides a list of raw database tuples (e.g., [('Item 1',), ('Item 2',)]), YOU MUST NEVER print the Python programming symbols like brackets [], parentheses (), or trailing commas. Extract the pure text and format it as a clean, human-readable bulleted list.

If the information to answer is not present in ANY of the provided contexts, explicitly state that the available data does not contain the answer and STOP. Do not invent or use prior knowledge.

You are given independent context sources related to the user's question:

GraphRAG context: {graph_res}

Graph lookup (JSON ID-to-Name mapping): {lookup_res}

VectorRAG context: {vector_res}

SQL context: {sql_res}

Your task is to:
Read the GraphRAG context. If it contains raw IDs (starting with "ns/"), use the Graph lookup JSON mapping to translate them into human-readable names. Do not ignore the GraphRAG context, just translate its entities.
Merge the translated GraphRAG context, the VectorRAG context, and the SQL context to answer the user's question.

Instructions:
If both sources provide relevant and compatible information, merge them.
If only one source provides useful content, use that.
If the sources conflict, select the more specific or factual one.
Output only one unified, conversational answer.

CRITICAL STYLE RULE: You MUST hide your internal RAG process from the user. NEVER use words like "context", "SQL context", "tuple", "VectorRAG", "GraphRAG", "pipeline", or "source". 
Do NOT say things like "According to the SQL context..." or "The database provides a tuple...". 
Just state the synthesized facts directly and naturally."""),
        ("user", """Question: {query}
        
Vector Search Context:
{vector_res}

Graph Search Context:
{graph_res}

Lookup Context:
{lookup_res}

SQL Context:
{sql_res}
""")
    ])
    
    chain = prompt | llm
    
    # Prepara input per LLM
    query = state.get("query", "")
    vector_res = state.get("vector_results", "")
    graph_res = state.get("graph_result", "")
    lookup_res = state.get("lookup_results", {})
    sql_res = state.get("sql_context", "")
    
    response = chain.invoke({
        "query": query,
        "vector_res": str(vector_res),
        "graph_res": str(graph_res),
        "lookup_res": str(lookup_res),
        "sql_res": str(sql_res)
    })
    
    usage = response.response_metadata.get('token_usage', {})
    inp = usage.get('prompt_tokens', 0)
    out = usage.get('completion_tokens', 0)
    pricing = LLM_PRICING.get("llama-3.3-70b-versatile", {"input": 0.0, "output": 0.0})
    cost = (inp * pricing["input"]) + (out * pricing["output"])
    
    return {"final_answer": response.content, "input_tokens": inp, "output_tokens": out, "total_cost": cost}

def direct_answer_node(state: AgentState):
    """Nodo per rispondere direttamente senza RAG (selezionato dal RL Router quando non serve interrogare DB)."""
    print("[Router] Esecuzione Direct Answer Node (No Retrieval)")
    query = state.get("query", "")
    
    llm = ChatGroq(
        temperature=0, 
        groq_api_key=GROQ_API_KEY, 
        model_name="llama-3.1-8b-instant"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Answer the user's question as directly and concisely as possible."),
        ("user", "{query}")
    ])
    
    chain = prompt | llm
    try:
        response = chain.invoke({"query": query})
        usage = response.response_metadata.get('token_usage', {})
        inp = usage.get('prompt_tokens', 0)
        out = usage.get('completion_tokens', 0)
        pricing = LLM_PRICING.get("llama-3.1-8b-instant", {"input": 0.0, "output": 0.0})
        cost = (inp * pricing["input"]) + (out * pricing["output"])
        return {"final_answer": response.content, "input_tokens": inp, "output_tokens": out, "total_cost": cost}
    except Exception as e:
        print(f"[Router] Errore Direct Answer: {e}")
        return {"final_answer": "Si è verificato un errore.", "input_tokens": 0, "output_tokens": 0, "total_cost": 0.0}

# 1. Inizializza il StateGraph
workflow = StateGraph(AgentState)

# 2. Aggiunge i nodi
workflow.add_node("parse_query", parse_query_node)
workflow.add_node("vector_search", vector_search_node)
workflow.add_node("graph_search", graph_search_node)
workflow.add_node("sql_search", sql_search_node)
workflow.add_node("lookup", lookup_node)
workflow.add_node("late_fusion", late_fusion_node)
workflow.add_node("direct_answer", direct_answer_node)

# 3. Definisce le connessioni base
workflow.add_edge(START, "parse_query")

# 5. L'Edge Condizionale Semplificato
def route_after_parse(state: AgentState):
    # Il nodo precedente ha già preso tutte le decisioni!
    return state["routing_decision"]

workflow.add_conditional_edges(
    "parse_query",
    route_after_parse,
    ["vector_search", "graph_search", "sql_search", "direct_answer"]
)

# vector_search e sql_search vanno dritti a late_fusion
workflow.add_edge("vector_search", "late_fusion")
workflow.add_edge("sql_search", "late_fusion")

# graph_search usa l'edge condizionale
workflow.add_conditional_edges(
    "graph_search",
    route_after_graph,
    {
        "lookup": "lookup",
        "late_fusion": "late_fusion"
    }
)

# Se lookup viene eseguito, confluisce in late_fusion
workflow.add_edge("lookup", "late_fusion")

# I nodi di generazione terminano l'esecuzione
workflow.add_edge("late_fusion", END)
workflow.add_edge("direct_answer", END)

# 7. Compila il grafo in un'applicazione eseguibile
app = workflow.compile()

if __name__ == "__main__":
    # Test 1: Query senza ID "ns/" -> Non eseguirà la lookup
    print("--- Test 1 ---")
    state1 = app.invoke({"query": "Qual è il significato della vita?"})
    print(f"Stato Finale 1: {state1}\n")
    
    # Test 2: Query con ID "ns/" -> Eseguirà la lookup in parallelo agli altri
    print("--- Test 2 ---")
    state2 = app.invoke({"query": "Dammi i log dell'istanza ns/server-prod-01 e del nodo ns/router-02"})
    print(f"Stato Finale 2: {state2}\n")
