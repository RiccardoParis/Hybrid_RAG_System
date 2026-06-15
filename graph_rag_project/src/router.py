import re
from typing import TypedDict, List, Any
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from config import GROQ_API_KEY

# Importiamo i retriever. 
# Nota: L'inizializzazione effettiva richiede DB/Ollama attivi.
from vector_retriever import VectorRetriever
from graph_retriever import GraphRetriever

# Definizione dello stato
class AgentState(TypedDict):
    query: str
    routing_decision: dict
    ns_ids: List[str]
    vector_results: Any
    graph_result: Any
    lookup_results: dict
    sql_context: str
    final_answer: str

# Inizializziamo i retriever
# Passiamo esplicitamente nomic-embed-text se impostato in ingest.py
vector_retriever = VectorRetriever(collection_name="hybrid_rag", model_name="nomic-embed-text")
graph_retriever = GraphRetriever()

# Inizializzazione SQL DB
import os
from dotenv import load_dotenv
from langchain_community.utilities import SQLDatabase
from langchain_community.tools import QuerySQLDatabaseTool
from langchain_classic.chains import create_sql_query_chain

load_dotenv()
postgres_uri = os.getenv("POSTGRES_URI", "")
if postgres_uri and "TUAPASSWORD" not in postgres_uri:
    db = SQLDatabase.from_uri(postgres_uri)
    llm_sql = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    execute_query = QuerySQLDatabaseTool(db=db)
    write_query = create_sql_query_chain(llm_sql, db)
    
    def clean_sql_output(text: str) -> str:
        # Estrae dai blocchi markdown sql
        match = re.search(r"```sql(.*?)```", text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        
        # Estrae da blocchi markdown generici
        match = re.search(r"```(.*?)```", text, re.DOTALL)
        if match:
            return match.group(1).strip()
            
        # Fallback su SQLQuery:
        if "SQLQuery:" in text:
            return text.split("SQLQuery:")[1].strip()
            
        return text.strip()
        
    sql_chain = write_query | clean_sql_output | execute_query
else:
    sql_chain = None

import json

def route_query(question: str) -> dict:
    """Classifica la domanda e decide quale database interrogare."""
    llm = ChatGroq(
        temperature=0, 
        groq_api_key=GROQ_API_KEY, 
        model_name="llama-3.1-8b-instant"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an intelligent routing agent for a multi-database Retrieval-Augmented Generation system.
Analyze the user's question and determine the best database to query.

RULES:

Output ONLY a valid JSON object with exactly this structure:
{{"vector": boolean, "graph": boolean, "sql": boolean}}

Imposta "sql": true se la domanda richiede calcoli, conteggi, medie, filtri esatti su dati strutturati o aggregazioni (es. 'prezzo medio', 'quante auto', 'lista dei concessionari').

Imposta "graph": true se la domanda riguarda le relazioni o le topologie tra entità (chi ha creato cosa, chi è connesso a chi).

Imposta "vector": true se la domanda cerca opinioni, descrizioni discorsive, riassunti o recensioni.

Puoi impostare più di un campo a true se la domanda è ibrida."""),
        ("user", "{question}")
    ])
    chain = prompt | llm
    try:
        response = chain.invoke({"question": question})
        content = response.content.strip()
        match = re.search(r'\{.*\}', content, re.DOTALL)
        if match:
            decision = json.loads(match.group(0))
        else:
            decision = json.loads(content)
        return decision
    except Exception as e:
        print(f"[Router] Errore nell'LLM Router ({e}). Fallback su VECTOR e GRAPH.")
        return {"vector": True, "graph": True, "sql": False}

def parse_query_node(state: AgentState):
    """Nodo pass-through per analizzare e instradare la query."""
    query = state["query"]
    decision = route_query(query)
    print(f"[Router] Decisione di instradamento: {decision}")
    return {"routing_decision": decision}

def vector_search_node(state: AgentState):
    """Esegue la ricerca sul vector store."""
    query = state["query"]
    print(f"[Router] Esecuzione Vector Search per: {query}")
    try:
        docs = vector_retriever.search(query)
        # Estraiamo il testo dai documenti per passarlo all'LLM
        results = [doc.page_content for doc in docs]
    except Exception as e:
        print(f"[Router] Errore Vector Search: {e}")
        results = [f"Errore Vector Search: {e}"]
    return {"vector_results": results}

def graph_search_node(state: AgentState):
    """Esegue la ricerca sul knowledge graph."""
    query = state["query"]
    print(f"[Router] Esecuzione Graph Search per: {query}")
    try:
        # Esegue una query con LangChain GraphCypherQAChain
        result = graph_retriever.ask(query)
        # Per assicurarci che gli ID entrino correttamente nello stato in base ai pattern LangGraph
        found_ids = list(set(re.findall(r'\bns/[\w-]+\b', str(result))))
    except Exception as e:
        print(f"[Router] Errore Graph Search: {e}")
        result = {"error": f"Errore Graph Search: {e}"}
        found_ids = []
    return {"graph_result": result, "ns_ids": found_ids}

def sql_search_node(state: AgentState):
    """Esegue la ricerca sul database SQL."""
    query = state["query"]
    print(f"[Router] Esecuzione SQL Search per: {query}")
    try:
        if sql_chain:
            result = sql_chain.invoke({"question": query})
        else:
            result = "Database SQL non configurato o credenziali assenti."
    except Exception as e:
        print(f"[Router] Errore SQL Search: {e}")
        result = f"Errore SQL Search: {e}"
    return {"sql_context": result}

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
        ("system", """You are given independent context sources related to the user's question:

GraphRAG context: {graph_res}

Graph lookup (JSON ID-to-Name mapping): {lookup_res}

VectorRAG context: {vector_res}

SQL context: {sql_res}

Your task is to:

Read the GraphRAG context. If it contains raw IDs (starting with "ns/"), use the Graph lookup JSON mapping to translate them into human-readable names. Do not ignore the GraphRAG context, just translate its entities.

Merge the translated GraphRAG context with the VectorRAG context to answer the user's question.

Instructions:

If both sources provide relevant and compatible information, merge them.

If only one source provides useful content, use that.

If the sources conflict, select the more specific or factual one.

Output only one unified answer. Do not mention GraphRAG, VectorRAG, or IDs."""),
        ("user", """Domanda: {query}
        
Contesto Vector Search:
{vector_res}

Contesto Graph Search:
{graph_res}

Contesto Lookup:
{lookup_res}

Contesto SQL:
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
    
    return {"final_answer": response.content}

# 1. Inizializza il StateGraph
workflow = StateGraph(AgentState)

# 2. Aggiunge i nodi
workflow.add_node("parse_query", parse_query_node)
workflow.add_node("vector_search", vector_search_node)
workflow.add_node("graph_search", graph_search_node)
workflow.add_node("sql_search", sql_search_node)
workflow.add_node("lookup", lookup_node)
workflow.add_node("late_fusion", late_fusion_node)

# 3. Definisce le connessioni base
workflow.add_edge(START, "parse_query")

# Instradamento dinamico basato sul Router (JSON)
def route_after_parse(state: AgentState):
    decision = state.get("routing_decision", {"vector": True, "graph": True, "sql": False})
    routes = []
    
    if decision.get("vector"):
        routes.append("vector_search")
    if decision.get("graph"):
        routes.append("graph_search")
    if decision.get("sql"):
        # NOTA: Assicurati di aggiungere e collegare un nodo 'sql_search' al grafo
        routes.append("sql_search")
        
    if not routes:
        routes.append("vector_search")
        
    return routes

workflow.add_conditional_edges(
    "parse_query",
    route_after_parse,
    ["vector_search", "graph_search", "sql_search"]
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

# Il nodo late_fusion termina l'esecuzione
workflow.add_edge("late_fusion", END)

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
