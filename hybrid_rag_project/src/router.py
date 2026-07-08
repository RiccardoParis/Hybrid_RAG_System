import re
import json
from typing import TypedDict, List, Any
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from config import GROQ_API_KEY

# Import dei retriever e moduli di sistema
from vector_retriever import VectorRetriever
from graph_retriever import GraphRetriever
from rl_router import RLBanditRouter
from schema_extractor import get_compact_schemas, get_detailed_schemas
from rl_logger import log_interaction

# 1. Definizione dello stato aggiornata
class AgentState(TypedDict):
    query: str
    chosen_arm: str
    log_id: int
    routing_decision: List[str]
    ns_ids: List[str]
    vector_results: Any
    graph_result: Any
    lookup_results: dict
    sql_context: str
    final_answer: str
    total_tokens: int

# 2. Inizializzazioni
vector_retriever = VectorRetriever(collection_name="hybrid_rag", model_name="intfloat/multilingual-e5-base")
graph_retriever = GraphRetriever()
bandit_router = RLBanditRouter()

# Inizializzazione SQL DB (invariata rispetto al tuo codice originale)
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
    from langchain_core.prompts import PromptTemplate
    custom_sql_template = PromptTemplate.from_template(
"""Sei un estrattore di dati (Data Extractor) esperto in SQL (dialetto {dialect}).
Il tuo UNICO obiettivo è scrivere la query SQL per estrarre le righe necessarie.

REGOLE RIGIDE (PARADIGMA TAG):
1. ESTRAZIONE GREZZA: Usa solo semplici query SELECT. Anche se l'utente chiede "quanti", "somma" o "totale", tu NON usare funzioni di aggregazione (SUM, COUNT, AVG). Devi estrarre le singole righe, sarà il sistema successivo a contarle o sommarle.
2. CAMPI: Estrai sempre i campi descrittivi (es. nct_id, title, enrollment, sponsor.name).
3. TESTO: Usa SEMPRE ILIKE '%Nome%' quando filtri per stringhe testuali.
4. ID: Se l'utente fornisce un ID come 'NCT...', usa WHERE studies.nct_id = 'NCT...'.
5. LIMITE: Limita sempre la query a un massimo di {top_k} risultati.

OUTPUT:
Restituisci ESCLUSIVAMENTE la query SQL valida. Nessuna spiegazione, nessuna introduzione, nessun markdown. Solo il codice.

Tabelle a disposizione:
{table_info}

Domanda: {input}
SQLQuery:"""
)
    execute_query = QuerySQLDatabaseTool(db=db)
    write_query = create_sql_query_chain(llm_sql, db, prompt=custom_sql_template)
    
    def clean_sql_output(text: str) -> str:
        match = re.search(r"```sql(.*?)```", text, re.DOTALL | re.IGNORECASE)
        if match: return match.group(1).strip()
        match = re.search(r"```(.*?)```", text, re.DOTALL)
        if match: return match.group(1).strip()
        if "SQLQuery:" in text: return text.split("SQLQuery:")[1].strip()
        return text.strip()
        
    sql_chain = write_query | clean_sql_output | execute_query
else:
    sql_chain = None

import json

# 3. Risoluzione Multi-Source usando gli Schemi Dettagliati
def resolve_multi_source(query: str, schemas: str) -> List[str]:
    llm = ChatGroq(temperature=0, groq_api_key=GROQ_API_KEY, model_name="llama-3.1-8b-instant")
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Analizza la query dell'utente e gli schemi forniti. Devi decidere quali database interrogare. Restituisci SOLO un array JSON contenente una o più delle seguenti stringhe: ["vector", "graph", "sql"]. Non aggiungere altro.
Schemi:
{schemas}"""),
        ("user", "{query}")
    ])
    try:
        res = (prompt | llm).invoke({"schemas": schemas, "query": query})
        tokens = res.response_metadata.get('token_usage', {}).get('total_tokens', 0)
        print(f"[Router - DEBUG] Token spesi per resolve_multi_source: {tokens}")
        content = res.content.strip()
        
        match = re.search(r'\[.*\]', content, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
                
        # Fallback manuale basato su substring se il JSON fallisce o manca
        fallback_routes = []
        content_lower = content.lower()
        for db_name in ["vector", "graph", "sql"]:
            if db_name in content_lower:
                fallback_routes.append(db_name)
                
        return fallback_routes if fallback_routes else ["vector", "graph"]
        
    except Exception as e:
        print(f"[Router] Errore in resolve_multi_source: {e}")
        return ["vector", "graph"]

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
    log_id = log_interaction(query, action_name, token_cost=0)
    
    routes = []
    if action_name == "multi":
        detailed_schemas = get_detailed_schemas()
        schemas_str = f"Vector: {detailed_schemas['vector']}\nGraph: {detailed_schemas['graph']}\nSQL: {detailed_schemas['sql']}"
        multi_decision = resolve_multi_source(query, schemas_str)
        for db_name in ["vector", "graph", "sql"]:
            if db_name in multi_decision: routes.append(f"{db_name}_search")
        if not routes: routes.append("vector_search")
    elif action_name == "no_retrieval":
        routes = ["direct_answer"]
    else:
        routes = [f"{action_name}_search"]
        
    return {"chosen_arm": action_name, "log_id": log_id, "routing_decision": routes, "total_tokens": 0}

def vector_search_node(state: AgentState):
    """Esegue la ricerca sul vector store."""
    original_query = state["query"]
    
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
        ("system", """Sei un assistente medico analitico. Riceverai dei frammenti di testo (da PubMed) e/o una lista di record grezzi estratti da un database relazionale (es. trial clinici).

Se l'utente ti chiede un calcolo, un conteggio o una somma, DEVI farla tu stesso leggendo attentamente i record SQL forniti nel contesto.

Spiega sempre il tuo ragionamento. Ad esempio: 'Ho trovato X studi. Lo studio A ha Y pazienti, lo studio B ha Z pazienti, per un totale di W pazienti.'

Se le informazioni per rispondere non sono presenti in NESSUNO dei contesti forniti, dichiara esplicitamente che i dati a tua disposizione non contengono la risposta e FERMATI. Non inventare o usare conoscenze pregresse.

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
Do NOT explain that you are merging sources. Just state the synthesized facts directly and naturally (e.g., simply say "Nel database ci sono in totale 6 auto usate.")."""),
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
    
    tokens = response.response_metadata.get('token_usage', {}).get('total_tokens', 0)
    new_total = state.get("total_tokens", 0) + tokens
    
    return {"final_answer": response.content, "total_tokens": new_total}

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
        ("system", "Sei un utile assistente. Rispondi alla domanda dell'utente nel modo più diretto e conciso possibile."),
        ("user", "{query}")
    ])
    
    chain = prompt | llm
    try:
        response = chain.invoke({"query": query})
        tokens = response.response_metadata.get('token_usage', {}).get('total_tokens', 0)
        new_total = state.get("total_tokens", 0) + tokens
        return {"final_answer": response.content, "total_tokens": new_total}
    except Exception as e:
        print(f"[Router] Errore Direct Answer: {e}")
        return {"final_answer": "Si è verificato un errore."}

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
