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
    ns_ids: List[str]
    vector_results: Any
    graph_result: Any
    lookup_results: List[str]
    final_answer: str

# Inizializziamo i retriever
# Passiamo esplicitamente nomic-embed-text se impostato in ingest.py
vector_retriever = VectorRetriever(collection_name="hybrid_rag", model_name="nomic-embed-text")
graph_retriever = GraphRetriever()

def parse_query_node(state: AgentState):
    """
    Cerca gli ID che iniziano per 'ns/' nella query.
    Popola la lista ns_ids nello stato.
    """
    query = state["query"]
    # Trova tutti i match per 'ns/' seguiti da caratteri alfanumerici o trattini
    ns_ids = re.findall(r'\bns/[\w-]+\b', query)
    print(f"[Router] Trovati IDs: {ns_ids}")
    return {"ns_ids": ns_ids}

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
    except Exception as e:
        print(f"[Router] Errore Graph Search: {e}")
        result = {"error": f"Errore Graph Search: {e}"}
    return {"graph_result": result}

def lookup_node(state: AgentState):
    """Esegue la funzione di lookup se sono stati trovati degli ID."""
    ns_ids = state.get("ns_ids", [])
    print(f"[Router] Esecuzione Lookup per gli IDs: {ns_ids}")
    
    lookup_results = []
    if ns_ids:
        try:
            # Esegue la query Cypher usando la connessione di GraphRetriever
            cypher_query = "MATCH (n) WHERE n.center IN $ids RETURN n.center AS id, n.title AS title, labels(n) AS type"
            res = graph_retriever.graph.query(cypher_query, {"ids": ns_ids})
            
            for r in res:
                lookup_results.append(f"ID: {r['id']} corrisponde a '{r['title']}' (Tipo: {r['type']})")
                
            if not lookup_results:
                lookup_results.append("Nessun match trovato nel database per gli ID richiesti.")
        except Exception as e:
            print(f"[Router] Errore Lookup: {e}")
            lookup_results.append(f"Errore Lookup: {e}")
            
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
        ("system", """Sei un assistente AI avanzato. Il tuo compito è rispondere alla domanda dell'utente fondendo le informazioni provenienti da diverse fonti di recupero.
        
Regole:
1. Se sono presenti informazioni di 'Lookup', utilizzale per sostituire gli ID (es. 'ns/...') provenienti dal grafo o dal vettore con i nomi reali o i dettagli forniti.
2. Fondi le informazioni di Vector Search e Graph Search. Se sono compatibili, uniscile in una risposta coesa.
3. Se ci sono conflitti tra Vector e Graph, scegli l'informazione che sembra più fattuale e accurata, oppure menziona entrambe se c'è incertezza.
4. Genera una risposta finale unificata, chiara e concisa in italiano."""),
        ("user", """Domanda: {query}
        
Contesto Vector Search:
{vector_res}

Contesto Graph Search:
{graph_res}

Contesto Lookup:
{lookup_res}
""")
    ])
    
    chain = prompt | llm
    
    # Prepara input per LLM
    query = state.get("query", "")
    vector_res = state.get("vector_results", "")
    graph_res = state.get("graph_result", "")
    lookup_res = state.get("lookup_results", [])
    
    response = chain.invoke({
        "query": query,
        "vector_res": str(vector_res),
        "graph_res": str(graph_res),
        "lookup_res": str(lookup_res) if lookup_res else "Nessun lookup effettuato."
    })
    
    return {"final_answer": response.content}

def route_parallel(state: AgentState):
    """
    Logica di routing condizionale.
    Restituisce una lista di nodi da eseguire in parallelo.
    """
    # Vogliamo sempre eseguire le due ricerche semantiche e a grafo
    routes = ["vector_search", "graph_search"]
    
    # Se ci sono ID rilevati, aggiungiamo anche il nodo di lookup
    if state.get("ns_ids"):
        routes.append("lookup")
        
    return routes

# 1. Inizializza il StateGraph
workflow = StateGraph(AgentState)

# 2. Aggiunge i nodi
workflow.add_node("parse_query", parse_query_node)
workflow.add_node("vector_search", vector_search_node)
workflow.add_node("graph_search", graph_search_node)
workflow.add_node("lookup", lookup_node)
workflow.add_node("late_fusion", late_fusion_node)

# 3. Definisce il punto di ingresso
workflow.add_edge(START, "parse_query")

# 4. Aggiunge i percorsi condizionali (con esecuzione in parallelo)
workflow.add_conditional_edges(
    "parse_query",
    route_parallel,
    {
        "vector_search": "vector_search",
        "graph_search": "graph_search",
        "lookup": "lookup"
    }
)

# 5. Tutti i percorsi paralleli convergono in late_fusion
workflow.add_edge("vector_search", "late_fusion")
workflow.add_edge("graph_search", "late_fusion")
workflow.add_edge("lookup", "late_fusion")

# 6. Il nodo late_fusion termina l'esecuzione
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
