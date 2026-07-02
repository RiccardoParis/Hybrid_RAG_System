from langchain_neo4j import Neo4jGraph, GraphCypherQAChain
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from config import NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, GROQ_API_KEY

CYPHER_PROMPT = PromptTemplate(
    input_variables=["schema", "question"],
    template="""Sei un esperto di Knowledge Graph e database Neo4j. Il tuo compito è tradurre la domanda dell'utente in una query Cypher valida, basandoti ESCLUSIVAMENTE sullo schema dinamico del database fornito di seguito.

SCHEMA DEL GRAFO:
{schema}

REGOLE DI TRADUZIONE AGNOSTICHE E ROBUSTE:

Uso dello Schema: Utilizza ESCLUSIVAMENTE le Node Labels, i Relationship Types e le Properties presenti nello schema. Non inventare o allucinare nomi di nodi. Se una label contiene spazi, usa i backtick (es. `Nome Label`).

Ricerca Fuzzy (Semantic Gap): I database a grafo usano termini specifici, mentre gli utenti usano termini generici. Quando filtri per proprietà testuali (es. title, name, id), NON usare MAI la corrispondenza esatta (=). Usa SEMPRE la ricerca parziale case-insensitive espandendo i concetti con sinonimi o radici della parola.
Esempio di pattern obbligatorio: WHERE toLower(n.nome_proprieta) CONTAINS 'radice1' OR toLower(n.nome_proprieta) CONTAINS 'sinonimo2'

Esplorazione Sicura: Se la domanda richiede di trovare elementi collegati, usa direzioni non orientate (es. MATCH (n)-[r]-(m)) per evitare di perdere risultati a causa della direzionalità degli archi.

TRADUZIONE IN INGLESE OBBLIGATORIA: Poiché i dati nel database a grafo sono in lingua inglese, quando espandi i concetti dell'utente per la ricerca tramite CONTAINS, DEVI prima tradurre i concetti e i sinonimi in INGLESE.
Esempio: Se l'utente chiede 'problemi cutanei', la clausola deve essere WHERE toLower(n.title) CONTAINS 'skin' OR toLower(n.title) CONTAINS 'rash' (non cercare 'pelle' o 'cutaneo').

EVITARE DUPLICATI (DISTINCT): Usa SEMPRE la keyword DISTINCT nella clausola di ritorno per evitare esplosioni cartesiane dovute a relazioni multiple.

OUTPUT PULITO: Non restituire mai l'intero nodo (es. vietato usare RETURN n). Restituisci solo la proprietà testuale rilevante.
Esempio di ritorno obbligatorio: RETURN DISTINCT n.title oppure RETURN DISTINCT n.name.

Restituisci SOLO ed ESCLUSIVAMENTE il codice Cypher crudo. Nessuna introduzione, nessun markdown (no ```cypher), nessuna spiegazione.

Domanda dell'utente: {question}
Query Cypher:
"""
)

QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""Based strictly on the information extracted from the graph provided below, answer the user question as clearly and precisely as possible.
Use natural language and include all relevant entities or values explicitly as they appear in the context.
Do not infer, assume, or fabricate any information not present in the context.

Graph Context:
{context}

User Question:
{question}

Answer:"""
)

class GraphRetriever:
    def __init__(self, model_name: str = "llama-3.3-70b-versatile"):
        # Inizializza la connessione al grafo Neo4j
        self.graph = Neo4jGraph(
            url=NEO4J_URI, 
            username=NEO4J_USERNAME, 
            password=NEO4J_PASSWORD
        )
        
        # Aggiorna lo schema leggendolo direttamente da Neo4j
        self.graph.refresh_schema()
        
        # Inizializza il modello linguistico Groq per generare query Cypher
        self.llm = ChatGroq(
            temperature=0, 
            groq_api_key=GROQ_API_KEY, 
            model_name=model_name
        )
        
        # Configura la chain per il QA su grafo
        self.chain = GraphCypherQAChain.from_llm(
            graph=self.graph,
            llm=self.llm,
            verbose=True,
            allow_dangerous_requests=True,
            cypher_prompt=CYPHER_PROMPT,
            qa_prompt=QA_PROMPT
        )

    def ask(self, question: str):
        """Pone una domanda basandosi sui dati nel knowledge graph."""
        try:
            return self.chain.invoke({"query": question})
        except Exception as e:
            print(f"[Router] Fallback attivato: Errore sintassi Cypher intercettato. Dettagli: {str(e)[:100]}...")
            print("[Router] Passaggio trasparente al Vector Search.")
            return ""
