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

Ricerca Fuzzy (Defensive Multi-Property Search): I database a grafo usano termini specifici, mentre gli utenti usano termini generici. Quando filtri per proprietà testuali, NON usare MAI la corrispondenza esatta (=). Usa SEMPRE la ricerca parziale case-insensitive espandendo i concetti con sinonimi o radici della parola. Poiché non puoi sapere a priori in quale proprietà (es. id, title, name) è salvato il valore cercato dall'utente, devi SEMPRE applicare l'operatore CONTAINS in OR su TUTTE le principali proprietà testuali del nodo indicate nello schema.
Esempio di pattern obbligatorio: WHERE toLower(n.title) CONTAINS 'valore' OR toLower(n.id) CONTAINS 'valore'

Esplorazione Sicura: Se la domanda richiede di trovare elementi collegati, usa direzioni non orientate (es. MATCH (n)-[r]-(m)) per evitare di perdere risultati a causa della direzionalità degli archi.

TRADUZIONE IN INGLESE OBBLIGATORIA: Poiché i dati nel database a grafo sono in lingua inglese, quando espandi i concetti dell'utente per la ricerca tramite CONTAINS, DEVI prima tradurre i concetti e i sinonimi in INGLESE.
Esempio: Se l'utente chiede 'problemi cutanei', la clausola deve essere WHERE toLower(n.title) CONTAINS 'skin' OR toLower(n.title) CONTAINS 'rash' (non cercare 'pelle' o 'cutaneo').

EVITARE DUPLICATI (DISTINCT): Usa SEMPRE la keyword DISTINCT nella clausola di ritorno per evitare esplosioni cartesiane dovute a relazioni multiple.

OUTPUT PULITO: Non restituire mai l'intero nodo (es. vietato usare RETURN n). Restituisci solo la proprietà testuale rilevante.
Esempio di ritorno obbligatorio: RETURN DISTINCT n.title oppure RETURN DISTINCT n.name.

DIVIETO CORRISPONDENZA INLINE: È SEVERAMENTE VIETATO filtrare i nodi inserendo le proprietà direttamente tra le parentesi graffe (es. SBAGLIATO: MATCH (n:Gene {{id: 'EDN3'}})). Devi SEMPRE usare la clausola WHERE con toLower(n.nome_proprieta) CONTAINS 'valore'.

ESTRAZIONE RELAZIONI (TRIPLE SEMANTICHE): Se la domanda dell'utente chiede esplicitamente di trovare le "relazioni" o i "collegamenti" tra nodi, non usare la ricerca a lunghezza variabile -[*]-. Mappa sempre l'arco con una variabile (es. -[r]-) e restituisci la tripla completa usando ESATTAMENTE questa sintassi di ritorno: RETURN DISTINCT n.title AS Partenza, type(r) AS Relazione, m.title AS Arrivo.

SINTASSI DELLE LABEL: NON usare MAI il carattere pipe (|) all'interno delle dichiarazioni dei nodi (es. VIETATO FARE (n:Disease|Gene)). Specifica un singolo nodo di partenza con una singola label, oppure usa un nodo generico (n) e applica i filtri nella clausola WHERE con la funzione labels(n).

DIVIETO TYPE() SUI NODI: In Cypher, non usare MAI la funzione type() su una variabile che rappresenta un nodo (es. type(n) produrrà un errore fatale). Se hai bisogno di restituire il tipo/categoria di un nodo, DEVI usare la funzione labels() (es. labels(n)). Usa type() ESCLUSIVAMENTE per le relazioni (es. type(r)).

VINCOLI DI OUTPUT ASSOLUTI:
OUTPUT FORMAT: Devi restituire ESCLUSIVAMENTE il codice Cypher crudo. È severamente vietato aggiungere spiegazioni, premesse, pensieri o testo discorsivo prima o dopo la query. Restituisci solo ed esclusivamente le righe di codice Cypher.
ZERO TESTO DISCORSIVO: Restituisci ESCLUSIVAMENTE la query Cypher nuda e cruda. Non aggiungere saluti, spiegazioni, commenti o testo come 'Ecco la query' o 'Oppure potresti usare'.
SINGOLA QUERY: Genera esattamente UNA (1) sola query Cypher valida che sia la migliore interpretazione della domanda. NON proporre varianti o opzioni alternative.
NESSUN MARKDOWN: Non racchiudere la query in blocchi di codice (es. non usare ```cypher ... ```). La risposta deve iniziare direttamente con MATCH o CALL.

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

    def ask(self, question: str, callbacks=None):
        """Pone una domanda basandosi sui dati nel knowledge graph."""
        try:
            return self.chain.invoke({"query": question}, config={"callbacks": callbacks})
        except Exception as e:
            print(f"[Router] Fallback attivato: Errore sintassi Cypher intercettato. Dettagli: {str(e)[:100]}...")
            print("[Router] Passaggio trasparente al Vector Search.")
            return ""
