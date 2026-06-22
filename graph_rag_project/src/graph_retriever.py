from langchain_neo4j import Neo4jGraph, GraphCypherQAChain
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from config import NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, GROQ_API_KEY

CYPHER_PROMPT = PromptTemplate(
    input_variables=["schema", "question"],
    template="""You are an expert Cypher query generator. Given a schema and a question, generate a valid Cypher query.
Schema:
{schema}

Question: {question}

Instructions:
1. Use the Cypher CONTAINS operator with case-insensitivity (e.g., toLower(n.title) CONTAINS toLower('keyword')).
2. DO NOT replace spaces with underscores. Keep spaces intact (e.g., 'Maniac Mansion').
3. DO NOT force the label '(n:Entity)'. Use generic nodes '(n)' or use the exact labels provided in the Schema.
4. Use undirected relationships `MATCH (n)-[r]-(m)` unless a specific direction is strictly necessary, to ensure you catch all connected nodes.
5. NEVER use 'type(n)' to filter node labels. To filter nodes, use the IN operator (e.g., 'Game' IN labels(n)) or direct label syntax (n:Game). 'type()' is strictly for relationships.
6. Avoid complex WITH clauses that drop variables. If you must use WITH, ensure you pass ALL variables you intend to RETURN later (e.g., WITH n, r, m). Keep queries simple: MATCH ... WHERE ... RETURN.
7. ALWAYS return the 'center' property of the connected nodes so that the system can resolve their IDs later. For example:
MATCH (n)-[r]-(m)
WHERE toLower(n.title) CONTAINS toLower('maniac mansion')
RETURN n.title, type(r), m.center
8. ALWAYS assign a variable to relationships if you intend to return them. NEVER write anonymous relationship filters like [:RELEASED_ON] if you use r in the RETURN clause. Always write -[r:RELEASED_ON]- or just -[r]-.
9. If you use collect() to aggregate results from multiple MATCH clauses, ALWAYS use DISTINCT to prevent Cartesian products (e.g., collect(DISTINCT p.title)).
10. Output ONLY the Cypher query. No explanations.
11. CRITICAL SYNTAX RULE: NEVER use the anonymous variable-length wildcard '-[*]-'. If you don't know the exact relationship between two nodes, you MUST use 'MATCH (n)-[r]-(m)' to capture ANY single relationship, ensuring it is always bound to the variable 'r'.
12. If you need to search across multiple hops, you MUST bind the path to a variable, e.g., 'MATCH p = (n)-[*1..3]-(m)', but avoid this unless strictly necessary.

CYPHER SYNTAX RULES:
1. MANDATORY ENGLISH TRANSLATION: The database nodes are entirely in ENGLISH. If the user asks a question in Italian (or any other language), you MUST translate the keywords to English BEFORE putting them inside the CONTAINS function. (e.g., if the user asks for 'Torre dell'Arsenale', you MUST use CONTAINS 'tower' OR CONTAINS 'arsenal'. NEVER use CONTAINS 'torre' or CONTAINS 'arsenale').
2. NEVER apply string functions like toLower() directly to pattern expressions. toLower((a)-[:REL]->(b)) is INVALID syntax.
3. NEVER introduce new node variables inside a WHERE clause or an OR condition.
4. ALWAYS MATCH the complete path first, then apply functions to the specific node PROPERTIES.
Example of CORRECT syntax:
MATCH (p:Person)-[r:LOCATED_IN]->(l:Location)
WHERE toLower(l.title) CONTAINS 'gallia'
5. FLEXIBILITY & MULTI-HOP RULE (CRITICAL): DO NOT force node labels (like :Location or :CreativeWork) unless you are 100% sure. Use generic nodes (n) and (m) to avoid missing data due to slight schema misclassifications.
6. If the user asks for relationships, connections, or structural links between two entities, NEVER use the broken syntax 'MATCH (n)-[*]->(m) RETURN type(r)'. You MUST use a variable-length path, assign the entire path to the variable 'p', and return 'p'.

Example of PERFECT syntax for finding connections:
MATCH p = shortestPath((n)-[*1..3]-(m))
WHERE (toLower(n.title) CONTAINS 'keyword1' OR toLower(n.title) CONTAINS 'keyword2')
  AND toLower(m.title) CONTAINS 'keyword3'
RETURN p LIMIT 5

EXAMPLES OF VALID CYPHER QUERIES:

Question: Chi ha sviluppato Valkyria Chronicles?
Cypher: MATCH (c:CreativeWork)-[:CREATED_BY]->(o:Organization) WHERE toLower(c.title) CONTAINS 'valkyria' RETURN o.title

Question: Dove si è svolta la Seconda Guerra Europea?
Cypher: MATCH (e:Event)-[:LOCATED_IN]->(l:Location) WHERE toLower(e.title) CONTAINS 'europan war' RETURN l.title

Question: Trova tutte le connessioni di Raita Honjou
Cypher: MATCH (p:Person)-[r]-(m) WHERE toLower(p.title) CONTAINS 'raita honjou' RETURN p.title, type(r), m.title

Question: Individua le relazioni strutturali che legano la Torre dell'Arsenale al MacArthur Museum.
Cypher: MATCH p = shortestPath((n)-[*1..3]-(m)) WHERE (toLower(n.title) CONTAINS 'tower' OR toLower(n.title) CONTAINS 'arsenal') AND toLower(m.title) CONTAINS 'macarthur' RETURN p LIMIT 5

{schema}"""
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
