from langchain_neo4j import Neo4jGraph, GraphCypherQAChain
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from config import NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, GROQ_API_KEY

STATIC_SCHEMA = """
Node properties:
Person (id: STRING, title: STRING, center: STRING)
Organization (id: STRING, title: STRING, center: STRING)
CreativeWork (id: STRING, title: STRING, center: STRING)
Event (id: STRING, title: STRING, center: STRING)
Location (id: STRING, title: STRING, center: STRING)
Product (id: STRING, title: STRING, center: STRING)
Concept (id: STRING, title: STRING, center: STRING)

Relationship properties:
(nessuna)

The relationships:
(:CreativeWork|Product|Event|Concept)-[:CREATED_BY]->(:Person|Organization)
(:Person|Organization|CreativeWork)-[:PART_OF]->(:Organization|Location|CreativeWork)
(:Event|Organization|Person|CreativeWork)-[:LOCATED_IN]->(:Location)
(:Person|Organization|CreativeWork)-[:INVOLVED_IN]->(:Event|CreativeWork|Organization)
(:CreativeWork|Product)-[:RELEASED_IN]->(:Location|Event|Product)
(:Person|Organization|CreativeWork|Event|Location|Product|Concept)-[:RELATED_TO]->(:Person|Organization|CreativeWork|Event|Location|Product|Concept)
"""

CYPHER_PROMPT = PromptTemplate(
    input_variables=["schema", "question"],
    template="""You are an expert Cypher query generator. Given a schema and a question, generate a valid Cypher query.
Schema:
""" + STATIC_SCHEMA + """
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

CYPHER SYNTAX RULES:
1. NEVER apply string functions like toLower() directly to pattern expressions. toLower((a)-[:REL]->(b)) is INVALID syntax.
2. NEVER introduce new node variables inside a WHERE clause or an OR condition.
3. ALWAYS MATCH the complete path first, then apply functions to the specific node PROPERTIES.
Example of CORRECT syntax:
MATCH (p:Person)-[:LOCATED_IN]->(l:Location)
WHERE toLower(l.title) CONTAINS 'gallia'

EXAMPLES OF VALID CYPHER QUERIES:

Question: Chi ha sviluppato Valkyria Chronicles?
Cypher: MATCH (c:CreativeWork)-[:CREATED_BY]->(o:Organization) WHERE toLower(c.title) CONTAINS 'valkyria' RETURN o.title

Question: Dove si è svolta la Seconda Guerra Europea?
Cypher: MATCH (e:Event)-[:LOCATED_IN]->(l:Location) WHERE toLower(e.title) CONTAINS 'europan war' RETURN l.title

Question: Trova tutte le connessioni di Raita Honjou
Cypher: MATCH (p:Person)-[r]-(m) WHERE toLower(p.title) CONTAINS 'raita honjou' RETURN p.title, type(r), m.title

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
    def __init__(self, model_name: str = "llama-3.1-8b-instant"):
        # Inizializza la connessione al grafo Neo4j
        self.graph = Neo4jGraph(
            url=NEO4J_URI, 
            username=NEO4J_USERNAME, 
            password=NEO4J_PASSWORD
        )
        
        # Disabilita il refresh dinamico e svuota la cache di LangChain per iniettare STATIC_SCHEMA via prompt
        self.graph.schema = ""
        
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
