from langchain_neo4j import Neo4jGraph, GraphCypherQAChain
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from config import NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, GROQ_API_KEY

CYPHER_PROMPT = PromptTemplate(
    input_variables=["schema", "question"],
    template="""You are an expert in Knowledge Graphs and Neo4j databases. Your task is to translate the user's question into a valid Cypher query, based EXCLUSIVELY on the dynamic database schema provided below.

GRAPH SCHEMA:
{schema}

AGNOSTIC AND ROBUST TRANSLATION RULES:

1. Schema Usage: Use EXCLUSIVELY the Node Labels, Relationship Types, and Properties present in the schema. Do not invent or hallucinate node names or relationship names. If a label contains spaces, use backticks (e.g., `Label Name`).

2. Fuzzy Search (Defensive Multi-Property Search): Graph databases use specific terms, while users use generic terms. When filtering by text properties, NEVER use exact match (=). ALWAYS use partial case-insensitive search. You MUST ALWAYS apply the CONTAINS operator with OR across ALL main text properties of the node indicated in the schema.
Mandatory pattern example: WHERE toLower(n.title) CONTAINS 'value' OR toLower(n.id) CONTAINS 'value'

3. Safe Exploration & Single Match (CRITICAL): Keep your query to a single, simple MATCH statement. If the question asks to find connected elements, ALWAYS use undirected relationships (e.g., MATCH (n)-[r]-(m)). Crucially, when applying filters (WHERE) to multiple nodes in the MATCH pattern, ensure the filters are correctly associated with their respective node variables (e.g., WHERE toLower(n.name) CONTAINS 'value1' AND toLower(m.name) CONTAINS 'value2'). DO NOT use directional arrows (-> or <-).

4. Synonym Expansion: You MUST expand the user's concepts using appropriate ENGLISH synonyms and medical terms for the CONTAINS search.
Example: If the user asks for 'skin problems', the clause must be: WHERE toLower(n.title) CONTAINS 'skin' OR toLower(n.title) CONTAINS 'rash' OR toLower(n.title) CONTAINS 'dermatitis'.

5. Avoid Duplicates (DISTINCT): ALWAYS use the DISTINCT keyword in the return clause to avoid Cartesian explosions due to multiple relationships.

6. Clean Output: Never return the entire node (e.g., forbidden to use RETURN n). Return only the relevant text property.
Mandatory return example: RETURN DISTINCT n.title or RETURN DISTINCT n.name.

7. FORBIDDEN INLINE MATCHING: It is STRICTLY FORBIDDEN to filter nodes by inserting properties directly inside curly braces (e.g., WRONG: MATCH (n:Gene {{id: 'EDN3'}})). You MUST ALWAYS use the WHERE clause with toLower(n.property_name) CONTAINS 'value'.

8. Relation Extraction (Semantic Triples): If the user explicitly asks to find the "relationships" or "connections" between nodes, do not use variable-length search -[*]-. Always map the edge with a variable (e.g., -[r]-) and return the complete triple using EXACTLY this return syntax: RETURN DISTINCT n.title AS Source, type(r) AS Relationship, m.title AS Target.

9. Label Syntax: NEVER use the pipe character (|) inside node declarations (e.g., FORBIDDEN: (n:Disease|Gene)). Specify a single starting node with a single label, or use a generic node (n) and apply filters in the WHERE clause using the labels(n) function.

10. FORBIDDEN type() ON NODES: In Cypher, NEVER use the type() function on a variable representing a node (it will produce a fatal error). If you need to return the type/category of a node, you MUST use the labels() function (e.g., labels(n)). Use type() EXCLUSIVELY for relationships (e.g., type(r)).

ABSOLUTE OUTPUT CONSTRAINTS:
- OUTPUT FORMAT: You must return EXCLUSIVELY the raw Cypher code. It is strictly forbidden to add explanations, premises, thoughts, or conversational text before or after the query. Return only the Cypher code lines.
- ZERO CONVERSATIONAL TEXT: Do not add greetings, comments, or text like 'Here is the query'.
- SINGLE QUERY: Generate exactly ONE (1) valid Cypher query that is the best interpretation of the question. DO NOT propose variants or alternative options.
- NO MARKDOWN: Do not enclose the query in code blocks (e.g., do not use ```cypher ... ```). The response must start directly with MATCH or CALL.

User Question: {question}
Cypher Query:
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
