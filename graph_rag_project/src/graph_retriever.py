from langchain_community.graphs import Neo4jGraph
from langchain_community.chains.graph_qa.cypher import GraphCypherQAChain
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from config import NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, GROQ_API_KEY

CYPHER_PROMPT = PromptTemplate(
    input_variables=["schema", "question"],
    template="""You are an expert Neo4j Cypher translator. Use the provided schema to write a Cypher query that answers the user's question.
Schema: {schema}
Question: {question}
CRITICAL RULE: If the question contains an ID that starts with 'ns/', you MUST search for it using the 'center' property (e.g., MATCH (n {{center: 'ns/...'}})).
Cypher query:"""
)

class GraphRetriever:
    def __init__(self, model_name: str = "llama-3.3-70b-versatile"):
        # Inizializza la connessione al grafo Neo4j
        self.graph = Neo4jGraph(
            url=NEO4J_URI, 
            username=NEO4J_USERNAME, 
            password=NEO4J_PASSWORD
        )
        
        # Inizializza il modello linguistico Groq per generare query Cypher
        self.llm = ChatGroq(
            temperature=0, 
            groq_api_key=GROQ_API_KEY, 
            model_name=model_name
        )
        
        # Aggiorna lo schema per assicurarsi che LangChain veda i nodi appena inseriti
        self.graph.refresh_schema()
        
        # Configura la chain per il QA su grafo
        self.chain = GraphCypherQAChain.from_llm(
            graph=self.graph,
            llm=self.llm,
            verbose=True,
            allow_dangerous_requests=True,
            cypher_prompt=CYPHER_PROMPT
        )

    def ask(self, question: str):
        """Pone una domanda basandosi sui dati nel knowledge graph."""
        return self.chain.invoke({"query": question})
