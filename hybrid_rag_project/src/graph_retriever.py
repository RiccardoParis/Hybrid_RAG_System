import os
import time
from langchain_neo4j import Neo4jGraph, GraphCypherQAChain
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from config import NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, GROQ_API_KEY

CYPHER_PROMPT = PromptTemplate(
    input_variables=["schema", "question"],
    template="""You are an expert in Medical Knowledge Graphs and Neo4j databases. Translate the user's question into a valid, single Cypher query based EXCLUSIVELY on the dynamic schema provided.

GRAPH SCHEMA:
{schema}

CRITICAL RULES:
1. STRICT SCHEMA MATCHING: Use EXCLUSIVELY the exact Node Labels, Relationship Types, and Properties from the schema.
2. THE BRACKET RULE (EXACT MATCH): If the user's question contains an exact entity enclosed in brackets (e.g., [Vemurafenib]), you MUST use exact string matching in lowercase without splitting the words.
   - CORRECT: WHERE toLower(n.name) = 'vemurafenib'
   - WRONG: WHERE toLower(n.name) CONTAINS 'vemurafenib'
3. FUZZY SEARCH (GENERIC TERMS): For non-bracketed medical concepts (e.g., skin problems), use partial matching: WHERE toLower(n.title) CONTAINS 'skin' OR toLower(n.title) CONTAINS 'rash'.
4. SINGLE RELATIONSHIP PATTERN: Use undirected relationships (MATCH (n)-[r]-(m)) unless direction is strictly necessary.
5. CLEAN OUTPUT: Return ONLY the raw Cypher code. No markdown formatting, no explanations.
6. NO CARTESIAN PRODUCTS: Always use RETURN DISTINCT.
7. FORBIDDEN type() ON NODES: Use labels(n) for node types, type(r) for relationship types.

User Question: {question}
Cypher Query:"""
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
    def __init__(self, model_name: str = "openai/gpt-oss-20b"):
        # Initialize Neo4j graph connection
        self.graph = Neo4jGraph(
            url=NEO4J_URI, 
            username=NEO4J_USERNAME, 
            password=NEO4J_PASSWORD
        )
        
        # Refresh schema directly from Neo4j
        self.graph.refresh_schema()
        
        # Dynamic LLM Initialization (Supports Groq and AWS Bedrock Mantle)
        if "anthropic" in model_name or "meta.llama" in model_name:
            aws_api_key = os.getenv("AWS_MANTLE_API_KEY")
            aws_region = os.getenv("AWS_MANTLE_REGION", "us-east-1")
            if not aws_api_key:
                raise ValueError("AWS_MANTLE_API_KEY is missing in environment variables.")
            self.llm = ChatOpenAI(
                model=model_name,
                temperature=0,
                api_key=aws_api_key,
                base_url=f"https://bedrock-mantle.{aws_region}.api.aws/v1"
            )
        else:
            self.llm = ChatGroq(
                temperature=0, 
                groq_api_key=GROQ_API_KEY, 
                model_name=model_name
            )
        
        # Configure Graph Cypher QA Chain
        self.chain = GraphCypherQAChain.from_llm(
            graph=self.graph,
            llm=self.llm,
            verbose=True,
            allow_dangerous_requests=True,
            cypher_prompt=CYPHER_PROMPT,
            #qa_prompt=QA_PROMPT
            return_direct=True
        )

    def ask(self, question: str, callbacks=None):
        """Asks a question based on the Knowledge Graph data (with Rate Limit retry mechanism)."""
        max_attempts = 3
        
        for attempt in range(max_attempts):
            try:
                # Attempt chain execution
                return self.chain.invoke({"query": question}, config={"callbacks": callbacks})
                
            except Exception as e:
                error_msg = str(e).lower()
                
                # If it's a Rate Limit error (429), pause and retry
                if "429" in error_msg or "rate limit" in error_msg:
                    wait_time = 5 * (attempt + 1) # Linear backoff: 5s, 10s, 15s
                    print(f"⚠️ [Graph] API Rate Limit reached. Pausing for {wait_time} seconds (Attempt {attempt+1}/{max_attempts})...")
                    time.sleep(wait_time)
                else:
                    # If it's a TRUE Cypher syntax error, trigger immediate fallback
                    print(f"[Router] Fallback triggered: Cypher syntax error intercepted. Details: {str(e)[:100]}...")
                    print("[Router] Transparent failover to Vector Search.")
                    return ""
                    
        # If all attempts fail due to rate limits
        print("❌ [Graph] Ultimate failure after exhausting Rate Limit retries.")
        return ""