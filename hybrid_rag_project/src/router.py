import re
import os
import json
from typing import TypedDict, List, Any, Annotated
import operator
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI  # Added for AWS Bedrock Mantle compatibility
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.callbacks import BaseCallbackHandler
from config import GROQ_API_KEY, AWS_MANTLE_API_KEY, AWS_MANTLE_BASE_URL

LLM_PRICING = {
    "llama-3.1-8b-instant": {"input": 0.05 / 1e6, "output": 0.08 / 1e6},
    "llama-3.3-70b-versatile": {"input": 0.59 / 1e6, "output": 0.79 / 1e6},
    "qwen/qwen3.6-27b": {"input": 0.60 / 1e6, "output": 3.00 / 1e6},
    "openai/gpt-oss-20b": {"input": 0.075 / 1e6, "output": 0.30 / 1e6},
    "openai/gpt-oss-120b": {"input": 0.15 / 1e6, "output": 0.60 / 1e6},
    # AWS Bedrock Pricing (Cost per 1M tokens)
    "qwen.qwen3-coder-30b-a3b-instruct": {"input": 0.15 / 1e6, "output": 0.60 / 1e6},
    "qwen.qwen3-coder-480b-a35b-instruct": {"input": 0.45 / 1e6, "output": 1.80 / 1e6}
}

class TokenCallback(BaseCallbackHandler):
    """Universal Callback for Token Tracking (Groq & OpenAI/Mantle Compatible)"""
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.input_tokens = 0
        self.output_tokens = 0
        self.total_cost = 0.0

    def on_llm_end(self, response, **kwargs):
        if response.generations:
            for gen in response.generations[0]:
                if hasattr(gen, 'message') and hasattr(gen.message, 'response_metadata'):
                    metadata = gen.message.response_metadata
                    
                    # Try Groq format first, then OpenAI/Mantle format
                    usage = metadata.get('token_usage', metadata.get('usage', {}))
                    
                    if hasattr(usage, 'prompt_tokens'): # Handle Pydantic objects if returned
                        inp = usage.prompt_tokens
                        out = usage.completion_tokens
                    else: # Handle standard dictionaries
                        inp = usage.get('prompt_tokens', 0)
                        out = usage.get('completion_tokens', 0)
                        
                    self.input_tokens += inp
                    self.output_tokens += out
                    
                    pricing = LLM_PRICING.get(self.model_name, {"input": 0.0, "output": 0.0})
                    self.total_cost += (inp * pricing["input"]) + (out * pricing["output"])

def get_chat_model(model_id: str, temperature: float = 0.0):
    """
    Inizializza dinamicamente il client corretto.
    Il trucco del punto: se l'ID contiene un '.' (es. openai.gpt-5.4), è un modello AWS Mantle.
    Se contiene uno '/' (es. openai/gpt-oss-20b), è un modello Groq.
    """
    if "." in model_id or ":" in model_id:
        if not AWS_MANTLE_API_KEY:
            raise ValueError("AWS_MANTLE_API_KEY mancante nel file .env.")
        return ChatOpenAI(
            model=model_id,
            temperature=temperature,
            api_key=AWS_MANTLE_API_KEY,
            base_url=AWS_MANTLE_BASE_URL,
            max_tokens=1024
        )
    else:
        return ChatGroq(
            model_name=model_id,
            temperature=temperature,
            groq_api_key=GROQ_API_KEY
        )

# Import retrievers and system modules
from vector_retriever import VectorRetriever
from graph_retriever import GraphRetriever
from sql_retriever import SQLRetriever
from rl_router import RLBanditRouter
from schema_extractor import get_compact_schemas, get_detailed_schemas
from rl_logger import log_interaction

# 1. State Definition
class AgentState(TypedDict):
    query: str
    vector_query: str
    sql_query: str
    graph_query: str
    chosen_arm: str
    log_id: int
    routing_decision: List[str]
    ns_ids: List[str]
    vector_results: Any
    graph_result: Any
    lookup_results: dict
    sql_context: str
    final_answer: str
    input_tokens: Annotated[int, operator.add]
    output_tokens: Annotated[int, operator.add]
    total_cost: Annotated[float, operator.add]

# 2. Initialization
vector_retriever = VectorRetriever(collection_name="hybrid_rag", model_name="intfloat/multilingual-e5-base")
graph_retriever = GraphRetriever()
sql_retriever=SQLRetriever()
bandit_router = RLBanditRouter()

# 3. Multi-Source Resolution using Detailed Schemas
def resolve_multi_source(query: str, schemas: str):
    # Change the model here for testing. Example: "anthropic.claude-3-5-sonnet-20240620-v1:0"
    model_id = "qwen.qwen3-coder-480b-a35b-instruct" 
    llm = get_chat_model(model_id=model_id, temperature=0.0)
    
    prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a Query Decomposition Master for a Multi-Source RAG system. 
The user's query requires information from different databases. Your task is to break down the query into specific SUB-QUESTIONS, assigning each to the correct database based on the provided schemas.

STRICT RULES:
1. Output EXCLUSIVELY a flat JSON dictionary. DO NOT create nested dictionaries.
2. The allowed keys are strictly: "vector", "sql", "graph".
3. The value for each key MUST BE A SIMPLE TEXT STRING (e.g., "How many patients are enrolled?"). 
4. DO NOT write code (no SQL, no Cypher) in the values, write only NATURAL LANGUAGE questions.

Database Schemas:
{schemas}"""),
      ("user", "{query}")
    ])
    try:
        res = (prompt | llm).invoke({"schemas": schemas, "query": query})
        
        # Token usage logic adapted for both Groq and OpenAI/Mantle metadata format
        usage = res.response_metadata.get('token_usage', res.response_metadata.get('usage', {}))
        if hasattr(usage, 'prompt_tokens'):
            inp = usage.prompt_tokens
            out = usage.completion_tokens
        else:
            inp = usage.get('prompt_tokens', 0)
            out = usage.get('completion_tokens', 0)
            
        pricing = LLM_PRICING.get(model_id, {"input": 0.0, "output": 0.0})
        cost = (inp * pricing["input"]) + (out * pricing["output"])
        print(f"[Router - DEBUG] Multi-source resolution cost: ${cost:.5f}")
        
        content = res.content.strip()
        match = re.search(r'\{.*\}', content, re.DOTALL)
        if match:
            try:
                parsed_dict = json.loads(match.group(0))
                return parsed_dict, inp, out, cost
            except json.JSONDecodeError:
                pass
        
        # Security fallback: return original query for both paths if JSON parsing fails
        return {"vector": query, "graph": query}, inp, out, cost
        
    except Exception as e:
        print(f"[Router] Error in resolve_multi_source: {e}")
        return {"vector": query, "graph": query}, 0, 0, 0.0

# 4. Decision Node (RL Core)
def parse_query_node(state: AgentState):
    query = state["query"]
    
    # Extract compact schemas from cache
    compact_schemas = get_compact_schemas()
    
    # Override Control Check
    forced_route = os.environ.get("FORCE_ROUTE")
    if forced_route:
        action_name = forced_route
        is_expl = False
        print(f"[Router] 🛑 OVERRIDE ACTIVE: Forced arm {action_name}")
    else:
        # Arm selection via DistilBERT (Standard Behavior)
        action_name, is_expl = bandit_router.choose_arm(
            query, epsilon=0.1, 
            vector_meta=compact_schemas["vector"], 
            graph_meta=compact_schemas["graph"], 
            sql_meta=compact_schemas["sql"]
        )
    
    print(f"[Router] Selected Arm: {action_name} (Exploration: {is_expl})")
    
    # Registration in PostgreSQL with initial token cost at 0
    log_id = log_interaction(query, action_name)
    
    routes = []
    inp, out, cost = 0, 0, 0.0
    state_updates = {}
    
    if action_name == "multi":
        detailed_schemas = get_detailed_schemas()
        schemas_str = f"Vector: {detailed_schemas['vector']}\nGraph: {detailed_schemas['graph']}\nSQL: {detailed_schemas['sql']}"
        
        # multi_decision is a DICTIONARY (e.g., {"vector": "sub-question", "sql": "sub-question"})
        multi_decision, inp, out, cost = resolve_multi_source(query, schemas_str)
        
        for db_name, sub_query in multi_decision.items():
            if db_name in ["vector", "graph", "sql"]:
                routes.append(f"{db_name}_search")
                state_updates[f"{db_name}_query"] = sub_query
                
        if not routes: 
            routes.append("vector_search")
            state_updates["vector_query"] = query
    elif action_name == "no_retrieval":
        routes = ["direct_answer"]
    else:
        routes = [f"{action_name}_search"]
        
    return {"chosen_arm": action_name, "log_id": log_id, "routing_decision": routes, "input_tokens": inp, "output_tokens": out, "total_cost": cost, **state_updates}

def vector_search_node(state: AgentState):
    """Executes search on the vector store."""
    original_query = state.get("vector_query") or state["query"]
    
    # E5 MODEL FIX: Adding the mandatory prefix for queries
    e5_query = f"query: {original_query}"
    
    print(f"[Router] Executing Vector Search for: {e5_query}")
    try:
        # Pass formatted query to Qdrant
        docs = vector_retriever.search(e5_query)
        # Extract text from documents for the LLM
        results = [doc.page_content for doc in docs]
    except Exception as e:
        print(f"[Router] Vector Search Error: {e}")
        results = [f"Vector Search Error: {e}"]
    return {"vector_results": results}

def graph_search_node(state: AgentState):
    """Executes search on the knowledge graph."""
    model_id = "qwen.qwen3-coder-480b-a35b-instruct"
    cb = TokenCallback(model_name=model_id)
    query = state.get("graph_query") or state["query"]
    print(f"[Router] Executing Graph Search for: {query}")
    try:
        # Execute query using LangChain GraphCypherQAChain
        result = graph_retriever.ask(query, callbacks=[cb])
        inp = cb.input_tokens
        out = cb.output_tokens
        cost = cb.total_cost
        # Ensure IDs enter the state correctly based on LangGraph patterns
        found_ids = list(set(re.findall(r'\bns/[\w-]+\b', str(result))))
    except Exception as e:
        print(f"[Router] Graph Search Error: {e}")
        result = {"error": f"Graph Search Error: {e}"}
        found_ids = []
        inp, out, cost = 0, 0, 0.0
        
    return {"graph_result": result, "ns_ids": found_ids, "input_tokens": inp, "output_tokens": out, "total_cost": cost}

def sql_search_node(state: AgentState):
    """Executes the search on the SQL database using the isolated module."""
    model_id = "qwen.qwen3-coder-480b-a35b-instruct"
    cb = TokenCallback(model_name=model_id)
    query = state.get("sql_query") or state["query"]
    print(f"[Router] Executing SQL Search for: {query}")
    
    try:
        result = sql_retriever.ask(query, callbacks=[cb])
        inp = cb.input_tokens
        out = cb.output_tokens
        cost = cb.total_cost
        
    except Exception as e:
        print(f"[Router] SQL Search Error: {e}")
        result = f"SQL Search Error: {e}"
        inp, out, cost = 0, 0, 0.0
        
    return {"sql_context": result, "input_tokens": inp, "output_tokens": out, "total_cost": cost}

def route_after_graph(state: AgentState):
    """
    Analyzes the string within state['graph_result']. 
    If IDs are found, it routes to lookup, otherwise it routes to late_fusion.
    """
    graph_res_str = str(state.get("graph_result", ""))
    found_ids = list(set(re.findall(r'\bns/[\w-]+\b', graph_res_str)))
    
    if found_ids:
        state['ns_ids'] = found_ids
        return "lookup"
        
    return "late_fusion"

def lookup_node(state: AgentState):
    """Executes the lookup function if IDs were found."""
    ns_ids = state.get("ns_ids", [])
    print(f"[Router] Executing Lookup for IDs: {ns_ids}")
    
    lookup_results = {}
    if ns_ids:
        try:
            # Execute Cypher query using GraphRetriever connection
            cypher_query = "MATCH (n) WHERE n.center IN $ids RETURN n.center AS id, n.title AS title, labels(n) AS type"
            res = graph_retriever.graph.query(cypher_query, {"ids": ns_ids})
            
            for r in res:
                lookup_results[r['id']] = r['title']
                
        except Exception as e:
            print(f"[Router] Lookup Error: {e}")
            
    return {"lookup_results": lookup_results}

def late_fusion_node(state: AgentState):
    """
    Collects results from Vector, Graph, and Lookup.
    Uses an LLM to fuse them into a unified response.
    """
    print("[Router] Executing Late Fusion Node")
    
    # Initialize LLM - Update model_id here to switch to Claude or others
    model_id = "qwen.qwen3-coder-480b-a35b-instruct"
    llm = get_chat_model(model_id=model_id, temperature=0.0)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an analytical medical assistant. You will receive text snippets (from PubMed) and/or records extracted from a relational database (e.g., clinical trials).

CRITICAL INSTRUCTIONS FOR SQL CONTEXT:
- If the user asked for a calculation, count, sum, or average, the SQL context will directly provide the exact computed number. Do NOT attempt to recount or recalculate; simply trust the provided number and formulate a clear, natural language answer.
- If the user asked for a semantic summary, the SQL context will provide raw text rows. Read them carefully and summarize them as requested.
- TRUST THE FILTERS: If the user's question contains specific filters (e.g., "with exactly 300 patients" or "sponsored by Pfizer"), assume the SQL database has already applied these filters perfectly. Do NOT refuse to answer just because the filtering values (like the number 300) are not explicitly printed in the SQL context.
- If the SQL context provides a list of raw database tuples (e.g., [('Item 1',), ('Item 2',)]), YOU MUST NEVER print the Python programming symbols like brackets [], parentheses (), or trailing commas. Extract the pure text and format it as a clean, human-readable bulleted list.

If the information to answer is not present in ANY of the provided contexts, explicitly state that the available data does not contain the answer and STOP. Do not invent or use prior knowledge.

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
Just state the synthesized facts directly and naturally."""),
        ("user", """Question: {query}
        
Vector Search Context:
{vector_res}

Graph Search Context:
{graph_res}

Lookup Context:
{lookup_res}

SQL Context:
{sql_res}
""")
    ])
    
    chain = prompt | llm
    
    # Prepare LLM input
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
    
    # Handle usage correctly for both Groq and OpenAI/Mantle structures
    usage = response.response_metadata.get('token_usage', response.response_metadata.get('usage', {}))
    if hasattr(usage, 'prompt_tokens'):
        inp = usage.prompt_tokens
        out = usage.completion_tokens
    else:
        inp = usage.get('prompt_tokens', 0)
        out = usage.get('completion_tokens', 0)
        
    pricing = LLM_PRICING.get(model_id, {"input": 0.0, "output": 0.0})
    cost = (inp * pricing["input"]) + (out * pricing["output"])
    
    return {"final_answer": response.content, "input_tokens": inp, "output_tokens": out, "total_cost": cost}

def direct_answer_node(state: AgentState):
    """Node for answering directly without RAG (selected by RL Router when DB query is unnecessary)."""
    print("[Router] Executing Direct Answer Node (No Retrieval)")
    query = state.get("query", "")
    
    model_id = "qwen.qwen3-coder-30b-a3b-instruct"
    llm = get_chat_model(model_id=model_id, temperature=0.0)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Answer the user's question as directly and concisely as possible."),
        ("user", "{query}")
    ])
    
    chain = prompt | llm
    try:
        response = chain.invoke({"query": query})
        usage = response.response_metadata.get('token_usage', response.response_metadata.get('usage', {}))
        if hasattr(usage, 'prompt_tokens'):
            inp = usage.prompt_tokens
            out = usage.completion_tokens
        else:
            inp = usage.get('prompt_tokens', 0)
            out = usage.get('completion_tokens', 0)
            
        pricing = LLM_PRICING.get(model_id, {"input": 0.0, "output": 0.0})
        cost = (inp * pricing["input"]) + (out * pricing["output"])
        return {"final_answer": response.content, "input_tokens": inp, "output_tokens": out, "total_cost": cost}
    except Exception as e:
        print(f"[Router] Direct Answer Error: {e}")
        return {"final_answer": "An error occurred during processing.", "input_tokens": 0, "output_tokens": 0, "total_cost": 0.0}

# 1. Initialize StateGraph
workflow = StateGraph(AgentState)

# 2. Add nodes
workflow.add_node("parse_query", parse_query_node)
workflow.add_node("vector_search", vector_search_node)
workflow.add_node("graph_search", graph_search_node)
workflow.add_node("sql_search", sql_search_node)
workflow.add_node("lookup", lookup_node)
workflow.add_node("late_fusion", late_fusion_node)
workflow.add_node("direct_answer", direct_answer_node)

# 3. Define base connections
workflow.add_edge(START, "parse_query")

# 5. Simplified Conditional Edge
def route_after_parse(state: AgentState):
    # The previous node has already made all routing decisions!
    return state["routing_decision"]

workflow.add_conditional_edges(
    "parse_query",
    route_after_parse,
    ["vector_search", "graph_search", "sql_search", "direct_answer"]
)

# vector_search and sql_search proceed directly to late_fusion
workflow.add_edge("vector_search", "late_fusion")
workflow.add_edge("sql_search", "late_fusion")

# graph_search uses the conditional edge
workflow.add_conditional_edges(
    "graph_search",
    route_after_graph,
    {
        "lookup": "lookup",
        "late_fusion": "late_fusion"
    }
)

# If lookup is executed, it flows into late_fusion
workflow.add_edge("lookup", "late_fusion")

# Generation nodes terminate execution
workflow.add_edge("late_fusion", END)
workflow.add_edge("direct_answer", END)

# 7. Compile graph into an executable application
app = workflow.compile()

if __name__ == "__main__":
    # Test 1: Query without "ns/" IDs -> Will not execute lookup
    print("--- Test 1 ---")
    state1 = app.invoke({"query": "What are the most recent therapeutic developments mentioned in the scientific literature?"})
    print(f"Final State 1: {state1}\n")
    
    # Test 2: Query with "ns/" IDs -> Will execute lookup in parallel with others
    print("--- Test 2 ---")
    state2 = app.invoke({"query": "Retrieve the instance logs for ns/server-prod-01 and node ns/router-02."})
    print(f"Final State 2: {state2}\n")