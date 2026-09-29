import os
import json
import pandas as pd
from sqlalchemy import create_engine, text
from config import QDRANT_URL, QDRANT_API_KEY, NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from neo4j import GraphDatabase
from vector_retriever import VectorRetriever

def ingest_document_to_vector(file_path: str, generate_metadata: bool = True):
    """
    Loads a file (.txt or .pdf), splits it into chunks, calculates embeddings
    with HuggingFaceEmbeddings, and saves it into Qdrant ('hybrid_rag' collection).
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file '{file_path}' does not exist.")
        
    print(f"[{file_path}] Loading document...")
    _, ext = os.path.splitext(file_path)
    if ext.lower() == '.pdf':
        loader = PyPDFLoader(file_path)
    elif ext.lower() == '.txt':
        loader = TextLoader(file_path, encoding='utf-8')
    else:
        raise ValueError("Unsupported file format. Please use .txt or .pdf")
        
    documents = loader.load()
    
    print(f"[{file_path}] Splitting document into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)
    
    api_key = QDRANT_API_KEY if QDRANT_API_KEY and QDRANT_API_KEY != "your_qdrant_api_key_here" else None
    
    print(f"[{file_path}] Initializing HuggingFaceEmbeddings (intfloat/multilingual-e5-base)...")
    embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-base")
    
    print(f"[{file_path}] Inserting {len(chunks)} chunks into Qdrant (collection: 'hybrid_rag')...")
    QdrantVectorStore.from_documents(
        chunks,
        embeddings,
        url=QDRANT_URL,
        api_key=api_key,
        collection_name="hybrid_rag",
        force_recreate=False,
        timeout=120  # Allows consecutive multiple insertions
    )
    print(f"[{file_path}] Qdrant ingestion completed successfully.")
    
    if generate_metadata:
        print(f"[{file_path}] Automatically generating vector metadata via LLM...")
        vr = VectorRetriever()
        vr.generate_and_save_metadata(documents)
        print(f"[{file_path}] Metadata successfully updated.")
        
    return documents

def ingest_graph_from_json(json_file_path: str):
    """
    Loads nodes and relationships from a JSON file and inserts them into Neo4j.
    The JSON must contain 'nodes' (with 'id', 'label', 'properties')
    and 'edges' (with 'source', 'target', 'type') keys.
    """
    if not os.path.exists(json_file_path):
        raise FileNotFoundError(f"The file '{json_file_path}' does not exist.")
        
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    nodes = data.get('nodes', [])
    edges = data.get('edges', [])
    
    print(f"[{json_file_path}] Found {len(nodes)} nodes and {len(edges)} relationships. Starting Neo4j import...")
    
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    
    with driver.session() as session:
        # For nodes: group by label to dynamically build the query without requiring APOC plugin
        nodes_by_label = {}
        for n in nodes:
            label = n.get('label', 'Entity')
            nodes_by_label.setdefault(label, []).append({
                'id': n['id'],
                'properties': n.get('properties', {})
            })
            
        for label, label_nodes in nodes_by_label.items():
            cypher = f"""
            UNWIND $nodes AS node
            MERGE (n:`{label}` {{id: node.id}})
            SET n += node.properties
            """
            session.run(cypher, nodes=label_nodes)
            
        # For relationships: group by type
        edges_by_type = {}
        for e in edges:
            rel_type = e.get('type', 'RELATED_TO')
            edges_by_type.setdefault(rel_type, []).append({
                'source': e['source'],
                'target': e['target']
            })
            
        for rel_type, type_edges in edges_by_type.items():
            cypher = f"""
            UNWIND $edges AS edge
            MATCH (source {{id: edge.source}})
            MATCH (target {{id: edge.target}})
            MERGE (source)-[:`{rel_type}`]->(target)
            """
            session.run(cypher, edges=type_edges)
            
    driver.close()
    print(f"[{json_file_path}] Neo4j ingestion completed successfully.")

def ingest_generic_sql(file_path: str, table_name: str):
    """
    Agnostic ingestion: loads CSV or JSON, automatically creates or updates the table.
    """
    postgres_uri = os.getenv("POSTGRES_URI")
    engine = create_engine(postgres_uri)
    
    # 1. Dynamic file reading
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith('.json'):
        df = pd.read_json(file_path)
    else:
        raise ValueError("Unsupported format. Please use .csv or .json")

    # 2. Dynamic SQL Ingestion
    # 'if_exists="append"' allows appending records to existing tables
    # 'index=False' prevents saving the dataframe index as a column
    df.to_sql(table_name, engine, if_exists='append', index=False)
    
    print(f"[SQL Ingest] Table '{table_name}' updated with {len(df)} rows.")

def batch_ingest_vectors(file_paths: list):
    """Handles multiple file ingestions and regenerates vector metadata only once at the end."""
    all_docs = []
    
    for path in file_paths:
        # Reuse the base function for each individual file
        docs = ingest_document_to_vector(path, generate_metadata=False)
        if docs:
            all_docs.extend(docs)
            
    # At the end of the loop, update metadata in a single shot
    if all_docs:
        vr = VectorRetriever()
        vr.generate_and_save_metadata(all_docs)
        
    return len(file_paths)

if __name__ == "__main__":
    print("Dynamic Ingestion Module. Use the following functions:")
    print("- ingest_document_to_vector('path/file.pdf')")
    print("- ingest_graph_from_json('path/graph.json')")