import os
import json
from config import QDRANT_URL, QDRANT_API_KEY, NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from neo4j import GraphDatabase

def ingest_document_to_vector(file_path: str):
    """
    Carica un file (.txt o .pdf), lo splitta in chunk, ne calcola gli embeddings
    con HuggingFaceEmbeddings e lo salva in Qdrant (collection 'hybrid_rag').
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Il file '{file_path}' non esiste.")
        
    print(f"[{file_path}] Caricamento documento...")
    _, ext = os.path.splitext(file_path)
    if ext.lower() == '.pdf':
        loader = PyPDFLoader(file_path)
    elif ext.lower() == '.txt':
        loader = TextLoader(file_path, encoding='utf-8')
    else:
        raise ValueError("Formato file non supportato. Usa .txt o .pdf")
        
    documents = loader.load()
    
    print(f"[{file_path}] Splitting del documento in chunk...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)
    
    api_key = QDRANT_API_KEY if QDRANT_API_KEY and QDRANT_API_KEY != "your_qdrant_api_key_here" else None
    
    print(f"[{file_path}] Inizializzazione HuggingFaceEmbeddings (BAAI/bge-base-en-v1.5)...")
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
    
    print(f"[{file_path}] Inserimento di {len(chunks)} chunk in Qdrant (collection: 'hybrid_rag')...")
    QdrantVectorStore.from_documents(
        chunks,
        embeddings,
        url=QDRANT_URL,
        api_key=api_key,
        collection_name="hybrid_rag",
        force_recreate=False  # Permette inserimenti multipli successivi
    )
    print(f"[{file_path}] Ingestione su Qdrant completata con successo.")


def ingest_graph_from_json(json_file_path: str):
    """
    Carica nodi e relazioni da un file JSON e li inserisce in Neo4j.
    Il JSON deve avere chiavi 'nodes' (con 'id', 'label', 'properties')
    e 'edges' (con 'source', 'target', 'type').
    """
    if not os.path.exists(json_file_path):
        raise FileNotFoundError(f"Il file '{json_file_path}' non esiste.")
        
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    nodes = data.get('nodes', [])
    edges = data.get('edges', [])
    
    print(f"[{json_file_path}] Trovati {len(nodes)} nodi e {len(edges)} relazioni. Avvio importazione in Neo4j...")
    
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    
    with driver.session() as session:
        # Per i nodi: raggruppiamo per label per poter creare la query dinamicamente in Neo4j senza APOC
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
            
        # Per le relazioni: raggruppiamo per type
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
    print(f"[{json_file_path}] Ingestione su Neo4j completata con successo.")

if __name__ == "__main__":
    print("Modulo di ingestione dinamica. Utilizzare le funzioni:")
    print("- ingest_document_to_vector('percorso/file.pdf')")
    print("- ingest_graph_from_json('percorso/grafo.json')")
