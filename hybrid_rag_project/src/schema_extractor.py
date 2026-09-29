import os
import json
from sqlalchemy import create_engine, inspect
from neo4j import GraphDatabase
from dotenv import load_dotenv

# Load environment to get connection strings
load_dotenv()
from config import NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD

def _extract_sql_raw():
    """Extracts raw data (tables, columns, and types) from PostgreSQL."""
    postgres_uri = os.getenv("POSTGRES_URI", "")
    if not postgres_uri or "TUAPASSWORD" in postgres_uri:
        return []
    
    tables_info = []
    try:
        engine = create_engine(postgres_uri)
        inspector = inspect(engine)
        tables = inspector.get_table_names()

        # List of tables to hide from the LLM
        excluded_tables = ['rl_logs', 'alembic_version']
        
        for table in tables:
            if table in excluded_tables:
                continue
            
            columns = inspector.get_columns(table)
            tables_info.append({"name": table, "columns": columns})
        return tables_info
    except Exception as e:
        print(f"[Schema Extractor] SQL connection error: {e}")
        return []

def _extract_graph_raw():
    """Extracts raw data (labels, relationships, and properties) from Neo4j."""
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
        
        with driver.session() as session:
            # Retrieve labels
            res_labels = session.run("CALL db.labels()")
            labels = [record["label"] for record in res_labels]
            
            # Retrieve relationship types
            res_rels = session.run("CALL db.relationshipTypes()")
            rels = [record["relationshipType"] for record in res_rels]
            
            # Retrieve node properties
            properties = {}
            try:
                res_props = session.run("CALL db.schema.nodeTypeProperties() YIELD nodeType, propertyName, propertyTypes")
                for record in res_props:
                    node_type = record["nodeType"]
                    # node_type returns formats like ':`LabelName`'
                    clean_label = node_type.replace(':`', '').replace('`', '')
                    prop = record["propertyName"]
                    ptypes = record["propertyTypes"]
                    p_type = ptypes[0] if ptypes else "String"
                    
                    if clean_label not in properties:
                        properties[clean_label] = []
                    properties[clean_label].append(f"{prop} ({p_type})")
            except Exception as inner_e:
                print(f"[Schema Extractor] Warning, nodeTypeProperties not available: {inner_e}")
            
        driver.close()
        return {"labels": labels, "relationships": rels, "properties": properties}
    except Exception as e:
        print(f"[Schema Extractor] Graph connection error: {e}")
        return {"labels": [], "relationships": [], "properties": {}}

def _extract_vector_raw():
    """Reads VectorDB metadata dynamically generated during ingestion."""
    project_root = os.path.dirname(os.path.dirname(__file__))
    meta_path = os.path.join(project_root, "data", "vector_metadata.json")
    
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"[Schema Extractor] Error reading vector_metadata.json: {e}")
            
    # Security fallback
    return {
        "compact": "UNSTRUCTURED TEXT: Generic documents.",
        "detailed": "Vector documents without automatic profiling."
    }

def get_compact_schemas():
    """Returns a dictionary of synthetic schemas, ideal for the RLBanditRouter."""
    print("[Schema Extractor] Building COMPACT schemas...")
    sql_raw = _extract_sql_raw()
    graph_raw = _extract_graph_raw()
    
    # Compact SQL
    sql_tables = ", ".join([t["name"] for t in sql_raw]) if sql_raw else "Empty or offline SQL Database."
    sql_compact = f"QUANTITATIVE TABULAR DATA: Counts, sums, ids (NCT...), phases, enrolled patients, tables: {sql_tables}"
    
    # Compact Graph
    g_labels = ", ".join(graph_raw["labels"]) if graph_raw["labels"] else "No nodes"
    g_rels = ", ".join(graph_raw["relationships"]) if graph_raw["relationships"] else "No relationships"
    graph_compact = f"SEMANTIC NETWORKS: Triples, paths, direct relationships between Nodes: {g_labels} and Edges: {g_rels}"
    
    # Vector DB (Prescriptive)
    vector_compact = _extract_vector_raw()["compact"]
    
    return {
        "vector": vector_compact,
        "graph": graph_compact,
        "sql": sql_compact
    }

def get_detailed_schemas():
    """Returns a dictionary of comprehensive schemas, ideal for Cypher/SQL generators."""
    print("[Schema Extractor] Building DETAILED schemas...")
    sql_raw = _extract_sql_raw()
    graph_raw = _extract_graph_raw()
    
    # Detailed SQL
    sql_parts = []
    for t in sql_raw:
        cols_str = ", ".join([f"{c['name']} ({c['type']})" for c in t["columns"]])
        sql_parts.append(f"Table: {t['name']} | Columns: {cols_str}")
    sql_detailed = "\n".join(sql_parts) if sql_parts else "Empty or offline SQL Database."
    
    # Detailed Graph
    graph_parts = []
    if graph_raw["properties"]:
        for label, props in graph_raw["properties"].items():
            graph_parts.append(f"Node '{label}' with properties: {', '.join(props)}")
    else:
        g_labels = ", ".join(graph_raw["labels"]) if graph_raw["labels"] else "No nodes"
        graph_parts.append(f"Nodes: {g_labels}")
        
    g_rels = ", ".join(graph_raw["relationships"]) if graph_raw["relationships"] else "No relationships"
    graph_parts.append(f"\nAvailable relationships: {g_rels}")
    graph_detailed = "\n".join(graph_parts)
    
    # Vector DB (Detailed)
    vector_detailed = _extract_vector_raw()["detailed"]
    
    return {
        "vector": vector_detailed,
        "graph": graph_detailed,
        "sql": sql_detailed
    }