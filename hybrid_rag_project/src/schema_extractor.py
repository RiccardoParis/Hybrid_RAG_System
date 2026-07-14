import os
import json
from sqlalchemy import create_engine, inspect
from neo4j import GraphDatabase
from dotenv import load_dotenv

# Carica l'ambiente per ottenere le stringhe di connessione
load_dotenv()
from config import NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD

def _extract_sql_raw():
    """Estrae i dati raw (tabelle, colonne e tipi) da PostgreSQL."""
    postgres_uri = os.getenv("POSTGRES_URI", "")
    if not postgres_uri or "TUAPASSWORD" in postgres_uri:
        return []
    
    tables_info = []
    try:
        engine = create_engine(postgres_uri)
        inspector = inspect(engine)
        tables = inspector.get_table_names()

        # Lista delle tabelle da nascondere all'LLM
        exluded_tables = ['rl_logs', 'alembic_version']
        
        for table in tables:
            if table in exluded_tables:
                continue
            
            columns = inspector.get_columns(table)
            tables_info.append({"name": table, "columns": columns})
        return tables_info
    except Exception as e:
        print(f"[Schema Extractor] Errore connessione SQL: {e}")
        return []

def _extract_graph_raw():
    """Estrae i dati raw (labels, relazioni e proprietà) da Neo4j."""
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
        
        with driver.session() as session:
            # Recupera le labels
            res_labels = session.run("CALL db.labels()")
            labels = [record["label"] for record in res_labels]
            
            # Recupera i tipi di relazioni
            res_rels = session.run("CALL db.relationshipTypes()")
            rels = [record["relationshipType"] for record in res_rels]
            
            # Recupera le proprietà dei nodi
            properties = {}
            try:
                res_props = session.run("CALL db.schema.nodeTypeProperties() YIELD nodeType, propertyName, propertyTypes")
                for record in res_props:
                    node_type = record["nodeType"]
                    # node_type restituisce formati come ':`NomeLabel`'
                    clean_label = node_type.replace(':`', '').replace('`', '')
                    prop = record["propertyName"]
                    ptypes = record["propertyTypes"]
                    p_type = ptypes[0] if ptypes else "String"
                    
                    if clean_label not in properties:
                        properties[clean_label] = []
                    properties[clean_label].append(f"{prop} ({p_type})")
            except Exception as inner_e:
                print(f"[Schema Extractor] Attenzione, nodeTypeProperties non disponibile: {inner_e}")
            
        driver.close()
        return {"labels": labels, "relationships": rels, "properties": properties}
    except Exception as e:
        print(f"[Schema Extractor] Errore connessione Graph: {e}")
        return {"labels": [], "relationships": [], "properties": {}}

def _extract_vector_raw():
    """Legge i metadati del VectorDB generati dinamicamente durante l'ingestione."""
    project_root = os.path.dirname(os.path.dirname(__file__))
    meta_path = os.path.join(project_root, "data", "vector_metadata.json")
    
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"[Schema Extractor] Errore lettura vector_metadata.json: {e}")
            
    # Fallback di sicurezza
    return {
        "compact": "UNSTRUCTURED TEXT: Generic documents.",
        "detailed": "Vector documents without automatic profiling."
    }

def get_compact_schemas():
    """Restituisce un dizionario di schemi sintetici, ideale per il RLBanditRouter."""
    print("[Schema Extractor] Costruzione schemi COMPATTI in corso...")
    sql_raw = _extract_sql_raw()
    graph_raw = _extract_graph_raw()
    
    # Compatto SQL
    sql_tables = ", ".join([t["name"] for t in sql_raw]) if sql_raw else "Empty or offline SQL Database."
    sql_compact = f"QUANTITATIVE TABULAR DATA: Counts, sums, ids (NCT...), phases, enrolled patients, tables: {sql_tables}"
    
    # Compatto Graph
    g_labels = ", ".join(graph_raw["labels"]) if graph_raw["labels"] else "No nodes"
    g_rels = ", ".join(graph_raw["relationships"]) if graph_raw["relationships"] else "No relationships"
    graph_compact = f"SEMANTIC NETWORKS: Triples, paths, direct relationships between Nodes: {g_labels} and Edges: {g_rels}"
    
    # Vector DB (Prescrittivo)
    vector_compact = _extract_vector_raw()["compact"]
    
    return {
        "vector": vector_compact,
        "graph": graph_compact,
        "sql": sql_compact
    }

def get_detailed_schemas():
    """Restituisce un dizionario di schemi completi, ideale per i generatori Cypher/SQL."""
    print("[Schema Extractor] Costruzione schemi DETTAGLIATI in corso...")
    sql_raw = _extract_sql_raw()
    graph_raw = _extract_graph_raw()
    
    # Dettagliato SQL
    sql_parts = []
    for t in sql_raw:
        cols_str = ", ".join([f"{c['name']} ({c['type']})" for c in t["columns"]])
        sql_parts.append(f"Table: {t['name']} | Columns: {cols_str}")
    sql_detailed = "\n".join(sql_parts) if sql_parts else "Empty or offline SQL Database."
    
    # Dettagliato Graph
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
    
    # Vector DB (Dettagliato)
    vector_detailed = _extract_vector_raw()["detailed"]
    
    return {
        "vector": vector_detailed,
        "graph": graph_detailed,
        "sql": sql_detailed
    }
