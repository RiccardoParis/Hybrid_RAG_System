import os
from functools import lru_cache
from sqlalchemy import create_engine, inspect
from neo4j import GraphDatabase
from dotenv import load_dotenv

# Carica l'ambiente per ottenere le stringhe di connessione
load_dotenv()
from config import NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD

@lru_cache(maxsize=1)
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
        
        for table in tables:
            columns = inspector.get_columns(table)
            tables_info.append({"name": table, "columns": columns})
        return tables_info
    except Exception as e:
        print(f"[Schema Extractor] Errore connessione SQL: {e}")
        return []

@lru_cache(maxsize=1)
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

@lru_cache(maxsize=1)
def get_compact_schemas():
    """Restituisce un dizionario di schemi sintetici, ideale per il RLBanditRouter."""
    print("[Schema Extractor] Costruzione schemi COMPATTI in corso...")
    sql_raw = _extract_sql_raw()
    graph_raw = _extract_graph_raw()
    
    # Compatto SQL: Enfasi sulle quantità + elenco dei nomi delle tabelle
    sql_tables = ", ".join([t["name"] for t in sql_raw]) if sql_raw else "Database SQL vuoto o offline."
    sql_compact = f"DATI TABELLARI QUANTITATIVI: Conteggi, somme, id (NCT...), fasi, pazienti arruolati, tabelle: {sql_tables}"
    
    # Compatto Graph: Enfasi sulle entità collegate + elenco di nodi e relazioni
    g_labels = ", ".join(graph_raw["labels"]) if graph_raw["labels"] else "Nessun nodo"
    g_rels = ", ".join(graph_raw["relationships"]) if graph_raw["relationships"] else "Nessuna relazione"
    graph_compact = f"RETI SEMANTICHE: Triple, percorsi, relazioni dirette tra Nodi: {g_labels} e Archi: {g_rels}"
    
    # Vector DB (Prescrittivo)
    vector_compact = "TESTI NON STRUTTURATI: Abstract, meccanismi d'azione, spiegazioni prolisse, letteratura medica da PubMed."
    
    return {
        "vector": vector_compact,
        "graph": graph_compact,
        "sql": sql_compact
    }

@lru_cache(maxsize=1)
def get_detailed_schemas():
    """Restituisce un dizionario di schemi completi, ideale per i generatori Cypher/SQL."""
    print("[Schema Extractor] Costruzione schemi DETTAGLIATI in corso...")
    sql_raw = _extract_sql_raw()
    graph_raw = _extract_graph_raw()
    
    # Dettagliato SQL: Nome tabella + Nome Colonne e Tipo Dato
    sql_parts = []
    for t in sql_raw:
        cols_str = ", ".join([f"{c['name']} ({c['type']})" for c in t["columns"]])
        sql_parts.append(f"Tabella: {t['name']} | Colonne: {cols_str}")
    sql_detailed = "\n".join(sql_parts) if sql_parts else "Database SQL vuoto o offline."
    
    # Dettagliato Graph: Nodi con Proprietà e Tipi di Relazioni
    graph_parts = []
    if graph_raw["properties"]:
        for label, props in graph_raw["properties"].items():
            graph_parts.append(f"Nodo '{label}' con proprietà: {', '.join(props)}")
    else:
        g_labels = ", ".join(graph_raw["labels"]) if graph_raw["labels"] else "Nessun nodo"
        graph_parts.append(f"Nodi: {g_labels}")
        
    g_rels = ", ".join(graph_raw["relationships"]) if graph_raw["relationships"] else "Nessuna relazione"
    graph_parts.append(f"\nRelazioni disponibili: {g_rels}")
    graph_detailed = "\n".join(graph_parts)
    
    # Vector DB (Dettagliato)
    vector_detailed = "Documenti vettoriali: Abstract di letteratura medica, articoli scientifici da PubMed, trial clinici e documenti medici destrutturati."
    
    return {
        "vector": vector_detailed,
        "graph": graph_detailed,
        "sql": sql_detailed
    }
