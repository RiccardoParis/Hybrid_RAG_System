import os
import json
import sqlite3
import pandas as pd
from sqlalchemy import create_engine, text

SPIDER_DEV_JSON_PATH = "../data/spider/dev.json"
SPIDER_SQLITE_DIR = "../data/spider/database" 
SPIDER_DB_URI = "postgresql://postgres:Password@127.0.0.1:5433/spider_eval"

def migrate_20_dbs():
    with open(SPIDER_DEV_JSON_PATH, 'r') as f:
        spider_data = json.load(f)
    
    # Forziamo anche il db_id in lowercase per evitare problemi di schema
    unique_dbs = list(set([item['db_id'].lower() for item in spider_data]))
    print(f"Trovati {len(unique_dbs)} database unici. Inizio migrazione normalizzata...")
    
    pg_engine = create_engine(SPIDER_DB_URI)
    
    for db_id in unique_dbs:
        sqlite_path = os.path.join(SPIDER_SQLITE_DIR, db_id, f"{db_id}.sqlite")
        
        if not os.path.exists(sqlite_path):
            continue
            
        print(f"Migrazione normalizzata per: {db_id}...")
        
        with pg_engine.connect() as pg_conn:
            pg_conn.execute(text(f'CREATE SCHEMA IF NOT EXISTS {db_id};'))
            pg_conn.commit()
            
        sqlite_conn = sqlite3.connect(sqlite_path)
        sqlite_conn.text_factory = lambda b: b.decode(errors='ignore')
        
        tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table';", sqlite_conn)['name'].tolist()
        
        for table in tables:
            if table == "sqlite_sequence": 
                continue
                
            df = pd.read_sql_query(f"SELECT * FROM [{table}]", sqlite_conn)
            
            # NORMALIZZAZIONE IN MINUSCOLO DI COLONNE E TABELLE
            df.columns = [col.lower() for col in df.columns]
            lower_table = table.lower()
            
            df.to_sql(name=lower_table, con=pg_engine, schema=db_id, if_exists='replace', index=False)
            
        sqlite_conn.close()
        
    print("Database normalizzato e pronto!")

if __name__ == "__main__":
    migrate_20_dbs()