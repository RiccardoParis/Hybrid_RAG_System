import os
import sys
import json
import time
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from langchain_groq import ChatGroq
from langchain_community.utilities import SQLDatabase
from langchain_classic.chains import create_sql_query_chain
from src.router import custom_sql_template, clean_sql_output

load_dotenv()

SPIDER_DB_URI = os.getenv("SPIDER_POSTGRES_URI", "postgresql://postgres:Password@127.0.0.1:5433/spider_eval")
SPIDER_DEV_JSON_PATH = "../data/spider/dev.json" 
CHECKPOINT_FILE = "spider_checkpoint.json"

def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_checkpoint(data):
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(data, f, indent=4)

def evaluate_spider():
    print("Avvio valutazione su Spider-dev con sistema di Checkpoint...")

    with open(SPIDER_DEV_JSON_PATH, 'r') as f:
        spider_data = json.load(f)
        spider_data_recover=spider_data[848:]
        
    checkpoint_data = load_checkpoint()
    
    pg_engine = create_engine(SPIDER_DB_URI)
    llm_sql = ChatGroq(model="llama-3.1-8b-instant", temperature=0.0)
    
    start_time = time.time()
    
    for idx, item in enumerate(spider_data_recover):
        str_idx = str(idx)
        
        # Se la query è già stata processata con successo in passato, la salta
        if str_idx in checkpoint_data and checkpoint_data[str_idx].get("status") != "RATE_LIMIT":
            continue
            
        db_id = item['db_id'].lower() 
        question = item['question']
        gold_query = item['query'].lower() 
        
        time.sleep(1.5) # Rispetto del Rate Limit
        
        os.environ["POSTGRES_URI"] = SPIDER_DB_URI
        
        # Inizializza il record per questo indice
        checkpoint_data[str_idx] = {"db_id": db_id, "status": "", "type": ""}
        
        try:
            db = SQLDatabase(engine=pg_engine, schema=db_id, sample_rows_in_table_info=0)
            write_query = create_sql_query_chain(llm_sql, db, prompt=custom_sql_template)
            raw_response = write_query.invoke({"question": question})
            predicted_query = clean_sql_output(raw_response).lower() 
            
            if not predicted_query.startswith("select"):
                predicted_query = "select " + predicted_query.split("select", 1)[-1]
                
        except Exception as e:
            if "429" in str(e) or "rate_limit" in str(e).lower():
                print(f"[{idx+1}/{len(spider_data_recover)}] 🛑 RATE LIMIT RAGGIUNTO! Fermo lo script.")
                checkpoint_data[str_idx]["status"] = "RATE_LIMIT"
                save_checkpoint(checkpoint_data)
                sys.exit(1) # Esce dallo script in modo pulito
            else:
                print(f"[{idx+1}/{len(spider_data_recover)}] DB: {db_id} | ⚠️ ERRORE GENERAZIONE: {e}")
                checkpoint_data[str_idx] = {"db_id": db_id, "status": "ERROR_LLM", "type": "generation_error"}
                save_checkpoint(checkpoint_data)
                continue
        
        try:
            with pg_engine.connect() as conn:
                conn.execute(text(f"SET search_path TO {db_id};"))
                
                try:
                    gold_result = set(conn.execute(text(gold_query)).fetchall())
                except Exception as e_gold:
                    print(f"[{idx+1}/{len(spider_data_recover)}] DB: {db_id} | ⚠️ CRASH GOLD QUERY (Ignorata)")
                    checkpoint_data[str_idx] = {"db_id": db_id, "status": "IGNORED", "type": "gold_crash"}
                    save_checkpoint(checkpoint_data)
                    continue
                    
                try:
                    predicted_result = set(conn.execute(text(predicted_query)).fetchall())
                except Exception as e_pred:
                    print(f"[{idx+1}/{len(spider_data_recover)}] DB: {db_id} | ❌ ERRATO SINTASSI LLM")
                    checkpoint_data[str_idx] = {"db_id": db_id, "status": "ERROR_LLM", "type": "syntax_error"}
                    save_checkpoint(checkpoint_data)
                    continue
                    
                if gold_result == predicted_result:
                    print(f"[{idx+1}/{len(spider_data_recover)}] DB: {db_id} | ✅ CORRETTO")
                    checkpoint_data[str_idx] = {"db_id": db_id, "status": "CORRECT", "type": "match"}
                else:
                    print(f"[{idx+1}/{len(spider_data_recover)}] DB: {db_id} | ❌ ERRATO LOGICA")
                    checkpoint_data[str_idx] = {"db_id": db_id, "status": "ERROR_LLM", "type": "logic_error"}
                    
                save_checkpoint(checkpoint_data)
                
        except Exception as e:
            print(f"[{idx+1}/{len(spider_data_recover)}] DB: {db_id} | ⚠️ ERRORE CONNESSIONE: {e}")
            checkpoint_data[str_idx] = {"db_id": db_id, "status": "ERROR_CONN", "type": "connection"}
            save_checkpoint(checkpoint_data)
            
    # CALCOLO METRICHE FINALI DAL CHECKPOINT
    evaluatable_queries = 0
    correct_queries = 0
    llm_errors = 0
    
    for key, data in checkpoint_data.items():
        if data["status"] == "IGNORED" or data["status"] == "RATE_LIMIT":
            continue
            
        evaluatable_queries += 1
        if data["status"] == "CORRECT":
            correct_queries += 1
        elif data["status"] == "ERROR_LLM":
            llm_errors += 1

    if evaluatable_queries > 0:
        execution_accuracy = (correct_queries / evaluatable_queries) * 100
        print("\n" + "="*50)
        print("📊 RISULTATI VALUTAZIONE COMPLETA SPIDER-DEV")
        print("="*50)
        print(f"Domande valutate e compatibili: {evaluatable_queries}")
        print(f"Query Corrette (EX): {correct_queries}")
        print(f"Errori LLM (Sintassi/Logica/Generazione): {llm_errors}")
        print(f"🎯 Execution Accuracy Finale (EX): {execution_accuracy:.2f}%")
        print("="*50)

if __name__ == "__main__":
    evaluate_spider()