import os
import sys
import json
import time
import pandas as pd
from dotenv import load_dotenv

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from src.router import app
from src.rl_logger import update_reward, get_engine

load_dotenv()

TEST_JSON_PATH = "../data/holdout_test_set.json"

def run_evaluation(phase_name="Baseline"):
    print(f"\n{'='*50}\n🚀 AVVIO VALUTAZIONE: FASE {phase_name.upper()}\n{'='*50}")
    
    with open(TEST_JSON_PATH, 'r', encoding='utf-8') as f:
        test_queries = json.load(f)
        
    checkpoint_file = f"eval_checkpoint_{phase_name.lower()}.json"
    csv_filename = f"eval_results_{phase_name.lower()}.csv"
    
    results = []
    
    # 1. CARICAMENTO CHECKPOINT
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        print(f"[Checkpoint] Trovati {len(results)} risultati precedenti. Riprendo l'esecuzione...")
        
    start_idx = len(results)
    total_queries = len(test_queries)
    
    if start_idx >= total_queries:
        print("[Info] Valutazione già completata per questa fase. Genero direttamente il CSV finale.")
        df = pd.DataFrame(results)
        df.to_csv(csv_filename, index=False)
        return

    # 2. CICLO DI VALUTAZIONE
    for idx in range(start_idx, total_queries):
        item = test_queries[idx]
        query = item["query"]
        expected_route = item["expected_route"]
        
        print(f"\n[{idx+1}/{total_queries}] Query: {query[:50]}...")
        
        start_time = time.time()
        
        try:
            # Esecuzione del sistema RAG
            final_state = app.invoke({"query": query})
            
            # Estrazione parametri dallo stato
            actual_route = final_state.get("chosen_arm", "unknown")
            log_id = final_state.get("log_id")
            inp = final_state.get("input_tokens", 0)
            out = final_state.get("output_tokens", 0)
            cost = final_state.get("total_cost", 0.0)
            latency = time.time() - start_time
            
            # Oracolo: Il feedback automatico
            is_correct = 1 if actual_route == expected_route else 0
            status_icon = "✅" if is_correct else "❌"
            
            # Lascia il feedback sul Database
            if log_id:
                update_reward(log_id, is_correct)
            
            print(f" -> Atteso: {expected_route} | Scelto: {actual_route} {status_icon} | Latenza: {latency:.2f}s | Costo: ${cost:.5f}")

            # Creazione di una stringa unica di contesto
            vector_ctx = final_state.get("vector_results", "")
            graph_ctx = final_state.get("graph_result", "")
            lookup_ctx = final_state.get("lookup_results", "")
            sql_ctx = final_state.get("sql_context", "")
            
            raw_context = f"VECTOR: {vector_ctx}\nGRAPH: {graph_ctx}\nLOOKUP: {lookup_ctx}\nSQL: {sql_ctx}"
            
            # Aggiunta ai risultati
            results.append({
                "Phase": phase_name,
                "Query_ID": idx + 1,
                "Query": query,
                "Expected_Route": expected_route,
                "Actual_Route": actual_route,
                "Is_Correct": is_correct,
                "Input_Tokens": inp,
                "Output_Tokens": out,
                "Total_Cost_USD": cost,
                "Latency_sec": latency,
                "Log_ID": log_id,
                "Final_Answer": final_state.get("final_answer", ""),
                "Context": raw_context
            })
            
            # 3. SALVATAGGIO SU DISCO AD OGNI ITERAZIONE
            with open(checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=4)
                
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "rate_limit" in error_msg.lower():
                print(f"\n🛑 [RATE LIMIT] Hai esaurito i token giornalieri (TPD) di Groq.")
                print(f"Progressi salvati in '{checkpoint_file}'. Puoi rilanciare lo script domani e riprenderà esattamente da qui!")
                sys.exit(0)
            else:
                print(f" -> ⚠️ ERRORE GENERICO DI SISTEMA: {e}")
                results.append({
                    "Phase": phase_name,
                    "Query_ID": idx + 1,
                    "Query": query,
                    "Expected_Route": expected_route,
                    "Actual_Route": "error",
                    "Is_Correct": 0,
                    "Input_Tokens": 0, "Output_Tokens": 0, "Total_Cost_USD": 0.0, "Latency_sec": 0.0, "Log_ID": None,
                    "Final_Answer": "Errore di sistema",
                    "Context": "Nessun contesto"
                })
                with open(checkpoint_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=4)
                    
        time.sleep(1.5) # Pausa di cortesia

    # Esportazione Finale in CSV
    df = pd.DataFrame(results)
    df.to_csv(csv_filename, index=False)
    
    accuracy = (df['Is_Correct'].sum() / len(df)) * 100
    print(f"\n{'='*50}\n📊 RISULTATI FASE {phase_name.upper()}")
    print(f"Accuracy del Router (EX): {accuracy:.2f}%")
    print(f"Costo Totale Test: ${df['Total_Cost_USD'].sum():.5f}")
    print(f"Dati esportati in: {csv_filename}\n{'='*50}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=str, default="Baseline", help="Nome della fase (es. Baseline, SFT, RL)")
    args = parser.parse_args()
    
    run_evaluation(phase_name=args.phase)