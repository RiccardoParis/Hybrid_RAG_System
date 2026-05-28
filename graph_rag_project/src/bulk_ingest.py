import os
import glob
import sys

# Assicuriamoci che il modulo ingest sia importabile sia se avviamo da root che da src/
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from ingest import ingest_document_to_vector, ingest_graph_from_json

def main():
    # Definisce i percorsi assoluti partendo dalla cartella in cui si trova questo script (src/)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    texts_dir = os.path.join(base_dir, "data", "texts")
    graphs_dir = os.path.join(base_dir, "data", "graphs")

    created = False
    
    # Crea le cartelle se non esistono
    if not os.path.exists(texts_dir):
        os.makedirs(texts_dir)
        created = True
        
    if not os.path.exists(graphs_dir):
        os.makedirs(graphs_dir)
        created = True

    if created:
        print(f"Ho appena creato le seguenti cartelle di sistema:")
        print(f"- {texts_dir}")
        print(f"- {graphs_dir}")
        print("\nLe cartelle erano vuote. Per favore, inserisci i file testuali (.txt/.pdf) in data/texts/ e i grafi (.json) in data/graphs/, quindi riavvia questo script.")
        sys.exit(0)

    print("=== Avvio Bulk Ingestion ===")
    
    # 1. Ingestione dei documenti di testo / PDF per Vector DB
    print("\n--- Analisi file di testo/PDF per Vector DB ---")
    text_files = glob.glob(os.path.join(texts_dir, "*.txt")) + glob.glob(os.path.join(texts_dir, "*.pdf"))
    
    if not text_files:
        print("Nessun file trovato in data/texts/")
    else:
        for file_path in text_files:
            filename = os.path.basename(file_path)
            try:
                print(f"Elaborazione documento: {filename}...")
                ingest_document_to_vector(file_path)
                print(f"[SUCCESS] {filename} ingerito con successo nel Vector DB.")
            except Exception as e:
                print(f"[ERROR] Impossibile elaborare {filename}: {e}")

    # 2. Ingestione dei file JSON per Graph DB
    print("\n--- Analisi file JSON per Graph DB ---")
    json_files = glob.glob(os.path.join(graphs_dir, "*.json"))
    
    if not json_files:
        print("Nessun file trovato in data/graphs/")
    else:
        for file_path in json_files:
            filename = os.path.basename(file_path)
            try:
                print(f"Elaborazione JSON grafo: {filename}...")
                ingest_graph_from_json(file_path)
                print(f"[SUCCESS] {filename} ingerito con successo nel Graph DB.")
            except Exception as e:
                print(f"[ERROR] Impossibile elaborare {filename}: {e}")
                
    print("\n=== Bulk Ingestion Completata ===")

if __name__ == "__main__":
    main()
