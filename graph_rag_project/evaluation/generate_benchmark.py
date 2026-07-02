import os
import sys
import glob
import json
import re

# Gestione path per permettere l'importazione di config.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import GROQ_API_KEY
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

# Definiamo i percorsi partendo dalla root del progetto
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
texts_dir = os.path.join(base_dir, "data", "texts")
output_file = os.path.join(base_dir, "benchmark_data.json")

# Template per generare il dataset di test
GENERATE_PROMPT = PromptTemplate(
    input_variables=["text"],
    template="""Sei un esperto generatore di dataset. Leggi il testo fornito e genera un array JSON contenente esattamente 4 oggetti (uno per ogni categoria: 'General Knowledge', 'Fine-grained Factual', 'Relation-centric', 'Adversarial').
Ogni oggetto deve avere questa struttura: {{"category": "...", "question": "...", "ground_truth": "..."}}.
Le domande devono riferirsi esplicitamente ai fatti del testo. Assicurati di generare un JSON valido, senza blocchi markdown extra.

Testo fornito:
{text}

Array JSON:"""
)

def main():
    if not os.path.exists(texts_dir):
        print(f"Cartella {texts_dir} non trovata. Assicurati che i testi siano al loro posto.")
        sys.exit(1)

    txt_files = glob.glob(os.path.join(texts_dir, "*.txt"))
    if not txt_files:
        print(f"Nessun file .txt trovato in {texts_dir}.")
        sys.exit(0)

    print(f"Inizializzo LLM per l'estrazione automatica di domande da {len(txt_files)} file...")
    
    # Inizializza l'LLM di Groq
    llm = ChatGroq(
        temperature=0.3, # Una leggera temperatura per generare domande differenziate
        groq_api_key=GROQ_API_KEY, 
        model_name="llama-3.3-70b-versatile"
    )
    
    chain = GENERATE_PROMPT | llm
    all_benchmarks = []

    for file_path in txt_files:
        filename = os.path.basename(file_path)
        print(f"Generazione benchmark per: {filename}...")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Generazione del JSON
            response = chain.invoke({"text": content})
            result_text = response.content.strip()
            
            # Utilizziamo una regex per catturare l'array JSON dalla risposta, aggirando eventuali preamboli testuali
            match = re.search(r'\[.*\]', result_text, re.DOTALL)
            if match:
                json_str = match.group(0)
                data = json.loads(json_str)
                
                if isinstance(data, list):
                    all_benchmarks.extend(data)
                else:
                    print(f"[WARN] La risposta per {filename} non contiene un array JSON primario.")
            else:
                print(f"[ERROR] Nessun blocco array identificato nella risposta per {filename}.")
                
        except json.JSONDecodeError as e:
            print(f"[ERROR] Decodifica JSON fallita per {filename}: {e}\nTesto generato: {result_text}")
        except Exception as e:
            print(f"[ERROR] Impossibile elaborare {filename}: {e}")

    # Salvataggio del dataset aggregato
    if all_benchmarks:
        print(f"\nCompletato. Generati {len(all_benchmarks)} test cases in totale.")
        print(f"Salvataggio in sovrascrittura su: {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_benchmarks, f, indent=4, ensure_ascii=False)
        print("Salvataggio avvenuto con successo.")
    else:
        print("\nNessun benchmark generato validamente. Il file precedente non è stato sovrascritto.")

if __name__ == "__main__":
    main()
