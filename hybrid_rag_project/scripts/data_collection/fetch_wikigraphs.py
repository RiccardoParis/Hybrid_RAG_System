import os
import sys
import json
import re
import requests
import time
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
import sys

# Aggiunge la cartella 'src' al path (due livelli sopra)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
from config import GROQ_API_KEY

# Imposta la base_dir alla root del progetto (tre livelli sopra)
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
texts_dir = os.path.join(base_dir, "data", "texts")
graphs_dir = os.path.join(base_dir, "data", "graphs")

# Setup dei path e import config
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import GROQ_API_KEY

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
texts_dir = os.path.join(base_dir, "data", "texts")
graphs_dir = os.path.join(base_dir, "data", "graphs")

os.makedirs(texts_dir, exist_ok=True)
os.makedirs(graphs_dir, exist_ok=True)

GRAPH_EXTRACTION_PROMPT = PromptTemplate(
    input_variables=["text", "current_article_title"],
    template="""Sei un estrattore di Knowledge Graph strettamente vincolato a un'ontologia fissa (simile a Freebase).
Estrai entità e relazioni dal testo fornito.
REGOLA 1 - LINGUA: Usa ESCLUSIVAMENTE la lingua INGLESE per tutte le etichette, le relazioni e i nomi delle chiavi.
REGOLA 2 - LABELS PERMESSE (Scegli SOLO da questa lista, in formato UpperCamelCase): Person, Organization, CreativeWork, Event, Location, Product, Concept. NON inventare altre label (es. non usare 'Game' o 'VideoGame', usa 'CreativeWork').
REGOLA 3 - RELATIONS PERMESSE (Scegli SOLO da questa lista, in formato UPPER_SNAKE_CASE): CREATED_BY, PART_OF, LOCATED_IN, INVOLVED_IN, RELEASED_IN, RELATED_TO.
REGOLA 4 - COREFERENCE: Usa il titolo dell'articolo '{current_article_title}' come ID e nome reale per risolvere i pronomi e i riferimenti generici come 'the game' o 'the company'.

Massimo 256 nodi (rispetta la variante max256).

Formato dei nodi: {{"id": "ns/{{id_univoco}}", "label": "TIPO_SPECIFICO", "properties": {{"title": "Nome Reale", "center": "ns/{{id_univoco}}"}}}}

Formato archi: {{"source": "ns/{{id_sorgente}}", "target": "ns/{{id_destinazione}}", "type": "TIPO_RELAZIONE"}} (TIPO_RELAZIONE in UPPERCASE con underscore).

Il JSON deve avere due chiavi root: "nodes" (lista) e "edges" (lista).
Non includere markdown (come ```json), stampa solo il JSON puro.

Testo:
{text}"""
)

def main():
    print("Inizializzazione LLM (Groq) in corso...")
    llm = ChatGroq(
        temperature=0,
        groq_api_key=GROQ_API_KEY,
        model_name="llama-3.1-8b-instant"
    )
    chain = GRAPH_EXTRACTION_PROMPT | llm
    
    current_article_title = "Sconosciuto"
    valid_saved = 0
    offset = 0
    
    print("Inizio estrazione dati tramite REST API (paginazione a blocchi di 100)...")
    
    while valid_saved < 50:
        api_url = f"https://datasets-server.huggingface.co/rows?dataset=Salesforce%2Fwikitext&config=wikitext-103-raw-v1&split=train&offset={offset}&length=100"
        
        try:
            response = requests.get(api_url)
            if response.status_code != 200:
                print(f"[ERROR] API fallita (Status: {response.status_code}). Interruzione ciclo.")
                break
                
            data = response.json()
        except Exception as e:
            print(f"[ERROR] Eccezione di rete: {e}")
            break
            
        rows = data.get('rows', [])
        if not rows:
            print("[INFO] Nessuna riga rimanente nel dataset.")
            break
            
        print(f"\nScaricato blocco (offset={offset}). Trovate {len(rows)} righe.")
        
        for item in rows:
            row_data = item.get('row', {})
            text = row_data.get('text', "").strip()
            
            # Intercettazione dei titoli (iniziano e finiscono con '=')
            if text.startswith('=') and text.endswith('='):
                clean_title = text.strip('=').strip()
                if clean_title:
                    current_article_title = clean_title
            
            # Filtro: Salta righe vuote e quelle sotto i 250 caratteri
            if not text or len(text) < 250:
                continue
                
            nome_file = f"doc_{valid_saved + 1:03d}"
            text_filepath = os.path.join(texts_dir, f"{nome_file}.txt")
            graph_filepath = os.path.join(graphs_dir, f"{nome_file}.json")
            
            print(f"Elaborazione {nome_file} ({len(text)} caratteri)...")
            
            # Salvataggio testo
            try:
                with open(text_filepath, 'w', encoding='utf-8') as f:
                    f.write(text)
            except Exception as e:
                print(f"[ERROR] Salvataggio testo {text_filepath} fallito: {e}")
                continue
                
            # Generazione grafo con logica di retry per Rate Limits
            success = False
            for attempt in range(5):
                try:
                    llm_response = chain.invoke({
                        "text": text,
                        "current_article_title": current_article_title
                    })
                    result_text = llm_response.content.strip()
                    
                    match = re.search(r'\{.*\}', result_text, re.DOTALL)
                    if not match:
                        raise ValueError("Nessun JSON rilevato.")
                        
                    graph_data = json.loads(match.group(0))
                    
                    if 'nodes' not in graph_data or 'edges' not in graph_data:
                        raise ValueError("Chiavi 'nodes' o 'edges' mancanti.")
                        
                    num_nodes = len(graph_data['nodes'])
                    if num_nodes > 256:
                        raise ValueError(f"MAX256 violato: {num_nodes} nodi.")
                        
                    with open(graph_filepath, 'w', encoding='utf-8') as f:
                        json.dump(graph_data, f, indent=4, ensure_ascii=False)
                        
                    print(f"[SUCCESS] {nome_file} salvato ({num_nodes} nodi).")
                    valid_saved += 1
                    success = True
                    break
                    
                except Exception as e:
                    error_msg = str(e).lower()
                    if '429' in error_msg or 'rate_limit' in error_msg:
                        print(f"[WARN] Rate limit raggiunto (Tentativo {attempt+1}/5). In pausa per 20 secondi...")
                        time.sleep(20)
                    else:
                        print(f"[WARN] Grafo fallito per {nome_file}: {e}")
                        break
                        
            if not success:
                if os.path.exists(text_filepath):
                    os.remove(text_filepath)
                continue
                
            # Ritardo preventivo per evitare di sforare i limiti (es. 12k TPM)
            time.sleep(8)
            
            if valid_saved == 50:
                break
                
        if valid_saved < 50:
            offset += 100
            
    print(f"\nOperazione di scaffolding completata! Creati {valid_saved} documenti e grafi testuali associati.")

if __name__ == "__main__":
    main()
