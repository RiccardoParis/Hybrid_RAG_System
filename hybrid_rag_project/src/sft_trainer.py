import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import random
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

# Caricamento variabili d'ambiente
load_dotenv()
from schema_extractor import get_compact_schemas

# Configurazione Base
MODEL_NAME_OR_PATH = "distilbert-base-multilingual-cased"
EPOCHS = 3
LEARNING_RATE = 2e-5

# Mappatura delle categorie a interi
ACTIONS_MAP = {
    "no_retrieval": 0,
    "vector": 1,
    "graph": 2,
    "sql": 3,
    "multi": 4
}

def train_sft():
    """
    Esegue un Supervised Fine-Tuning iniziale (SFT) sul dataset di warmup.
    Questo dà al modello una buona 'base' prima dell'RL dinamico.
    """
    print("[SFT Trainer] Avvio Supervised Fine-Tuning iniziale...")
    
    # 1. Caricamento del dataset
    project_root = os.path.dirname(os.path.dirname(__file__))
    file_path = os.path.join(project_root, "data", "synthetic_warmup_queries.json")
    
    if not os.path.exists(file_path):
        print(f"[SFT Trainer] Errore: File {file_path} non trovato. Lancia auto_warmup.py prima.")
        return
        
    with open(file_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
        
    if not dataset:
        print("[SFT Trainer] Dataset vuoto.")
        return
        
    # Shuffle del dataset per prevenire catastrophic forgetting
    random.shuffle(dataset)
    print(f"[SFT Trainer] Trovate {len(dataset)} query. Acquisizione degli schemi...")
    
    # 2. Acquisizione Schemi Compatti (Lo Stato)
    schemas = get_compact_schemas()
    
    # 3. Setup PyTorch e Modello
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[SFT Trainer] Device in uso: {device}")
    
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME_OR_PATH)
    # num_labels=5 copre le nostre 5 categorie di routing
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_NAME_OR_PATH, num_labels=5)
    model.to(device)
    model.train()
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    # Loss Categorica classica per task di classificazione Supervisionati
    loss_fn = nn.CrossEntropyLoss()
    
    # 4. Loop di Addestramento
    for epoch in range(1, EPOCHS + 1):
        epoch_loss = 0.0
        correct_predictions = 0
        
        for idx, item in enumerate(dataset):
            query = item.get("query", "")
            expected_route = item.get("expected_route", "")
            
            # Map stringa a int (fallback a 4 se non trovata)
            target_idx = ACTIONS_MAP.get(expected_route, 4)
            
            # Ricostruisce la stringa di input esattamente come il router live
            state_text = f"Query: {query}\nVector Meta: {schemas['vector']}\nGraph Meta: {schemas['graph']}\nSQL Meta: {schemas['sql']}"
            
            # Tokenizzazione
            inputs = tokenizer(state_text, return_tensors="pt", truncation=True, padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Tensore target (CrossEntropyLoss accetta 1D tensor di classi index)
            target_tensor = torch.tensor([target_idx], dtype=torch.long).to(device)
            
            # Forward Pass
            optimizer.zero_grad()
            outputs = model(**inputs)
            
            # Calcolo Loss sui logits grezzi (non serve softmax perché è interno a CrossEntropyLoss)
            loss = loss_fn(outputs.logits, target_tensor)
            
            # Backpropagation
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # Calcolo accuracy (opzionale per display)
            pred_idx = torch.argmax(outputs.logits, dim=-1).item()
            if pred_idx == target_idx:
                correct_predictions += 1
                
        # Statistiche di fine Epoca
        avg_loss = epoch_loss / len(dataset)
        accuracy = (correct_predictions / len(dataset)) * 100
        print(f"[SFT Trainer] Epoca {epoch}/{EPOCHS} | Loss Media: {avg_loss:.4f} | Accuratezza: {accuracy:.2f}%")
        
    # 5. Salvataggio del modello addestrato (Sovrascrive i pesi in locale)
    print(f"[SFT Trainer] Addestramento concluso! Salvataggio pesi in '{MODEL_NAME_OR_PATH}'...")
    model.save_pretrained(MODEL_NAME_OR_PATH)
    tokenizer.save_pretrained(MODEL_NAME_OR_PATH)
    print("[SFT Trainer] Modello salvato con successo. Il Router è pronto all'uso!")
    
    # 6. Aggiornamento dello stato nel Database per le query di warmup
    try:
        print("[SFT Trainer] Contrassegno dei log di warmup come processati nel database...")
        postgres_uri = os.getenv("POSTGRES_URI")
        if postgres_uri and "TUAPASSWORD" not in postgres_uri:
            engine = create_engine(postgres_uri)
            with engine.begin() as conn:
                conn.execute(text("UPDATE rl_logs SET processed = TRUE WHERE processed = FALSE;"))
            print("[SFT Trainer] Database aggiornato con successo.")
    except Exception as e:
        print(f"[SFT Trainer] Impossibile aggiornare il database: {e}")

if __name__ == "__main__":
    train_sft()
