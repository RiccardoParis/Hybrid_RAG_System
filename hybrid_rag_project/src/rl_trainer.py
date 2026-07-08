import os
import torch
import torch.nn as nn
import torch.optim as optim
import random
from sqlalchemy import create_engine, text
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

# Caricamento variabili d'ambiente e moduli di supporto
load_dotenv()
from config import GROQ_API_KEY
from schema_extractor import get_compact_schemas

# Parametri di configurazione
LAMBDA_PENALTY = 0.0
MODEL_NAME_OR_PATH = "distilbert-base-multilingual-cased"

ACTIONS_MAP = {
    "no_retrieval": 0,
    "vector": 1,
    "graph": 2,
    "sql": 3,
    "multi": 4
}

def get_engine():
    """Recupera l'engine di connessione al database PostgreSQL."""
    postgres_uri = os.getenv("POSTGRES_URI", "")
    if not postgres_uri or "TUAPASSWORD" in postgres_uri:
        raise ValueError("POSTGRES_URI mancante o invalido.")
    return create_engine(postgres_uri)

def ensure_schema_updates(engine):
    """
    Assicura che la tabella rl_logs abbia le colonne per la risposta finale e il tracciamento
    dell'avvenuto addestramento, aggiungendole dinamicamente se necessario.
    """
    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE rl_logs ADD COLUMN IF NOT EXISTS final_answer TEXT;"))
        conn.execute(text("ALTER TABLE rl_logs ADD COLUMN IF NOT EXISTS processed BOOLEAN DEFAULT FALSE;"))

def evaluate_with_llm(query: str, answer: str) -> float:
    """
    Usa LLM-as-a-Judge per valutare la correttezza della risposta (da 0.0 a 1.0).
    Se la risposta è mancante, restituisce un valore neutro.
    """
    if not answer or answer.strip() == "":
        return 0.5 
        
    llm = ChatGroq(
        temperature=0,
        groq_api_key=GROQ_API_KEY,
        model_name="llama-3.3-70b-versatile"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Sei un giudice rigoroso e imparziale. Valuta la correttezza e l'utilità della seguente risposta rispetto alla domanda dell'utente. Restituisci ESCLUSIVAMENTE un numero decimale compreso tra 0.0 e 1.0. Non aggiungere testo o spiegazioni."),
        ("user", f"Domanda: {query}\nRisposta: {answer}")
    ])
    chain = prompt | llm
    
    try:
        res = chain.invoke({})
        score_text = res.content.strip()
        score = float(score_text)
        return max(0.0, min(1.0, score))
    except Exception as e:
        print(f"[LLM Judge] Errore di parsing o di connessione ({e}). Punteggio: 0.5")
        return 0.5

def train_batch():
    """Recupera i log non processati, calcola i reward ed esegue backpropagation sul modello."""
    engine = get_engine()
    ensure_schema_updates(engine)
    
    print("[RL Trainer] Connessione a PostgreSQL stabilita. Ricerca dei log non processati...")
    with engine.begin() as conn:
        result = conn.execute(text("""
            SELECT id, query, final_answer, chosen_arm, token_cost, user_reward 
            FROM rl_logs 
            WHERE processed = FALSE
        """))
        rows = result.fetchall()
        
    # Rimescola il dataset per evitare il catastrophic forgetting
    random.shuffle(rows)
        
    if not rows:
        print("[RL Trainer] Nessun nuovo log da elaborare. Uscita.")
        return

    print(f"[RL Trainer] Trovati {len(rows)} campioni per il fine-tuning. Caricamento del modello...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME_OR_PATH)
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_NAME_OR_PATH, num_labels=5)
    model.to(device)
    model.train() # Mette il modello in modalità addestramento (attiva Dropout/Grad)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-5)
    loss_fn = nn.MSELoss()
    
    # Recupera i metadati compatti (lo stato) esattamente come fa il Router online
    schemas = get_compact_schemas()
    
    total_loss = 0.0
    processed_ids = []
    
    for row in rows:
        log_id, query, final_answer, chosen_arm, token_cost, user_reward = row
        
        # 1. Valutazione dell'Accuratezza
        if user_reward is not None:
            # Feedback esplicito dell'utente (mappiamo pollice su a 1.0, pollice giù a 0.0)
            accuracy = 1.0 if user_reward > 0 else 0.0
            print(f"[Log {log_id}] Feedback dell'utente rilevato: {accuracy}")
        else:
            # Nessun feedback: invochiamo l'LLM come giudice
            accuracy = evaluate_with_llm(query, final_answer)
            print(f"[Log {log_id}] Valutazione LLM Judge: {accuracy:.2f}")
            
        # 2. Calcolo della Ricompensa Penalizzata: r_a = Accuratezza - (lambda * Token_Cost)
        # Il router deve imparare che estrarre troppi dati (multi-source) ha un costo!
        reward = accuracy - (LAMBDA_PENALTY * token_cost)
        
        # 3. Ricostruzione dello Stato originario
        state_text = f"Query: {query}\nVector Meta: {schemas['vector']}\nGraph Meta: {schemas['graph']}\nSQL Meta: {schemas['sql']}"
        inputs = tokenizer(state_text, return_tensors="pt", truncation=True, padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # 4. Forward Pass e Calcolo della probabilità (Policy)
        optimizer.zero_grad()
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)[0]
        
        # Isoliamo la probabilità che il modello aveva predetto per il braccio effettivamente tirato
        arm_idx = ACTIONS_MAP.get(chosen_arm, 4)
        predicted_prob = probs[arm_idx]
        
        # 5. Backpropagation (Calcolo Loss e Aggiornamento Pesi)
        # Usiamo il reward calcolato come target desiderato per l'aggiornamento
        target_tensor = torch.tensor(reward, dtype=torch.float32).to(device)
        
        loss = loss_fn(predicted_prob, target_tensor)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        processed_ids.append(log_id)
        
        print(f" -> Arm: {chosen_arm} | Token: {token_cost} | Reward R_a: {reward:.3f} | Pred: {predicted_prob.item():.3f} | Loss: {loss.item():.4f}\n")
        
    avg_loss = total_loss / len(rows)
    print(f"[RL Trainer] Fine-tuning completato. Loss Media del batch: {avg_loss:.4f}")
    
    # 6. Salvataggio del modello per le future inferenze
    print(f"[RL Trainer] Sovrascrittura dei pesi in '{MODEL_NAME_OR_PATH}'...")
    model.save_pretrained(MODEL_NAME_OR_PATH)
    tokenizer.save_pretrained(MODEL_NAME_OR_PATH)
    
    # 7. Aggiornamento dello stato nel Database
    print(f"[RL Trainer] Chiusura dei log processati...")
    with engine.begin() as conn:
        conn.execute(
            text("UPDATE rl_logs SET processed = TRUE WHERE id = ANY(:ids)"), 
            {"ids": processed_ids}
        )
        
    print("[RL Trainer] Operazione completata con successo. Il Router ora è più intelligente!")

if __name__ == "__main__":
    train_batch()
