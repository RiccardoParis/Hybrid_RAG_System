import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import random
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()
from schema_extractor import get_compact_schemas

# Base Configuration upgraded to pure English model
MODEL_NAME_OR_PATH = "distilroberta-base"
SAVE_PATH = "distilroberta-base-router"
EPOCHS = 3
LEARNING_RATE = 2e-5

ACTIONS_MAP = {
    "no_retrieval": 0,
    "vector": 1,
    "graph": 2,
    "sql": 3,
    "multi": 4
}

def train_sft():
    """
    Executes an initial Supervised Fine-Tuning (SFT) on the warmup dataset.
    """
    print("[SFT Trainer] Starting initial Supervised Fine-Tuning...")
    
    project_root = os.path.dirname(os.path.dirname(__file__))
    file_path = os.path.join(project_root, "data", "synthetic_warmup_queries.json")
    
    if not os.path.exists(file_path):
        print(f"[SFT Trainer] Error: File {file_path} not found. Run auto_warmup.py first.")
        return
        
    with open(file_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
        
    if not dataset:
        print("[SFT Trainer] Empty dataset.")
        return
        
    random.shuffle(dataset)
    print(f"[SFT Trainer] Found {len(dataset)} queries. Fetching schemas...")
    
    schemas = get_compact_schemas()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[SFT Trainer] Device in use: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_OR_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME_OR_PATH, num_labels=5)
    model.to(device)
    model.train()
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss()
    
    for epoch in range(1, EPOCHS + 1):
        epoch_loss = 0.0
        correct_predictions = 0
        
        for idx, item in enumerate(dataset):
            query = item.get("query", "")
            expected_route = item.get("expected_route", "")
            target_idx = ACTIONS_MAP.get(expected_route, 4)
            
            # Optimized Attention Window
            state_text = f"[AVAILABLE SCHEMAS]\nVector: {schemas['vector']}\nGraph: {schemas['graph']}\nSQL: {schemas['sql']}\n\n[USER QUERY]\n{query}"
            
            inputs = tokenizer(state_text, return_tensors="pt", truncation=True, padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            target_tensor = torch.tensor([target_idx], dtype=torch.long).to(device)
            
            optimizer.zero_grad()
            outputs = model(**inputs)
            
            loss = loss_fn(outputs.logits, target_tensor)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            pred_idx = torch.argmax(outputs.logits, dim=-1).item()
            if pred_idx == target_idx:
                correct_predictions += 1
                
        avg_loss = epoch_loss / len(dataset)
        accuracy = (correct_predictions / len(dataset)) * 100
        print(f"[SFT Trainer] Epoch {epoch}/{EPOCHS} | Average Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}%")
        
    print(f"[SFT Trainer] Training complete! Saving weights to '{SAVE_PATH}'...")
    model.save_pretrained(SAVE_PATH)
    tokenizer.save_pretrained(SAVE_PATH)
    print("[SFT Trainer] Model successfully saved. The Router is ready to use!")
    
    try:
        print("[SFT Trainer] Marking warmup logs as processed in the database...")
        postgres_uri = os.getenv("POSTGRES_URI")
        if postgres_uri and "TUAPASSWORD" not in postgres_uri:
            engine = create_engine(postgres_uri)
            with engine.begin() as conn:
                conn.execute(text("UPDATE rl_logs SET processed = TRUE WHERE processed = FALSE;"))
            print("[SFT Trainer] Database successfully updated.")
    except Exception as e:
        print(f"[SFT Trainer] Unable to update the database: {e}")

if __name__ == "__main__":
    train_sft()