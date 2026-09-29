import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import random
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

# Load environment variables
load_dotenv()
from schema_extractor import get_compact_schemas

# Base Configuration
MODEL_NAME_OR_PATH = "distilbert-base-multilingual-cased"
EPOCHS = 3
LEARNING_RATE = 2e-5

# Category to integer mapping
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
    This gives the model a solid 'baseline' before dynamic RL.
    """
    print("[SFT Trainer] Starting initial Supervised Fine-Tuning...")
    
    # 1. Dataset loading
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
        
    # Dataset shuffle to prevent catastrophic forgetting
    random.shuffle(dataset)
    print(f"[SFT Trainer] Found {len(dataset)} queries. Fetching schemas...")
    
    # 2. Fetch Compact Schemas (The State)
    schemas = get_compact_schemas()
    
    # 3. PyTorch and Model Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[SFT Trainer] Device in use: {device}")
    
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME_OR_PATH)
    # num_labels=5 covers our 5 routing categories
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_NAME_OR_PATH, num_labels=5)
    model.to(device)
    model.train()
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    # Standard Categorical Loss for Supervised classification tasks
    loss_fn = nn.CrossEntropyLoss()
    
    # 4. Training Loop
    for epoch in range(1, EPOCHS + 1):
        epoch_loss = 0.0
        correct_predictions = 0
        
        for idx, item in enumerate(dataset):
            query = item.get("query", "")
            expected_route = item.get("expected_route", "")
            
            # Map string to int (fallback to 4 if not found)
            target_idx = ACTIONS_MAP.get(expected_route, 4)
            
            # Reconstruct the input string exactly as the live router does
            state_text = f"Query: {query}\nVector Meta: {schemas['vector']}\nGraph Meta: {schemas['graph']}\nSQL Meta: {schemas['sql']}"
            
            # Tokenization
            inputs = tokenizer(state_text, return_tensors="pt", truncation=True, padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Target tensor (CrossEntropyLoss accepts 1D tensor of class indices)
            target_tensor = torch.tensor([target_idx], dtype=torch.long).to(device)
            
            # Forward Pass
            optimizer.zero_grad()
            outputs = model(**inputs)
            
            # Loss calculation on raw logits (softmax is internal to CrossEntropyLoss)
            loss = loss_fn(outputs.logits, target_tensor)
            
            # Backpropagation
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # Accuracy calculation (optional for display)
            pred_idx = torch.argmax(outputs.logits, dim=-1).item()
            if pred_idx == target_idx:
                correct_predictions += 1
                
        # End of Epoch Statistics
        avg_loss = epoch_loss / len(dataset)
        accuracy = (correct_predictions / len(dataset)) * 100
        print(f"[SFT Trainer] Epoch {epoch}/{EPOCHS} | Average Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}%")
        
    # 5. Save the trained model (Overwrites local weights)
    print(f"[SFT Trainer] Training complete! Saving weights to '{MODEL_NAME_OR_PATH}'...")
    model.save_pretrained(MODEL_NAME_OR_PATH)
    tokenizer.save_pretrained(MODEL_NAME_OR_PATH)
    print("[SFT Trainer] Model successfully saved. The Router is ready to use!")
    
    # 6. Update database state for warmup queries
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