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

# Load environment variables and support modules
load_dotenv()
from config import GROQ_API_KEY
from schema_extractor import get_compact_schemas

# Configuration Parameters
LAMBDA_PENALTY = 0.00005
MODEL_NAME_OR_PATH = "distilbert-base-multilingual-cased"

ACTIONS_MAP = {
    "no_retrieval": 0,
    "vector": 1,
    "graph": 2,
    "sql": 3,
    "multi": 4
}

def get_engine():
    """Retrieves the PostgreSQL database connection engine."""
    postgres_uri = os.getenv("POSTGRES_URI", "")
    if not postgres_uri or "TUAPASSWORD" in postgres_uri:
        raise ValueError("Missing or invalid POSTGRES_URI.")
    return create_engine(postgres_uri)

def ensure_schema_updates(engine):
    """
    Ensures the rl_logs table has columns for the final answer and training tracking,
    adding them dynamically if necessary.
    """
    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE rl_logs ADD COLUMN IF NOT EXISTS final_answer TEXT;"))
        conn.execute(text("ALTER TABLE rl_logs ADD COLUMN IF NOT EXISTS processed BOOLEAN DEFAULT FALSE;"))

def evaluate_with_llm(query: str, answer: str) -> float:
    """
    Uses LLM-as-a-Judge to evaluate answer correctness (from 0.0 to 1.0).
    If the answer is missing, it returns a neutral value.
    """
    if not answer or answer.strip() == "":
        return 0.5 
        
    llm = ChatGroq(
        temperature=0,
        groq_api_key=GROQ_API_KEY,
        model_name="llama-3.3-70b-versatile"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a strict and impartial judge. Evaluate the correctness and usefulness of the following answer regarding the user's question. Return EXCLUSIVELY a decimal number between 0.0 and 1.0. Do not add any text or explanations."),
        ("user", f"Question: {query}\nAnswer: {answer}")
    ])
    chain = prompt | llm
    
    try:
        res = chain.invoke({})
        score_text = res.content.strip()
        score = float(score_text)
        return max(0.0, min(1.0, score))
    except Exception as e:
        print(f"[LLM Judge] Parsing or connection error ({e}). Score: 0.5")
        return 0.5

def train_batch():
    """Fetches unprocessed logs, calculates rewards, and performs backpropagation on the model."""
    engine = get_engine()
    ensure_schema_updates(engine)
    
    print("[RL Trainer] PostgreSQL connection established. Searching for unprocessed logs...")
    with engine.begin() as conn:
        result = conn.execute(text("""
            SELECT id, query, final_answer, chosen_arm, input_tokens, output_tokens, user_reward 
            FROM rl_logs 
            WHERE processed = FALSE
        """))
        rows = result.fetchall()
        
    # Shuffle the dataset to prevent catastrophic forgetting
    random.shuffle(rows)
        
    if not rows:
        print("[RL Trainer] No new logs to process. Exiting.")
        return

    print(f"[RL Trainer] Found {len(rows)} samples for fine-tuning. Loading model...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME_OR_PATH)
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_NAME_OR_PATH, num_labels=5)
    model.to(device)
    model.train() # Sets the model to training mode (activates Dropout/Grad)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-5)
    loss_fn = nn.MSELoss()
    
    # Retrieve compact metadata (the state) exactly as the online Router does
    schemas = get_compact_schemas()
    
    total_loss = 0.0
    processed_ids = []
    
    for row in rows:
        log_id, query, final_answer, chosen_arm, inp, out, user_reward = row
        
        # 1. Accuracy Evaluation
        if user_reward is not None:
            # Explicit user feedback (mapping thumbs up to 1.0, thumbs down to 0.0)
            accuracy = 1.0 if user_reward > 0 else 0.0
            print(f"[Log {log_id}] User feedback detected: {accuracy}")
        else:
            # No feedback: invoke LLM as a judge
            accuracy = evaluate_with_llm(query, final_answer)
            print(f"[Log {log_id}] LLM Judge evaluation: {accuracy:.2f}")
            
        # 2. Compute Penalized Reward based on Computational Effort
        # Output tokens cost more computational resources (KV Cache) than inputs
        output_weight = 3 
        computational_effort = inp + (output_weight * out)
        
        # Weighted reward calculation
        reward = accuracy - (LAMBDA_PENALTY * computational_effort)
        
        # 3. Reconstruct original State
        state_text = f"Query: {query}\nVector Meta: {schemas['vector']}\nGraph Meta: {schemas['graph']}\nSQL Meta: {schemas['sql']}"
        inputs = tokenizer(state_text, return_tensors="pt", truncation=True, padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # 4. Forward Pass and Probability Calculation (Policy)
        optimizer.zero_grad()
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)[0]
        
        # Isolate the probability the model predicted for the actually pulled arm
        arm_idx = ACTIONS_MAP.get(chosen_arm, 4)
        predicted_prob = probs[arm_idx]
        
        # 5. Backpropagation (Loss Calculation and Weights Update)
        # We use the calculated reward as the desired target for the update
        target_tensor = torch.tensor(reward, dtype=torch.float32).to(device)
        
        loss = loss_fn(predicted_prob, target_tensor)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        processed_ids.append(log_id)
        
        print(f" -> Arm: {chosen_arm} | Effort: {computational_effort} | Reward R_a: {reward:.3f} | Pred: {predicted_prob.item():.3f} | Loss: {loss.item():.4f}\n")
        
    avg_loss = total_loss / len(rows)
    print(f"[RL Trainer] Fine-tuning completed. Average batch loss: {avg_loss:.4f}")
    
    # 6. Save the model for future inferences
    print(f"[RL Trainer] Overwriting weights in '{MODEL_NAME_OR_PATH}'...")
    model.save_pretrained(MODEL_NAME_OR_PATH)
    tokenizer.save_pretrained(MODEL_NAME_OR_PATH)
    
    # 7. Update state in the Database
    print(f"[RL Trainer] Closing processed logs...")
    with engine.begin() as conn:
        conn.execute(
            text("UPDATE rl_logs SET processed = TRUE WHERE id = ANY(:ids)"), 
            {"ids": processed_ids}
        )
        
    print("[RL Trainer] Operation successfully completed. The Router is now smarter!")

if __name__ == "__main__":
    train_batch()