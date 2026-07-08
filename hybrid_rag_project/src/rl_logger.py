import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# Carica l'ambiente per ottenere POSTGRES_URI
load_dotenv()

def get_engine():
    postgres_uri = os.getenv("POSTGRES_URI", "")
    if not postgres_uri or "TUAPASSWORD" in postgres_uri:
        raise ValueError("Configurazione POSTGRES_URI mancante o non valida in .env")
    return create_engine(postgres_uri)

def init_log_table(engine):
    """Crea la tabella rl_logs se non esiste."""
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS rl_logs (
        id SERIAL PRIMARY KEY,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        query TEXT NOT NULL,
        chosen_arm VARCHAR(50) NOT NULL,
        token_cost INTEGER NOT NULL,
        user_reward INTEGER NULL
    );
    """
    with engine.begin() as conn:
        conn.execute(text(create_table_sql))

def log_interaction(query: str, chosen_arm: str, token_cost: int, user_reward: int = None) -> int:
    """
    Inserisce un nuovo record. Se user_reward è passato, lo salva immediatamente.
    """
    engine = get_engine()
    init_log_table(engine)
    
    insert_sql = """
    INSERT INTO rl_logs (query, chosen_arm, token_cost, user_reward)
    VALUES (:query, :chosen_arm, :token_cost, :user_reward)
    RETURNING id;
    """
    with engine.begin() as conn:
        result = conn.execute(text(insert_sql), {
            "query": query,
            "chosen_arm": chosen_arm,
            "token_cost": token_cost,
            "user_reward": user_reward
        })
        log_id = result.scalar()
        return log_id

def update_reward(log_id: int, reward_value: int):
    """
    Aggiorna il punteggio di un log esistente quando l'utente fornisce feedback.
    """
    engine = get_engine()
    
    update_sql = """
    UPDATE rl_logs
    SET user_reward = :reward
    WHERE id = :log_id;
    """
    with engine.begin() as conn:
        conn.execute(text(update_sql), {
            "reward": reward_value,
            "log_id": log_id
        })

def update_log_final_metrics(log_id: int, answer: str, real_token_cost: int):
    """
    Aggiorna la colonna final_answer e il costo reale dei token di un log esistente.
    """
    engine = get_engine()
    
    update_sql = """
    UPDATE rl_logs
    SET final_answer = :answer, token_cost = :real_token_cost
    WHERE id = :log_id;
    """
    with engine.begin() as conn:
        conn.execute(text(update_sql), {
            "answer": answer,
            "real_token_cost": real_token_cost,
            "log_id": log_id
        })
