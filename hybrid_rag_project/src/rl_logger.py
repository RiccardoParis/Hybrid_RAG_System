import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# Load environment to get POSTGRES_URI
load_dotenv()

def get_engine():
    postgres_uri = os.getenv("POSTGRES_URI", "")
    if not postgres_uri or "TUAPASSWORD" in postgres_uri:
        raise ValueError("Missing or invalid POSTGRES_URI configuration in .env")
    return create_engine(postgres_uri)

def init_log_table(engine):
    """Creates the rl_logs table if it does not exist."""
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS rl_logs (
        id SERIAL PRIMARY KEY,
        query TEXT,
        chosen_arm VARCHAR(50),
        user_reward INTEGER DEFAULT 0,
        final_answer TEXT,
        input_tokens INTEGER DEFAULT 0,
        output_tokens INTEGER DEFAULT 0,
        total_cost DOUBLE PRECISION DEFAULT 0.0,
        latency DOUBLE PRECISION DEFAULT 0.0,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """
    with engine.begin() as conn:
        conn.execute(text(create_table_sql))
        # Dynamic schema update (legacy compatibility)
        try:
            conn.execute(text("ALTER TABLE rl_logs DROP COLUMN IF EXISTS token_cost;"))
        except:
            pass
        conn.execute(text("ALTER TABLE rl_logs ADD COLUMN IF NOT EXISTS input_tokens INTEGER DEFAULT 0;"))
        conn.execute(text("ALTER TABLE rl_logs ADD COLUMN IF NOT EXISTS output_tokens INTEGER DEFAULT 0;"))
        conn.execute(text("ALTER TABLE rl_logs ADD COLUMN IF NOT EXISTS total_cost FLOAT DEFAULT 0.0;"))
        conn.execute(text("ALTER TABLE rl_logs ADD COLUMN IF NOT EXISTS latency FLOAT DEFAULT 0.0;"))

def log_interaction(query: str, chosen_arm: str, user_reward: int = None) -> int:
    """
    Inserts a new record. If user_reward is passed, it saves it immediately.
    """
    engine = get_engine()
    init_log_table(engine)
    
    insert_sql = """
    INSERT INTO rl_logs (query, chosen_arm, user_reward)
    VALUES (:query, :chosen_arm, :user_reward)
    RETURNING id;
    """
    with engine.begin() as conn:
        result = conn.execute(text(insert_sql), {
            "query": query,
            "chosen_arm": chosen_arm,
            "user_reward": user_reward
        })
        log_id = result.scalar()
        return log_id

def update_reward(log_id: int, reward_value: int):
    """
    Updates the score of an existing log when the user provides feedback.
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

def update_log_final_metrics(log_id: int, answer: str, input_tokens: int, output_tokens: int, total_cost: float, latency: float):
    """
    Saves performance metrics to the database once the processing is complete.
    """
    engine = get_engine()
    update_sql = """
    UPDATE rl_logs
    SET final_answer = :answer, 
        input_tokens = :input_tokens, 
        output_tokens = :output_tokens, 
        total_cost = :total_cost, 
        latency = :latency
    WHERE id = :log_id;
    """
    with engine.begin() as conn:
        conn.execute(text(update_sql), {
            "answer": answer, 
            "input_tokens": input_tokens, 
            "output_tokens": output_tokens, 
            "total_cost": total_cost, 
            "latency": latency, 
            "log_id": log_id
        })