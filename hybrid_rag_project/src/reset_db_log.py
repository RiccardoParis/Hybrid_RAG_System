import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

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

load_dotenv()
engine = create_engine(os.getenv("POSTGRES_URI"))

with engine.begin() as conn:
    conn.execute(text("DROP TABLE IF EXISTS rl_logs;"))
    conn.execute(text(create_table_sql))
    
print("Table permanently reset. Goodbye old logs!")