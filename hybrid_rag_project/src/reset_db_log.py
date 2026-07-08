import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()
engine = create_engine(os.getenv("POSTGRES_URI"))
with engine.begin() as conn:
    conn.execute(text("TRUNCATE TABLE rl_logs RESTART IDENTITY;"))
print("Tabella resettata in modo permanente. Addio vecchi log!")