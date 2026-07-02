import sys
print(f"[Debug] L'eseguibile Python in uso è: {sys.executable}")

import os
import re
from dotenv import load_dotenv
from langchain_community.utilities import SQLDatabase
from langchain_community.tools import QuerySQLDatabaseTool
from langchain_groq import ChatGroq
from langchain_classic.chains import create_sql_query_chain

load_dotenv()

def test_sql_connection():
    print("[SQL Node] Inizializzazione...")
    postgres_uri = os.getenv("POSTGRES_URI")
    
    if "TUAPASSWORD" in postgres_uri:
        print("[Errore] Sostituisci TUAPASSWORD nel file .env prima di procedere.")
        return

    try:
        # 1. Connessione al Database
        db = SQLDatabase.from_uri(postgres_uri)
        print(f"[SQL Node] Connesso a PostgreSQL. Dialetto: {db.dialect}")
        
        # 2. Estrazione Schema
        tables = db.get_usable_table_names()
        print(f"[SQL Node] Tabelle trovate: {tables}")
        
        print("[SQL Node] Inizializzazione catena Text-to-SQL...")
        llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
        execute_query = QuerySQLDatabaseTool(db=db)
        write_query = create_sql_query_chain(llm, db)
        
        # Creiamo un estrattore robusto per pulire l'output dell'LLM
        def clean_sql_output(text: str) -> str:
            # Estrae dai blocchi markdown sql
            match = re.search(r"```sql(.*?)```", text, re.DOTALL | re.IGNORECASE)
            if match:
                return match.group(1).strip()
            
            # Estrae da blocchi markdown generici
            match = re.search(r"```(.*?)```", text, re.DOTALL)
            if match:
                return match.group(1).strip()
                
            # Fallback su SQLQuery:
            if "SQLQuery:" in text:
                return text.split("SQLQuery:")[1].strip()
                
            return text.strip()

        # Inseriamo il pulitore tra la scrittura e l'esecuzione
        chain = write_query | clean_sql_output | execute_query
        print("[SQL Node] Catena configurata. Avvio test LLM...")

        query = "Qual è il prezzo medio delle auto usate vendute dal concessionario 'Texas Motors'?"
        print(f"\n[Test Domanda]: {query}")

        response = chain.invoke({"question": query})
        print(f"\n[Risultato dal Database]:\n{response}")
            
    except Exception as e:
        print(f"[SQL Node] Errore di connessione: {e}")

if __name__ == "__main__":
    test_sql_connection()