import os
import re
from dotenv import load_dotenv
from langchain_community.utilities import SQLDatabase
from langchain_community.tools import QuerySQLDatabaseTool
from langchain_classic.chains import create_sql_query_chain
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

load_dotenv()

SQL_PROMPT_TEMPLATE = PromptTemplate.from_template(
"""You are a Data Extractor expert in SQL (dialect {dialect}).
Your SOLE objective is to write the SQL query to extract the necessary data.

STRICT RULES (TAG PARADIGM):
1. DELEGATION OF RESPONSIBILITY:
   - If the user asks for a calculation, aggregation, or statistical operation (e.g., "how many", "average", "total", "group by"), YOU MUST use native SQL aggregation functions (COUNT, SUM, AVG, GROUP BY). The database is designed for fast math.
   - If the user asks for semantic reasoning over text (e.g., "summarize the reviews", "what do the reports say"), DO NOT use aggregations. Extract the raw text rows (e.g., SELECT text_field FROM...) so the downstream LLM can read them.
2. FIELDS: Always extract descriptive fields relevant to the question.
3. TEXT MATCHING: ALWAYS use ILIKE '%Name%' when filtering TEXT or VARCHAR strings. DO NOT use ILIKE on numeric or date columns.
4. AVOID HALLUCINATED FILTERS: DO NOT invent WHERE clauses. If the user simply asks to aggregate or group by a column (e.g., 'grouped by phase', 'by status'), you MUST NOT add a WHERE clause for that column. Just use the GROUP BY clause.
5. ID MATCHING: If the user provides an ID like 'NCT...', use WHERE studies.nct_id = 'NCT...'.
6. LIMIT: If extracting raw rows, always limit the query to a maximum of {top_k} results.
7. SYNTAX GUARDRAIL: Use EXACTLY the table names declared in the schema. Do not invent undeclared aliases in the FROM clause.

OUTPUT:
Return EXCLUSIVELY the valid SQL query. No explanations, no introductions, no markdown. Just the code.

Available tables:
{table_info}

Question: {input}
SQLQuery:"""
)

def clean_sql_output(text: str) -> str:
    match = re.search(r"```sql(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if match: return match.group(1).strip()
    match = re.search(r"```(.*?)```", text, re.DOTALL)
    if match: return match.group(1).strip()
    if "SQLQuery:" in text: return text.split("SQLQuery:")[1].strip()
    return text.strip()

class SQLRetriever:
    def __init__(self, model_name="llama-3.1-8b-instant", temperature=0):
        postgres_uri = os.getenv("POSTGRES_URI", "")
        if not postgres_uri or "TUAPASSWORD" in postgres_uri:
            self.chain = None
            return
            
        self.db = SQLDatabase.from_uri(postgres_uri, sample_rows_in_table_info=0)
        self.llm = ChatGroq(model=model_name, temperature=temperature)
        
        write_query = create_sql_query_chain(self.llm, self.db, prompt=SQL_PROMPT_TEMPLATE)
        execute_query = QuerySQLDatabaseTool(db=self.db)
        
        self.chain = write_query | clean_sql_output | execute_query

    def ask(self, question: str, callbacks=None) -> str:
        if not self.chain:
            return "Database SQL non configurato."
        return self.chain.invoke({"question": question}, config={"callbacks": callbacks})
