import pandas as pd
import json
import re
import time
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# Il 70B è d'obbligo come Giudice per valutare la Faithfulness
JUDGE_MODEL = "llama-3.3-70b-versatile" 
CSV_INPUT_PATH = "eval_results_gpt.csv"
CSV_OUTPUT_PATH = "rag_metrics_results_gpt.csv"

def get_judge_llm():
    return ChatGroq(
        temperature=0.0,
        model_name=JUDGE_MODEL,
        max_retries=1 # Gestiamo i retry manualmente per il rate limit
    )

def evaluate_rag_metrics():
    print(f"⚖️ Avvio LLM-as-a-Judge ({JUDGE_MODEL}) su {CSV_INPUT_PATH}...")
    
    # 1. Sistema di Resume (Ripresa da dove si era interrotto)
    if os.path.exists(CSV_OUTPUT_PATH):
        print(f"📂 Trovato salvataggio precedente in {CSV_OUTPUT_PATH}. Ripresa dell'elaborazione...")
        df = pd.read_csv(CSV_OUTPUT_PATH)
    else:
        df = pd.read_csv(CSV_INPUT_PATH)
        # Filtro iniziale e creazione colonne
        df = df[df['Final_Answer'].notna() & (df['Final_Answer'] != "Errore di sistema")]
        df['Relevance_Score'] = pd.NA
        df['Fluency_Score'] = pd.NA
        df['Faithfulness_Score'] = pd.NA

    llm = get_judge_llm()
    
    eval_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an impartial and rigorous expert evaluator for a Medical RAG system. 
You will receive a User Question, the Raw Context extracted from databases, and the AI's Final Answer.

Evaluate the answer on three metrics, scoring from 1 to 5:
1. Relevance (1-5): Does the answer directly and completely address the user's question without adding irrelevant fluff?
2. Fluency (1-5): Is the language natural, grammatically correct, and free of system artifacts (e.g., JSON brackets, Python code)?
3. Faithfulness (1-5): Is the information in the answer STRICTLY supported by the provided Context? If the AI invents information, drugs, or numbers not present in the context, score 1.

OUTPUT FORMAT:
You MUST output strictly a valid JSON object and nothing else. No markdown, no explanations.
Example:
{{"relevance": 4, "fluency": 5, "faithfulness": 5}}"""),
        ("user", "Question: {question}\n\nContext: {context}\n\nAI Answer: {answer}")
    ])
    
    chain = eval_prompt | llm
    
    for index, row in df.iterrows():
        # Salta le righe già valutate con successo
        if pd.notna(row.get('Relevance_Score')) and pd.notna(row.get('Faithfulness_Score')):
            continue
            
        question = row['Query']
        answer = row['Final_Answer']
        context = str(row['Context'])
        
        print(f"\nValutazione Query [{index+1}/{len(df)}]: {question[:40]}...")
        
        success = False
        attempts = 0
        
        # Loop di Retry interno in caso di API Rate Limit
        while not success and attempts < 3:
            try:
                response = chain.invoke({"question": question, "context": context, "answer": answer})
                content = response.content.strip()
                
                match = re.search(r'\{.*\}', content, re.DOTALL)
                if match:
                    content = match.group(0)
                    
                scores = json.loads(content)
                
                df.at[index, 'Relevance_Score'] = float(scores.get('relevance', 0))
                df.at[index, 'Fluency_Score'] = float(scores.get('fluency', 0))
                df.at[index, 'Faithfulness_Score'] = float(scores.get('faithfulness', 0))
                
                print(f" -> Relevance: {scores.get('relevance')}/5 | Fluency: {scores.get('fluency')}/5 | Faithfulness: {scores.get('faithfulness')}/5")
                success = True
                
                # SALVATAGGIO INCREMENTALE (Blindatura dei dati)
                df.to_csv(CSV_OUTPUT_PATH, index=False)
                time.sleep(1.5)
                
            except Exception as e:
                error_msg = str(e).lower()
                attempts += 1
                if "rate limit" in error_msg or "429" in error_msg:
                    print(f" -> ⚠️ Rate limit raggiunto. Pausa di 60 secondi (Tentativo {attempts}/3)...")
                    time.sleep(60)
                else:
                    print(f" -> ⚠️ Errore critico JSON/API: {e}. Riprovo tra 5 sec...")
                    time.sleep(5)
                    
        if not success:
            print(f" ❌ Fallimento definitivo per la query {index+1}. Verrà saltata per ora.")
            # Il sistema passa alla successiva e salverà il resto
            
    print(f"\n{'='*50}")
    print("📊 RISULTATI MEDI RAG (Modello 20B)")
    print(f"Relevance:    {df['Relevance_Score'].mean():.2f} / 5")
    print(f"Fluency:      {df['Fluency_Score'].mean():.2f} / 5")
    print(f"Faithfulness: {df['Faithfulness_Score'].mean():.2f} / 5")
    print(f"File salvato in: {CSV_OUTPUT_PATH}")
    print(f"{'='*50}")

if __name__ == "__main__":
    evaluate_rag_metrics()