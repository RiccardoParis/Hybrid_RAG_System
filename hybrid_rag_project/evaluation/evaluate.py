import json
import csv
import os
import re
import time
import random
import sys

# Aggiunge la cartella 'src' al path per permettere le importazioni
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from config import GROQ_API_KEY
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from router import app as langgraph_app

EVAL_PROMPT = PromptTemplate(
    input_variables=["question", "ground_truth", "context", "answer"],
    template="""You are an impartial expert judge evaluating a Retrieval-Augmented Generation (RAG) system.
Evaluate the following System Answer based on the User Question, the Ground Truth, and the Retrieved Context.

Question: {question}
Ground Truth: {ground_truth}
Retrieved Context: {context}
System Answer: {answer}

Rate the System Answer on a scale from 1 to 10 for the following three metrics:
- Relevance: How well does the answer address the question using the context?
- Faithfulness: Are all facts in the answer present in the retrieved context? (Penalize heavily for hallucinations).
- Fluency: Is the answer grammatically correct, natural, and coherent?

Output ONLY a valid JSON object with the keys: "relevance", "faithfulness", "fluency". Provide the raw integer scores (1-10). Do not output any markdown formatting like ```json."""
)

def evaluate_pipeline():
    benchmark_file = "benchmark_data.json"
    output_csv = "evaluation_results.csv"

    # Gestione path a seconda di dove viene lanciato lo script
    if not os.path.exists(benchmark_file):
        if os.path.exists("../benchmark_data.json"):
            benchmark_file = "../benchmark_data.json"
        else:
            print(f"File {benchmark_file} non trovato. Assicurati di avviarlo dalla cartella principale.")
            return

    with open(benchmark_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    random.seed(42)
    data = random.sample(data, min(100, len(data)))

    print(f"Inizio valutazione per {len(data)} domande...")

    llm_judge = ChatGroq(
        temperature=0, 
        groq_api_key=GROQ_API_KEY, 
        model_name="llama-3.3-70b-versatile"
    )
    
    eval_chain = EVAL_PROMPT | llm_judge

    fieldnames = ['question', 'category', 'relevance', 'faithfulness', 'fluency']
    evaluated_questions = set()
    total_relevance = 0.0
    total_faithfulness = 0.0
    total_fluency = 0.0
    count = 0

    if os.path.exists(output_csv):
        with open(output_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                q = row.get('question')
                if q:
                    evaluated_questions.add(q)
                    try:
                        total_relevance += float(row.get('relevance', 0))
                        total_faithfulness += float(row.get('faithfulness', 0))
                        total_fluency += float(row.get('fluency', 0))
                        count += 1
                    except ValueError:
                        pass
        print(f"Ripristino da checkpoint: trovate {len(evaluated_questions)} domande già valutate.")
    else:
        with open(output_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

    for idx, item in enumerate(data, 1):
        category = item.get("category", "Uncategorized")
        question = item.get("question", "")
        ground_truth = item.get("ground_truth", "")

        if question in evaluated_questions:
            print(f"\n[{idx}/{len(data)}] Domanda saltata (già valutata): {question}")
            continue

        print(f"\n[{idx}/{len(data)}] Analisi Domanda: {question}")
        
        # 1. Recupero dal sistema RAG
        try:
            state = langgraph_app.invoke({"query": question})
            generated_answer = state.get("final_answer", "")
            
            # Creazione di un contesto combinato dalle fonti estratte
            vector_res = state.get("vector_results", "")
            graph_res = state.get("graph_result", "")
            lookup_res = state.get("lookup_results", {})
            combined_context = f"VectorRAG: {vector_res}\nGraphRAG: {graph_res}\nLookup: {lookup_res}"
            
        except Exception as e:
            print(f"Errore RAG: {e}")
            generated_answer = ""
            combined_context = ""

        # 2. Valutazione LLM-as-a-Judge
        try:
            eval_response = eval_chain.invoke({
                "question": question,
                "ground_truth": ground_truth,
                "context": combined_context,
                "answer": generated_answer
            })
            
            content = eval_response.content.strip()
            # Estrazione sicura del blocco JSON usando Regex, aggirando eventuali Markdown format
            match = re.search(r'\{.*\}', content, re.DOTALL)
            if match:
                scores = json.loads(match.group(0))
                # Normalizzazione dei punteggi 1-10 sulla scala 0-1 (come nel paper)
                rel = scores.get("relevance", 0) / 10.0
                fai = scores.get("faithfulness", 0) / 10.0
                flu = scores.get("fluency", 0) / 10.0
            else:
                rel, fai, flu = 0.0, 0.0, 0.0
                print(f"Errore parsing del JSON fornito dal giudice:\n{content}")
                
        except Exception as e:
            print(f"Errore Judge: {e}")
            rel, fai, flu = 0.0, 0.0, 0.0

        # Accumulo per le medie
        total_relevance += rel
        total_faithfulness += fai
        total_fluency += flu
        count += 1
        
        # Salvataggio immediato in append
        with open(output_csv, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow({
                "question": question,
                "category": category,
                "relevance": rel,
                "faithfulness": fai,
                "fluency": flu
            })
            
        # Pausa di raffreddamento API per evitare Rate Limit (es. TPM Groq)
        time.sleep(15)

    # 4. Stampa finale delle medie
    print(f"\nValutazione completata. Dati salvati in: {output_csv}")
    
    if count > 0:
        avg_rel = total_relevance / count
        avg_fai = total_faithfulness / count
        avg_flu = total_fluency / count
        print("\n=== MEDIE FINALI (SCALA 0.0 - 1.0) ===")
        print(f"Relevance:    {avg_rel:.3f}")
        print(f"Faithfulness: {avg_fai:.3f}")
        print(f"Fluency:      {avg_flu:.3f}")

if __name__ == "__main__":
    evaluate_pipeline()
