import os
import json
import time
import streamlit as st
from router import app as langgraph_app
from ingest import ingest_document_to_vector, ingest_graph_from_json, ingest_generic_sql
from rl_logger import update_reward, update_log_final_metrics

# Page config
st.set_page_config(page_title="Hybrid RAG Dashboard", page_icon="🤖", layout="centered")

# --- SIDEBAR: File Upload & Status ---
with st.sidebar:
    st.header("Data Management")
    st.write("Upload new documents to populate the databases.")
    
    vector_files = st.file_uploader("Text/PDF Documents (Vector DB)", type=["txt", "pdf"], accept_multiple_files=True)
    graph_file = st.file_uploader("Structured Graph (JSON)", type=["json"])
    sql_files = st.file_uploader("Tabular Data (JSON/CSV for SQL DB)", type=["json", "csv"], accept_multiple_files=True)
    
    if st.button("Process Files"):
        if vector_files or graph_file or sql_files:
            os.makedirs("temp_uploads", exist_ok=True)
            
            # --- VECTOR DB BATCH PROCESSING ---
            if vector_files:
                from vector_retriever import VectorRetriever
                all_docs = []
                with st.spinner(f"Processing {len(vector_files)} documents for Vector DB..."):
                    for vf in vector_files:
                        temp_path = os.path.join("temp_uploads", vf.name)
                        with open(temp_path, "wb") as f:
                            f.write(vf.getbuffer())
                        try:
                            # Disabilitiamo la generazione metadati per ogni singolo file
                            docs = ingest_document_to_vector(temp_path, generate_metadata=False)
                            all_docs.extend(docs)
                        except Exception as e:
                            st.error(f"Error {vf.name}: {e}")
                    
                    # Generiamo i metadati una sola volta alla fine
                    if all_docs:
                        vr = VectorRetriever()
                        vr.generate_and_save_metadata(all_docs)
                        st.success(f"{len(vector_files)} documents successfully processed (Vector DB)!")
                        
            # --- GRAPH DB PROCESSING ---
            if graph_file:
                temp_path = os.path.join("temp_uploads", graph_file.name)
                with open(temp_path, "wb") as f:
                    f.write(graph_file.getbuffer())
                
                with st.spinner(f"Reading and ingesting {graph_file.name}..."):
                    try:
                        ingest_graph_from_json(temp_path)
                        st.success(f"{graph_file.name} successfully imported (Graph DB)!")
                    except Exception as e:
                        st.error(f"Error {graph_file.name}: {e}")
                        
            # --- SQL DB BATCH PROCESSING ---
            if sql_files:
                with st.spinner(f"Ingesting {len(sql_files)} tabular files..."):
                    for sf in sql_files:
                        temp_path = os.path.join("temp_uploads", sf.name)
                        with open(temp_path, "wb") as f:
                            f.write(sf.getbuffer())
                        
                        # Inferiamo il nome della tabella dal nome del file
                        inferred_table_name = os.path.splitext(sf.name)[0]
                        try:
                            ingest_generic_sql(temp_path, inferred_table_name)
                            st.success(f"{sf.name} successfully inserted into table '{inferred_table_name}'!")
                        except Exception as e:
                            st.error(f"Error {sf.name}: {e}")
        else:
            st.warning("Please select at least one file before clicking 'Process Files'.")
            

# -----------------------------

st.title("Hybrid Multi-Source RAG - Dashboard")

# Chat history initialization
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display past messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input (Modificato per il dominio medico)
if prompt := st.chat_input("Ask a question (e.g., 'How many patients are enrolled in phase 3 trials?')..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        final_state = None
        with st.status("Processing...", expanded=True) as status:
            try:
                st.write("Starting LangGraph router...")
                start_time = time.time()
                final_state = langgraph_app.invoke({"query": prompt})
                latency = time.time() - start_time
                
                st.write("**Vector Search Results:**")
                st.write(final_state.get("vector_results", "No results."))
                
                st.write("**Graph Search Results:**")
                st.write(final_state.get("graph_result", "No results."))
                
                st.write("**SQL Search Results:**")
                st.write(final_state.get("sql_context", "No results."))
                
                if final_state.get("ns_ids"):
                    st.write(f"**IDs Found ({len(final_state['ns_ids'])}):** {final_state['ns_ids']}")
                    st.write("**Lookup Results:**")
                    st.write(final_state.get("lookup_results", "No results."))
                else:
                    st.write("**Lookup:** Not triggered (no IDs found).")
                
                st.write("Starting Late Fusion Node...")
                
                status.update(label="Processing complete!", state="complete", expanded=False)
                
            except Exception as e:
                status.update(label="Error during processing", state="error", expanded=True)
                st.error(f"An error occurred: {e}")

        if final_state and "final_answer" in final_state:
            final_answer = final_state["final_answer"]
            st.markdown(final_answer)
            st.session_state.messages.append({"role": "assistant", "content": final_answer})
            
            if "log_id" in final_state:
                log_id = final_state["log_id"]
                
                inp = final_state.get("input_tokens", 0)
                out = final_state.get("output_tokens", 0)
                cost = final_state.get("total_cost", 0.0)
                
                with st.expander("📊 Execution Metrics (Performance)"):
                    st.caption(f"**Input Tokens:** {inp:,} | **Output Tokens:** {out:,} | **Total Cost:** ${cost:.6f} | **Latency:** {latency:.2f}s")
                
                update_log_final_metrics(
                    log_id=log_id, 
                    answer=final_state.get("final_answer", ""), 
                    input_tokens=inp, 
                    output_tokens=out, 
                    total_cost=cost,
                    latency=latency
                )
                
                def handle_feedback(val):
                    update_reward(log_id, val)
                    st.toast("Feedback recorded! Thank you for contributing to the training.", icon="✅")
                
                st.write("---")
                st.write("*Rate this answer to improve system routing:*")
                col1, col2, _ = st.columns([1, 1, 8])
                with col1:
                    st.button("👍 Helpful", on_click=handle_feedback, args=(1,), key=f"up_{log_id}")
                with col2:
                    st.button("👎 Not Helpful", on_click=handle_feedback, args=(0,), key=f"down_{log_id}")
                    
        elif final_state:
            fallback = "No final answer generated by the system."
            st.markdown(fallback)
            st.session_state.messages.append({"role": "assistant", "content": fallback})
