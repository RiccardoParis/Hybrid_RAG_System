import os
import streamlit as st
from router import app as langgraph_app
from ingest import ingest_document_to_vector, ingest_graph_from_json

# Configurazione della pagina (opzionale ma raccomandata per Dashboard)
st.set_page_config(page_title="Hybrid RAG Dashboard", page_icon="🤖", layout="centered")

# --- SIDEBAR: Upload File ---
with st.sidebar:
    st.header("Gestione Dati")
    st.write("Carica nuovi documenti o file JSON per popolare i database.")
    
    vector_file = st.file_uploader("Documento Testo/PDF (Vector DB)", type=["txt", "pdf"])
    graph_file = st.file_uploader("Grafo Strutturato (JSON)", type=["json"])
    
    if st.button("Elabora File"):
        if vector_file or graph_file:
            # Assicurati che la cartella temporanea esista
            os.makedirs("temp_uploads", exist_ok=True)
            
            if vector_file:
                temp_path = os.path.join("temp_uploads", vector_file.name)
                with open(temp_path, "wb") as f:
                    f.write(vector_file.getbuffer())
                
                with st.spinner(f"Elaborazione e ingestione di {vector_file.name}..."):
                    try:
                        ingest_document_to_vector(temp_path)
                        st.success(f"{vector_file.name} elaborato con successo (Vector DB)!")
                    except Exception as e:
                        st.error(f"Errore {vector_file.name}: {e}")
                        
            if graph_file:
                temp_path = os.path.join("temp_uploads", graph_file.name)
                with open(temp_path, "wb") as f:
                    f.write(graph_file.getbuffer())
                
                with st.spinner(f"Lettura e ingestione di {graph_file.name}..."):
                    try:
                        ingest_graph_from_json(temp_path)
                        st.success(f"{graph_file.name} importato con successo (Graph DB)!")
                    except Exception as e:
                        st.error(f"Errore {graph_file.name}: {e}")
        else:
            st.warning("Per favore, seleziona almeno un file prima di cliccare su 'Elabora File'.")
# -----------------------------

st.title("Hybrid Multi-Source RAG - Dashboard")

# Inizializzazione dello storico della chat nello stato della sessione
if "messages" not in st.session_state:
    st.session_state.messages = []

# Visualizzazione dei messaggi passati
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input dell'utente
if prompt := st.chat_input("Fai una domanda (es. 'Qual è lo stato di ns/server-01?' oppure 'Cos'è l'xG?')..."):
    # Aggiungi e mostra il messaggio dell'utente
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Contenitore per la risposta dell'assistente
    with st.chat_message("assistant"):
        final_state = None
        # Status per mostrare l'elaborazione in tempo reale
        with st.status("Elaborazione in corso...", expanded=True) as status:
            try:
                st.write("Avvio del router LangGraph...")
                # Invocazione del grafo LangGraph con lo stato iniziale
                final_state = langgraph_app.invoke({"query": prompt})
                
                # Mostra i risultati intermedi estratti dallo stato
                st.write("**Risultati Vector Search:**")
                st.write(final_state.get("vector_results", "Nessun risultato."))
                
                st.write("**Risultati Graph Search:**")
                st.write(final_state.get("graph_result", "Nessun risultato."))
                
                st.write("**Risultati SQL Search:**")
                st.write(final_state.get("sql_context", "Nessun risultato."))
                
                # Mostra i risultati del Lookup solo se ci sono stati ID rilevati
                if final_state.get("ns_ids"):
                    st.write(f"**ID Trovati ({len(final_state['ns_ids'])}):** {final_state['ns_ids']}")
                    st.write("**Risultati Lookup:**")
                    st.write(final_state.get("lookup_results", "Nessun risultato."))
                else:
                    st.write("**Lookup:** Non attivato (nessun ID trovato).")
                
                st.write("Avvio del Late Fusion Node in corso...")
                
                # Aggiorna lo status per segnalare il completamento
                status.update(label="Elaborazione completata!", state="complete", expanded=False)
                
            except Exception as e:
                status.update(label="Errore durante l'elaborazione", state="error", expanded=True)
                st.error(f"Si è verificato un errore: {e}")

        # Stampa la final_answer e salvala nello storico
        if final_state and "final_answer" in final_state:
            final_answer = final_state["final_answer"]
            st.markdown(final_answer)
            st.session_state.messages.append({"role": "assistant", "content": final_answer})
        elif final_state:
            fallback = "Nessuna risposta finale generata dal sistema."
            st.markdown(fallback)
            st.session_state.messages.append({"role": "assistant", "content": fallback})
