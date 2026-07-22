import torch
import numpy as np
import random
import os
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

class RLBanditRouter:
    def __init__(self):
        # Imposta il device (cuda se disponibile, altrimenti cpu)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[RL Router] Inizializzazione su device: {self.device}")
        
        # Dizionario di mapping per le classi (le 5 "braccia" del bandito)
        self.actions = {
            0: "no_retrieval",
            1: "vector",
            2: "graph",
            3: "sql",
            4: "multi"
        }
        
        
        try:
            # 1. Calcola il percorso assoluto in modo dinamico
            # Risale dalla cartella 'src' (dove si trova rl_router.py) alla root del progetto
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(current_dir)
            
            # Punta direttamente alla cartella salvata dal training
            local_model_path = os.path.join(project_root, "distilbert-base-multilingual-cased")
            
            # 2. Controllo di sicurezza
            if os.path.exists(local_model_path):
                print(f"[RL Router] 🎯 Trovati pesi addestrati localmente in: {local_model_path}")
                model_to_load = local_model_path
            else:
                print(f"[RL Router] ⚠️ ATTENZIONE: Cartella {local_model_path} non trovata.")
                print("[RL Router] Scarico il modello base (non addestrato) da HuggingFace...")
                model_to_load = "distilbert-base-multilingual-cased"
            
            # 3. Caricamento effettivo
            self.tokenizer = DistilBertTokenizer.from_pretrained(model_to_load)
            self.model = DistilBertForSequenceClassification.from_pretrained(
                model_to_load, 
                num_labels=5
            )
            
            self.model.to(self.device)
            print("[RL Router] Modello caricato con successo.")
            
        except Exception as e:
            print(f"[RL Router] Errore critico durante il caricamento del modello o tokenizer: {e}")
            self.tokenizer = None
            self.model = None

    def get_state_representation(self, query, vector_meta, graph_meta, sql_meta):
        """Crea una stringa compatta che concatena la query dell'utente con i metadati."""
        return f"Query: {query}\nVector Meta: {vector_meta}\nGraph Meta: {graph_meta}\nSQL Meta: {sql_meta}"

    def predict_probabilities(self, state_text):
        """Usa il modello per predire le probabilità delle azioni senza calcolare i gradienti."""
        if self.model is None or self.tokenizer is None:
            # Fallback a probabilità uniformi
            return np.array([0.2, 0.2, 0.2, 0.2, 0.2])
            
        inputs = self.tokenizer(state_text, return_tensors="pt", truncation=True, padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        try:
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
            return probs
        except Exception as e:
            print(f"[RL Router] Errore durante la predizione: {e}")
            return np.array([0.2, 0.2, 0.2, 0.2, 0.2])

    def choose_arm(self, query, epsilon=0.2, vector_meta="", graph_meta="", sql_meta=""):
        """Sceglie l'azione da eseguire basandosi su epsilon-greedy."""
        if random.random() < epsilon:
            # Fase di Explorazione: scelta random
            chosen_idx = random.randint(0, 4)
            is_exploration = True
        else:
            # Fase di Exploitation: scelta basata sul modello
            state_text = self.get_state_representation(query, vector_meta, graph_meta, sql_meta)
            probs = self.predict_probabilities(state_text)
            chosen_idx = int(np.argmax(probs))
            is_exploration = False
            
        action_name = self.actions.get(chosen_idx, "multi")
        return action_name, is_exploration
    
