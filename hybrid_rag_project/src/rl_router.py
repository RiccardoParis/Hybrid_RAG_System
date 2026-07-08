import torch
import numpy as np
import random
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
            # Carica il tokenizer e il modello pre-addestrato
            model_name = "distilbert-base-multilingual-cased"
            print(f"[RL Router] Caricamento modello '{model_name}'...")
            
            self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
            
            # Assicurati di specificare num_labels=5 per la classificazione a 5 vie
            self.model = DistilBertForSequenceClassification.from_pretrained(
                model_name, 
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
