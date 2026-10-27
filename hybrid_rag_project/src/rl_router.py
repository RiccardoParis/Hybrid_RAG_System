import torch
import numpy as np
import random
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class RLBanditRouter:
    def __init__(self):
        # Set the device (cuda if available, otherwise cpu)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[RL Router] Initializing on device: {self.device}")
        
        # Mapping dictionary for classes (the 5 "arms" of the bandit)
        self.actions = {
            0: "no_retrieval",
            1: "vector",
            2: "graph",
            3: "sql",
            4: "multi"
        }
        
        try:
            # Prende il percorso assoluto della cartella 'src' (dove si trova rl_router.py)
            current_dir = os.path.dirname(os.path.abspath(__file__))
            
            # Cerca i pesi direttamente dentro 'src'
            local_model_path = os.path.join(current_dir, "distilroberta-base-router")
            
            if os.path.exists(local_model_path):
                print(f"[RL Router] 🎯 Found locally trained weights in: {local_model_path}")
                model_to_load = local_model_path
            else:
                print(f"[RL Router] ⚠️ WARNING: Directory {local_model_path} not found.")
                print("[RL Router] Downloading base English model from HuggingFace...")
                model_to_load = "distilroberta-base"
            
            # Dynamic Loading using AutoClasses for flexibility
            self.tokenizer = AutoTokenizer.from_pretrained(model_to_load)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_to_load, 
                num_labels=5
            )
            
            self.model.to(self.device)
            print("[RL Router] Model successfully loaded.")
            
        except Exception as e:
            print(f"[RL Router] Critical error loading model or tokenizer: {e}")
            self.tokenizer = None
            self.model = None

    def get_state_representation(self, query, vector_meta, graph_meta, sql_meta):
        """
        Creates a compact string. Schemas are placed first as context, 
        and the user query is placed at the very end to maximize attention focus.
        """
        return f"[USER QUERY]\n{query}\n\n[AVAILABLE SCHEMAS]\nVector: {vector_meta}\nGraph: {graph_meta}\nSQL: {sql_meta}"

    def predict_probabilities(self, state_text):
        """Uses the model to predict action probabilities without computing gradients."""
        if self.model is None or self.tokenizer is None:
            return np.array([0.2, 0.2, 0.2, 0.2, 0.2])
            
        inputs = self.tokenizer(state_text, return_tensors="pt", truncation=True, padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        try:
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
            return probs
        except Exception as e:
            print(f"[RL Router] Error during prediction: {e}")
            return np.array([0.2, 0.2, 0.2, 0.2, 0.2])

    def choose_arm(self, query, epsilon=0.2, vector_meta="", graph_meta="", sql_meta=""):
        """Chooses the action to execute based on epsilon-greedy strategy."""
        if random.random() < epsilon:
            chosen_idx = random.randint(0, 4)
            is_exploration = True
        else:
            state_text = self.get_state_representation(query, vector_meta, graph_meta, sql_meta)
            probs = self.predict_probabilities(state_text)
            chosen_idx = int(np.argmax(probs))
            is_exploration = False
            
        action_name = self.actions.get(chosen_idx, "multi")
        return action_name, is_exploration