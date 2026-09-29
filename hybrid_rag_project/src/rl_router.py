import torch
import numpy as np
import random
import os
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

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
            # 1. Dynamically calculate the absolute path
            # Navigates up from the 'src' folder (where rl_router.py is located) to the project root
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(current_dir)
            
            # Points directly to the folder saved during training
            local_model_path = os.path.join(project_root, "distilbert-base-multilingual-cased")
            
            # 2. Security check
            if os.path.exists(local_model_path):
                print(f"[RL Router] 🎯 Found locally trained weights in: {local_model_path}")
                model_to_load = local_model_path
            else:
                print(f"[RL Router] ⚠️ WARNING: Directory {local_model_path} not found.")
                print("[RL Router] Downloading base model (untrained) from HuggingFace...")
                model_to_load = "distilbert-base-multilingual-cased"
            
            # 3. Actual loading
            self.tokenizer = DistilBertTokenizer.from_pretrained(model_to_load)
            self.model = DistilBertForSequenceClassification.from_pretrained(
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
        """Creates a compact string concatenating the user query with the metadata."""
        return f"Query: {query}\nVector Meta: {vector_meta}\nGraph Meta: {graph_meta}\nSQL Meta: {sql_meta}"

    def predict_probabilities(self, state_text):
        """Uses the model to predict action probabilities without computing gradients."""
        if self.model is None or self.tokenizer is None:
            # Fallback to uniform probabilities
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
            # Exploration phase: random choice
            chosen_idx = random.randint(0, 4)
            is_exploration = True
        else:
            # Exploitation phase: model-based choice
            state_text = self.get_state_representation(query, vector_meta, graph_meta, sql_meta)
            probs = self.predict_probabilities(state_text)
            chosen_idx = int(np.argmax(probs))
            is_exploration = False
            
        action_name = self.actions.get(chosen_idx, "multi")
        return action_name, is_exploration