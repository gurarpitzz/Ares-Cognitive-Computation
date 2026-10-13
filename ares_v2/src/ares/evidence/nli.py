"""Natural Language Inference stance operator N(C, E_i)."""

from typing import List, Dict, Any, Tuple
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class NLIOperator:
    """NLI Stance Classifier for claim verification given evidence passages."""
    
    def __init__(self, model_name: str = "typeform/distilbert-base-uncased-mnli", device: str = None, local_files_only: bool = False):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, local_files_only=local_files_only).to(self.device)
        self.model.eval()
        
        # Discover label mapping
        id2label = self.model.config.id2label
        self.entailment_idx = 0
        self.contradiction_idx = 2
        self.neutral_idx = 1
        
        for idx, lbl in id2label.items():
            l_lower = lbl.lower()
            if "entail" in l_lower or "support" in l_lower:
                self.entailment_idx = int(idx)
            elif "contra" in l_lower or "refut" in l_lower:
                self.contradiction_idx = int(idx)
            elif "neut" in l_lower:
                self.neutral_idx = int(idx)
                
    @torch.no_grad()
    def predict_pair(self, premise: str, hypothesis: str) -> Dict[str, float]:
        """Predict NLI distribution for (premise=evidence, hypothesis=claim)."""
        inputs = self.tokenizer(
            premise,
            hypothesis,
            truncation=True,
            max_length=256,
            return_tensors="pt"
        ).to(self.device)
        
        logits = self.model(**inputs).logits
        probs = F.softmax(logits, dim=-1)[0].cpu().numpy()
        
        p_ent = float(probs[self.entailment_idx])
        p_con = float(probs[self.contradiction_idx])
        p_neu = float(probs[self.neutral_idx])
        
        soft_n = p_ent - p_con
        
        if p_ent > p_con and p_ent > p_neu and p_ent >= 0.5:
            hard_n = 1.0
        elif p_con > p_ent and p_con > p_neu and p_con >= 0.5:
            hard_n = -1.0
        else:
            hard_n = 0.0
            
        return {
            "p_entailment": p_ent,
            "p_contradiction": p_con,
            "p_neutral": p_neu,
            "soft_n": soft_n,
            "hard_n": hard_n
        }
        
    @torch.no_grad()
    def predict_batch(self, premises: List[str], hypotheses: List[str], batch_size: int = 32) -> List[Dict[str, float]]:
        """Batch inference for multiple (premise, hypothesis) pairs."""
        results = []
        for i in range(0, len(premises), batch_size):
            p_chunk = premises[i:i+batch_size]
            h_chunk = hypotheses[i:i+batch_size]
            
            inputs = self.tokenizer(
                p_chunk,
                h_chunk,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt"
            ).to(self.device)
            
            logits = self.model(**inputs).logits
            probs = F.softmax(logits, dim=-1).cpu().numpy()
            
            for row in probs:
                p_ent = float(row[self.entailment_idx])
                p_con = float(row[self.contradiction_idx])
                p_neu = float(row[self.neutral_idx])
                soft_n = p_ent - p_con
                
                if p_ent > p_con and p_ent > p_neu and p_ent >= 0.5:
                    hard_n = 1.0
                elif p_con > p_ent and p_con > p_neu and p_con >= 0.5:
                    hard_n = -1.0
                else:
                    hard_n = 0.0
                    
                results.append({
                    "p_entailment": p_ent,
                    "p_contradiction": p_con,
                    "p_neutral": p_neu,
                    "soft_n": soft_n,
                    "hard_n": hard_n
                })
        return results
