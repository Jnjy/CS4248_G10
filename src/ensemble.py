from typing import Literal, NamedTuple

import torch
from torch.nn import Module
from transformers import PreTrainedTokenizerBase


class AnswerPrediction(NamedTuple):
    start_logits: torch.Tensor
    start_softmax: torch.Tensor
    start_argmax: int
    end_logits: torch.Tensor
    end_softmax: torch.Tensor
    end_argmax: int
    weight: float


class EnsembleUnit:
    """Component representing one submodel of the ensemble model"""
    def __init__(self, model: Module, tokenizer: PreTrainedTokenizerBase, weight: float, device: str):
        model.to(device)
        self.model = model
        self.tokenizer = tokenizer
        self.weight = weight
        self.device = device

    def predict(self, question: str, context: str) -> AnswerPrediction:
        """
        return question answering prediction for the context span (tokenized by words)
        the shape of the tensors in AnswerPrediction are (1, d) where d = len(context.split())
        """
        unified_tokens = context.split()
        num_subwords_per_token = [len(self.tokenizer.encode(x, add_special_tokens=False)) for x in unified_tokens]
        last_encoded_subword_token = self.tokenizer.encode(unified_tokens[-1], add_special_tokens=False)[-1]
        d = len(unified_tokens)
        inputs = self.tokenizer(question, context, return_tensors="pt")
        inputs.to(self.device)
        inputs_idx = [0] * (d+1)
        inputs_idx[d] = (inputs.input_ids[0] == last_encoded_subword_token).nonzero().max().item() + 1
        for i in range(d-1, -1, -1):
            inputs_idx[i] = inputs_idx[i+1] - num_subwords_per_token[i]
        with torch.no_grad():
            outputs = self.model(**inputs)
        start_logits_list = [0.0] * d
        end_logits_list = [0.0] * d
        for i in range(d):
            # probability of starting/ending the answer at a word is equal to the max value
            # of the probabilities of starting/ending the answer at the sub-words of a word
            start_idx, end_idx = inputs_idx[i], inputs_idx[i+1]
            start_logits_list[i] = outputs.start_logits[0, start_idx:end_idx].max().item()
            end_logits_list[i] = outputs.end_logits[0, start_idx:end_idx].max().item()
        start_logits = torch.tensor([start_logits_list], dtype=torch.float32, device=self.device)
        start_softmax = torch.softmax(start_logits, dim=1)
        start_argmax = start_logits.argmax().item()
        end_logits = torch.tensor([end_logits_list], dtype=torch.float32, device=self.device)
        end_softmax = torch.softmax(end_logits, dim=1)
        end_argmax = end_logits.argmax().item()
        prediction = AnswerPrediction(
            start_logits, start_softmax, start_argmax, end_logits, end_softmax, end_argmax, self.weight
        )
        return prediction


class EnsembleModel:
    """Ensemble model for question answering models built from hugging face transformers"""
    def __init__(self, device: str = "cpu"):
        self.units: list[EnsembleUnit] = []
        self.total_weight = 0.0
        self.device = device

    def add_model(self, model: Module, tokenizer: PreTrainedTokenizerBase, weight: float = 1.0):
        unit = EnsembleUnit(model, tokenizer, weight, self.device)
        self.units.append(unit)
        self.total_weight += weight

    def predict(self, question: str, context: str, mode: Literal["soft", "hard"] = "soft") -> str:
        """
        mode = soft: combine the weighted softmax probabilities of each model
        mode = hard: combine the weighted final answer vote of each model
        """
        unified_tokens = context.split()
        d = len(unified_tokens)
        predictions = [unit.predict(question, context) for unit in self.units]
        start_vals = torch.tensor([[0.0] * d], dtype=torch.float32, device=self.device)
        end_vals = torch.tensor([[0.0] * d], dtype=torch.float32, device=self.device)
        for pred in predictions:
            if mode == "soft":
                start_vals = start_vals + (pred.weight / self.total_weight) * pred.start_softmax
                end_vals = end_vals + (pred.weight / self.total_weight) * pred.end_softmax
            elif mode == "hard":
                start_vals[0, pred.start_argmax] += pred.weight
                end_vals[0, pred.end_argmax] += pred.weight
        answer_start_idx = start_vals.argmax().item()
        answer_end_idx = end_vals.argmax().item() + 1
        answer = " ".join(unified_tokens[answer_start_idx:answer_end_idx])
        return answer
