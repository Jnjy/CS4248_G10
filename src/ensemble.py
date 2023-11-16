import json
import os
import logging
import subprocess

from typing import Literal, NamedTuple

import torch
from torch.nn import Module
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, RobertaTokenizerFast, PreTrainedTokenizerFast
from tqdm import tqdm

CWD = os.getcwd()
logger = logging.getLogger()

class AnswerPrediction(NamedTuple):
    start_logits: torch.Tensor
    start_softmax: torch.Tensor
    start_argmax: int
    end_logits: torch.Tensor
    end_softmax: torch.Tensor
    end_argmax: int
    weight: float


class WordTokenizationResult(NamedTuple):
    unified_tokens: list[str]
    num_word_tokens: int
    word_span_end: list[int]


class EnsembleUnit:
    """Component representing one submodel of the ensemble model"""
    def __init__(self, model: Module, tokenizer: PreTrainedTokenizerFast, weight: float, device: str):
        model.to(device)
        self.model = model
        self.tokenizer = tokenizer
        self.weight = weight
        self.device = device

    def predict(self, question: str, context: str, word_tokens: WordTokenizationResult) -> AnswerPrediction:
        """
        return question answering prediction for the context span (tokenized by words)
        the shape of the tensors in AnswerPrediction are (1, d) where d = word_tokens.num_word_tokens
        """
        # map sub-word tokenization to word tokenization
        num_word_tokens = word_tokens.num_word_tokens
        word_span_end = word_tokens.word_span_end
        inputs = self.tokenizer(question, context, return_tensors="pt", truncation=True)
        inputs.to(self.device)
        inputs_idx = [0] * (num_word_tokens+1)
        num_subword_tokens = len(inputs.word_ids())
        special_count = 0
        word_idx = -1
        for i in range(num_subword_tokens):
            char_span = inputs.token_to_chars(i)
            if char_span is None:
                special_count += 1
                continue
            if special_count < 2:
                # skip the question part
                continue
            if word_idx < 0:
                # initialize start of first word
                word_idx = 0
                inputs_idx[word_idx] = i
            char_start_idx, _ = char_span
            while char_start_idx >= word_span_end[word_idx]:
                # this token is the start of next word
                word_idx += 1
                inputs_idx[word_idx] = i
        for i in range(word_idx+1, num_word_tokens+1):
            inputs_idx[i] = num_subword_tokens
        # get original prediction logits
        with torch.no_grad():
            outputs = self.model(**inputs)
        # convert to unified word tokenization prediction logits
        start_logits_list = [0.0] * num_word_tokens
        end_logits_list = [0.0] * num_word_tokens
        for i in range(num_word_tokens):
            # probability of starting/ending the answer at a word is equal to the max value
            # of the probabilities of starting/ending the answer at the sub-words of a word
            start_idx, end_idx = inputs_idx[i], inputs_idx[i+1]
            if start_idx >= end_idx:
                continue
            start_logits_list[i] = outputs.start_logits[0, start_idx:end_idx].max().item()
            end_logits_list[i] = outputs.end_logits[0, start_idx:end_idx].max().item()
        # calculate probabilities
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

    def add_model(self, model: Module, tokenizer: PreTrainedTokenizerFast, weight: float = 1.0):
        unit = EnsembleUnit(model, tokenizer, weight, self.device)
        self.units.append(unit)
        self.total_weight += weight

    def predict(self, question: str, context: str, mode: Literal["soft", "hard"] = "soft") -> str:
        """
        mode = soft: combine the weighted softmax probabilities of each model
        mode = hard: combine the weighted final answer vote of each model
        """
        # Preprocess inputs (remove tab/newline)
        question = " ".join(question.split())
        context = " ".join(context.split())
        # Word tokenization
        unified_tokens = context.split()
        num_word_tokens = len(unified_tokens)
        word_span_end = [0] * num_word_tokens
        word_idx = 0
        for i in range(len(context)):
            if context[i] != " ":
                continue
            word_span_end[word_idx] = i
            word_idx += 1
        for i in range(word_idx, num_word_tokens):
            word_span_end[i] = len(context)
        word_tokens = WordTokenizationResult(unified_tokens, num_word_tokens, word_span_end)
        # Ensemble predictions
        predictions = [unit.predict(question, context, word_tokens) for unit in self.units]
        start_vals = torch.tensor([[0.0] * num_word_tokens], dtype=torch.float32, device=self.device)
        end_vals = torch.tensor([[0.0] * num_word_tokens], dtype=torch.float32, device=self.device)
        for pred in predictions:
            if mode == "soft":
                start_vals = start_vals + (pred.weight / self.total_weight) * pred.start_softmax
                end_vals = end_vals + (pred.weight / self.total_weight) * pred.end_softmax
            elif mode == "hard":
                start_vals[0, pred.start_argmax] += pred.weight
                end_vals[0, pred.end_argmax] += pred.weight
        # Convert to answer string
        answer_start_idx = start_vals.argmax().item()
        answer_end_idx = end_vals.argmax().item() + 1
        answer = " ".join(unified_tokens[answer_start_idx:answer_end_idx])
        return answer

    def generate_prediction_json(self, inpath: str, outpath: str, mode: Literal["soft", "hard"] = "soft"):
        """
        mode = soft: combine the weighted softmax probabilities of each model
        mode = hard: combine the weighted final answer vote of each model
        """
        contexts, queries = [], []
        with open(inpath, mode="r") as infile:
            dataset = json.load(infile, strict=False)["data"]
            for data in dataset:
                for paragraph in data["paragraphs"]:
                    contexts.append(paragraph["context"])
                    context_id = len(contexts) - 1
                    for qa in paragraph["qas"]:
                        id = qa["id"]
                        question = qa["question"]
                        queries.append((id, question, context_id))
        result = {}
        for query in tqdm(queries):
            id, question, context_id = query
            context = contexts[context_id]
            result[id] = self.predict(question, context, mode)
        with open(outpath, mode="w") as outfile:
            json.dump(result, outfile)

def run_soft_ensemble():
    ''' 1. load models '''
    # distilbert
    logger.info("Load Model: Distilbert")
    distilbert_name = "jeffnjy/distilbert-base-test"
    distilbert_tokenizer = AutoTokenizer.from_pretrained(distilbert_name)
    distilbert_model = AutoModelForQuestionAnswering.from_pretrained(distilbert_name)
    
    # albert
    logger.info("Load Model: Albert")
    albert_name = "jeffnjy/albert-base-test"
    albert_tokenizer = AutoTokenizer.from_pretrained(albert_name)
    albert_model = AutoModelForQuestionAnswering.from_pretrained(albert_name)
    
    # bert
    logger.info("Load Model: Bert")
    bert_name = "jeffnjy/bert-base-test"
    bert_tokenizer = AutoTokenizer.from_pretrained(bert_name)
    bert_model = AutoModelForQuestionAnswering.from_pretrained(bert_name)
    
    # roberta
    logger.info("Load Model: Roberta")
    roberta_name = "jeffnjy/roberta-base-test"
    roberta_tokenizer = RobertaTokenizerFast.from_pretrained(roberta_name)
    roberta_model = AutoModelForQuestionAnswering.from_pretrained(roberta_name)

    
    ''' 2. create ensemble model '''
    soft_weights = [(1, 1, 1, 1), (0.5, 0.5, 0.5, 1), (0.1, 0.1, 0.1, 1), (0.1, 0.2, 0.1, 1), (0.2, 0.1, 0.1, 1), (0.1, 0.5, 0.1, 1),
                    (0.2, 0.2, 0.1, 1), (0.2, 0.3, 0.1, 1), (0.2, 0.4, 0.1, 1), (0.2, 0.5, 0.1, 1), (0.2, 1, 0.1, 1), 
                    (0.5, 0.1, 0.1, 1), (0.5, 0.2, 0.1, 1), (0.5, 0.5, 0.1, 1), (0.5, 1, 0.1, 1), (20.1425, 21.4624, 18.985, 20.7625),
                    (0, 0.4, 0, 1), (0.2, 0.4, 0, 1), (0.2, 0.4, 0.2, 1), (0.2, 0.4, 0.3, 1),
                    (0.2, 0.4, 0.4, 1), (0.3, 0.4, 0.1, 1), (0.3, 0.4, 0.2, 1), (0.3, 0.4, 0.3, 1), (0.3, 0.4, 0.4, 1), 
                    (0.4, 0.4, 0.1, 1), (0.4, 0.4, 0.2, 1), (0.4, 0.4, 0.3, 1), (0.4, 0.4, 0.4, 1)]

    for soft_weight in soft_weights:
        bert_sw = soft_weight[0]
        albert_sw = soft_weight[1]
        distilbert_sw = soft_weight[2]
        roberta_sw = soft_weight[3]

        model_weights = {
            "bert": bert_sw,
            "albert": albert_sw,
            "distilbert": distilbert_sw,
            "roberta": roberta_sw,
        }
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
        print("running ensemble-soft for ", model_weights)
        
        ensemble_model = EnsembleModel(device=device)
        ensemble_model.add_model(model=distilbert_model, tokenizer=distilbert_tokenizer, weight=model_weights["distilbert"])
        ensemble_model.add_model(model=albert_model, tokenizer=albert_tokenizer, weight=model_weights["albert"])
        ensemble_model.add_model(model=roberta_model, tokenizer=roberta_tokenizer, weight=model_weights["roberta"])
        ensemble_model.add_model(model=bert_model, tokenizer=bert_tokenizer, weight=model_weights["bert"])
        
        ''' 3. make predictions '''
        logger.info("Making predictions")
        inpath = f'{CWD}/dataset/dev-v1.1.json'
        outpath = f'{CWD}/result/prediction/ensemble-soft/ensemble_soft_{bert_sw}-{albert_sw}-{distilbert_sw}-{roberta_sw}.json'
        ensemble_model.generate_prediction_json(inpath, outpath, mode="soft")

        list_files = subprocess.run(["python3",
            f'{CWD}/src/evaluate-v2.0.py',
            f'{CWD}/dataset/dev-v1.1.json', outpath], text=True)

def run_hard_ensemble():
    ''' 1. load models '''
    # distilbert
    logger.info("Load Model: Distilbert")
    distilbert_name = "jeffnjy/distilbert-base-test"
    distilbert_tokenizer = AutoTokenizer.from_pretrained(distilbert_name)
    distilbert_model = AutoModelForQuestionAnswering.from_pretrained(distilbert_name)
    
    # albert
    logger.info("Load Model: Albert")
    albert_name = "jeffnjy/albert-base-test"
    albert_tokenizer = AutoTokenizer.from_pretrained(albert_name)
    albert_model = AutoModelForQuestionAnswering.from_pretrained(albert_name)
    
    # bert
    logger.info("Load Model: Bert")
    bert_name = "jeffnjy/bert-base-test"
    bert_tokenizer = AutoTokenizer.from_pretrained(bert_name)
    bert_model = AutoModelForQuestionAnswering.from_pretrained(bert_name)
    
    # roberta
    logger.info("Load Model: Roberta")
    roberta_name = "jeffnjy/roberta-base-test"
    roberta_tokenizer = RobertaTokenizerFast.from_pretrained(roberta_name)
    roberta_model = AutoModelForQuestionAnswering.from_pretrained(roberta_name)

    
    ''' 2. create ensemble model '''
    hard_weights = [(1, 1, 1, 1), (0.5, 0.5, 0.5, 1), (0.1, 0.1, 0.1, 1), (0.1, 0.2, 0.1, 1), 
                    (0.2, 0.1, 0.1, 1), (0.1, 0.5, 0.1, 1), (0.2, 0.2, 0.1, 1), (0.2, 0.3, 0.1, 1), 
                    (0.2, 0.4, 0.1, 1), (0.2, 0.5, 0.1, 1), (0.2, 1, 0.1, 1), (0.5, 0.1, 0.1, 1), 
                    (0.5, 0.2, 0.1, 1), (0.5, 0.5, 0.1, 1), (0.5, 1, 0.1, 1), (20.1425, 21.4624, 18.985, 20.7625),
                    (0, 0.4, 0, 1), (0.2, 0.4, 0, 1), (0.2, 0.4, 0.2, 1), (0.2, 0.4, 0.3, 1),
                    (0.2, 0.4, 0.4, 1), (0.3, 0.4, 0.1, 1), (0.3, 0.4, 0.2, 1), (0.3, 0.4, 0.3, 1), 
                    (0.3, 0.4, 0.4, 1), (0.4, 0.4, 0.1, 1), (0.4, 0.4, 0.2, 1), (0.4, 0.4, 0.3, 1), 
                    (0.4, 0.4, 0.4, 1)]
    
    for hard_weight in hard_weights:
        bert_hw = hard_weight[0]
        albert_hw = hard_weight[1]
        distilbert_hw = hard_weight[2]
        roberta_hw = hard_weight[3]
        
        model_weights = {
            "bert": bert_hw,
            "albert": albert_hw,
            "distilbert": distilbert_hw,
            "roberta": roberta_hw,
        }
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

        print("running ensemble-hard for ", model_weights)
        
        ensemble_model = EnsembleModel(device=device)
        ensemble_model.add_model(model=distilbert_model, tokenizer=distilbert_tokenizer, weight=model_weights["distilbert"])
        ensemble_model.add_model(model=albert_model, tokenizer=albert_tokenizer, weight=model_weights["albert"])
        ensemble_model.add_model(model=roberta_model, tokenizer=roberta_tokenizer, weight=model_weights["roberta"])
        ensemble_model.add_model(model=bert_model, tokenizer=bert_tokenizer, weight=model_weights["bert"])
        
        ''' 3. make predictions '''
        logger.info("Making predictions")
        inpath = f'{CWD}/dataset/dev-v1.1.json'
        outpath = f'{CWD}/result/prediction/ensemble-hard/ensemble_hard_{bert_hw}-{albert_hw}-{distilbert_hw}-{roberta_hw}.json'
        ensemble_model.generate_prediction_json(inpath, outpath, mode="hard")

        list_files = subprocess.run(["python3",
            f'{CWD}/src/evaluate-v2.0.py',
            f'{CWD}/dataset/dev-v1.1.json', outpath], text=True)

if __name__ == '__main__':
    run_soft_ensemble()
    # run_hard_ensemble()
    
