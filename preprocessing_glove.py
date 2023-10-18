import json
from datasets import Dataset
import pandas as pd
from enum import Enum
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
import torch

VEC_UNK = torch.tensor([0.0] * 50, dtype=torch.float32)
 
class DataType(Enum):
    TRAIN = 1
    TEST = 2

def open_file(data_type=None, file_path=None):
    f = None
    
    if data_type == DataType.Train:
        f = open("train-v1.1.json")
    elif data_type == DataType.TEST:
        f = open("dev-v1.1.json")
    elif file_path and not data_type:
        f = open(file_path)
    else:
        return []

    dataset = json.load(f, strict=False)["data"]
    return dataset
            
def dataset_parse(dataset):
    pre_dataframe = []
    for data in dataset:
        for p in data["paragraphs"]:
            for pqas in p["qas"]:
                for ans in pqas["answers"]:
#                     pre_dataframe.append(map(str, [p["context"], pqas["question"].strip(), ans["answer_start"], ans["text"]]))
                    pre_dataframe.append([p["context"], pqas["question"].strip(), { "answer_start": [int(ans["answer_start"])], "text": ans["text"] }])

    
    df = pd.DataFrame(pre_dataframe, columns=["context", "question", "answers"])

    return df

def preprocess_function(examples, tokenizer):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=384,
        truncation="only_second",
        return_offsets_mapping=True,
        padding="max_length",
    )

    offset_mapping = inputs.pop("offset_mapping")
    answers = examples["answers"]
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        answer = answers[i]
        start_char = answer["answer_start"][0]
        end_char = answer["answer_start"][0] + len(answer["text"][0])
        sequence_ids = inputs.sequence_ids(i)

        # Find the start and end of the context
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # If the answer is not fully inside the context, label it (0, 0)
        if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Otherwise it's the start and end token positions
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs

def tokenize(text):
    return word_tokenize(text)

def encode_answer(paragraph_tokens, answer_tokens):
    paragraph_len = len(paragraph_tokens)
    answer_len = len(answer_tokens)
    answer_ptr = 0
    ans_start = None
    ans_end = None
    
    for idx, paragraph_token in enumerate(paragraph_tokens):
        if paragraph_token == answer_tokens[answer_ptr]:
            answer_ptr += 1

            if ans_start == None:
                ans_start = idx
            
            if answer_ptr == answer_len:
                ans_end = idx
                break

        # When the tokens check are not part of the answer
        elif ans_start != None:
            ans_start = None
            ans_end = None
            answer_ptr = 0

    
    if ans_start == None or ans_end == None:
        return [0] * paragraph_len
    else:
        start = [0] * ans_start
        ans = [1] * answer_len
        end = [0] * (paragraph_len - ans_end)
    
        return start + ans + end
    
def embedding_init(embedding_path):
    embeddings = {}
    with open(embedding_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word, vector = values[0], values[1:]
            embeddings[word] = vector
            
    return embeddings
    
def embed(tokens, embeddings):
    vectors = []
    for token in tokens:
        token = token.lower()
        if token in embeddings:
            vectors.append(torch.tensor([float(e) for e in embeddings[token]]))
        else:
            vectors.append(torch.tensor(VEC_UNK))

    return torch.stack(vectors)

def pad_tensor(data):
    max_length = max(len(tensor) for tensor in data)
    padded_tensors = []
    
    for tensor in data:
        padding_size = max_length - len(tensor)
        if padding_size > 0:
            padding = torch.stack([VEC_UNK for i in range(padding_size)])
            padded_tensor = torch.cat([tensor, padding])
            padded_tensors.append(padded_tensor)
        else:
            padded_tensors.append(tensor)
    
    return torch.stack(padded_tensors)

def preprocess(data_type, file_path = None, tokenizer=None, embedding_path="./glove.6B.50d.txt"):
    embeddings = embedding_init(embedding_path)
    
    if data_type:
        data = open_file(data_type=data_type)
    else:
        data = open_file(file_path=file_path)
    
    data = dataset_parse(data)
    data = data.applymap(tokenize)
    data['answers'] = data.apply(lambda row: encode_answer(row['paragraph'], row['answers']), axis=1)
    data['question'] = pad_tensor(data["question"].map(lambda tokens : embed(tokens, embeddings)))
    data['context'] = pad_tensor(data["context"].map(lambda tokens : embed(tokens, embeddings)))
    
    
    tokenized = data.map(lambda x: preprocess_function(x, tokenizer), batched=True)
    
    return tokenized

    