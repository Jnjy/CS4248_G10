# https://huggingface.co/docs/transformers/tasks/question_answering
import json
import pandas as pd
from datasets import Dataset
from enum import Enum
import os
CWD = os.getcwd()

class DataType(Enum):
    TRAIN = 1
    TEST = 2

def open_file(data_type=None, file_path=None):
    f = None
    
    if data_type == DataType.TRAIN:
        f = open("..\\squad\\train-v1.1.json")
    elif data_type == DataType.TEST:
        f = open("..\\squad\\dev-v1.1.json")
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
                    pre_dataframe.append([p["context"], pqas["question"].strip(), { "answer_start": [int(ans["answer_start"])], "text": ans["text"] }, pqas["id"]])

    
    df = pd.DataFrame(pre_dataframe, columns=["context", "question", "answers", "id"])
    ds = Dataset.from_pandas(df)

    return ds

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


def preprocess(data_type, file_path = None, tokenizer=None):
    if data_type:
        data = open_file(data_type=data_type)
    else:
        data = open_file(file_path=file_path)
    
    data = dataset_parse(data)
    
    #tokenized = data.map(lambda x: preprocess_function(x, tokenizer), batched=True)
    
    return data