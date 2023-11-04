import os
import json
import pandas as pd

from enum import Enum
from transformers import AutoTokenizer
from evaluate import load
from datasets import Dataset, load_dataset, DatasetDict

CWD = os.getcwd()

'''
Code largely follows huggingface @ https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/question_answering.ipynb#scrollTo=Xmr7_N7fMBIX
'''
class SQUAD():
    class DataType(Enum):
        TRAIN = 1
        TEST = 2

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

        self.train_squad = open_file("/dataset/train-v1.1.json")
        # self.train_data = dataset_parse(self.train_squad).train_test_split(test_size=0.2)
        self.train_data = dataset_parse(self.train_squad)

        self.dev_squad = open_file("/dataset/dev-v1.1.json")
        self.dev_data = dataset_parse(self.dev_squad)
        self.data = load_dataset('squad')
        # self.data = DatasetDict({"train": self.train_data, "validation": self.dev_data})

    def prepare_train_features(self, examples):
        # Some of the questions have lots of whitespace on the left, which is not useful and will make the
        # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
        # left whitespace
        examples["question"] = [q.lstrip() for q in examples["question"]]

        # Tokenize our examples with truncation and padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        tokenized_examples = self.tokenizer(
            examples["question"],
            examples["context"],
            truncation="only_second",
            max_length=384,
            stride=128,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        # The offset mappings will give us a map from token to character position in the original context. This will
        # help us compute the start_positions and end_positions.
        offset_mapping = tokenized_examples.pop("offset_mapping")

        # Let's label those examples!
        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []

        for i, offsets in enumerate(offset_mapping):
            # We will label impossible answers with the index of the CLS token.
            input_ids = tokenized_examples["input_ids"][i]
            cls_index = input_ids.index(self.tokenizer.cls_token_id)

            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            answers = examples["answers"][sample_index]
            # If no answers are given, set the cls_index as answer.
            if len(answers["answer_start"]) == 0:
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Start/end character index of the answer in the text.
                start_char = answers["answer_start"][0]
                end_char = start_char + len(answers["text"][0])

                # Start token index of the current span in the text.
                token_start_index = 0
                while sequence_ids[token_start_index] != 1:
                    token_start_index += 1

                # End token index of the current span in the text.
                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != 1:
                    token_end_index -= 1

                # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
                if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)
                else:
                    # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                    # Note: we could go after the last offset if the answer is the last word (edge case).
                    while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                        token_start_index += 1
                    tokenized_examples["start_positions"].append(token_start_index - 1)
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_examples["end_positions"].append(token_end_index + 1)

        return tokenized_examples

    def prepare_validation_features(self, examples):
        # Some of the questions have lots of whitespace on the left, which is not useful and will make the
        # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
        # left whitespace
        examples["question"] = [q.lstrip() for q in examples["question"]]

        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        tokenized_examples = self.tokenizer(
            examples["question"],
            examples["context"],
            truncation="only_second",
            max_length=384,
            stride=128,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

        # We keep the example_id that gave us this feature and we will store the offset mappings.
        tokenized_examples["example_id"] = []

        for i in range(len(tokenized_examples["input_ids"])):
            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = 1

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            tokenized_examples["example_id"].append(examples["id"][sample_index])

            # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
            # position is part of the context or not.
            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]

        return tokenized_examples

    def get_train_set(self):
        ds = self.data
        tokenized = ds.map(self.prepare_train_features, batched=True, remove_columns=ds["train"].column_names)

        return tokenized
    
    def get_test_set(self):
        ds = self.data
        validation_features = ds.map(
            self.prepare_validation_features,
            batched=True,
            remove_columns=ds["validation"].column_names
        )

        return validation_features
    
    def get_data(self):
        return self.data

def open_file(file_path):
    try:
        f = open(f'{CWD}{file_path}')
        dataset = json.load(f, strict=False)["data"]
        return dataset
    except:
        print("Invalid file path.")

def dataset_parse(dataset):
    ids = []
    titles = []
    contexts = []
    questions = []
    answers = []

    for data in dataset:
        for p in data["paragraphs"]:
            for pqas in p["qas"]:
                titles.append(data["title"])
                contexts.append(p["context"])
                ids.append(pqas["id"])
                questions.append(pqas["question"])

                text = []
                answer_start = []
                for ans in pqas["answers"]:
                    text.append(ans["text"])
                    answer_start.append(ans["answer_start"])
                
                ans = dict({'text': text, 'answer_start': answer_start})
                answers.append(ans)

    parsed_data = dict({'id': ids, 'title': titles, 'context': contexts, 'question': questions, 'answers': answers})

if __name__ == '__main__':
    model_checkpoint = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    squad = SQUAD(tokenizer)
    print(squad.get_train_set())
