import os
import json

from argparse import ArgumentParser
from transformers import TrainingArguments, AutoTokenizer
from preprocess import SQUAD

CWD = os.getcwd()

def main():
    # parser = ArgumentParser(description="Parser")
    # parser.add_argument("path_to_json", help="Path to the json", type=str)
    # args = parser.parse_args()

    f = open(CWD + "/dataset/train-v1.1.json")
    dataset = json.load(f, strict=False)["data"]

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    squad = SQUAD(tokenizer)

    tokenized_dataset = squad.get_train_set(dataset)

if __name__ == '__main__':
    main()

