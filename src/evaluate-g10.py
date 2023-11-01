import os
import json

from evaluate import load
from datasets import load_dataset

CWD = os.getcwd()

def evaluate():
    dataset = load_dataset("squad")
    squad_metric = load("squad")

    references = dataset["validation"]
    references = process_ref(references)

    predictions = open(f'{CWD}/result/predictions.json')
    predictions = (json.load(predictions))
    predictions = process_pred(predictions)

    results = squad_metric.compute(predictions=predictions, references=references)

    print(results)
    
    with open(f'{CWD}/result/eval.json', 'w') as file:
        json.dump(results, file)
        file.close()

'''
example format for the parameters, will need to follow the format if wish to use this evalation function:
predictions = [{'prediction_text': '1976', 'id': '56e10a3be3433e1400422b22'}]
references = [{'answers': {'answer_start': [97], 'text': ['1976']}, 'id': '56e10a3be3433e1400422b22'}]
'''
def process_ref(references):
    new_references = list()

    for reference in references:
        new_reference = {'answers': reference['answers'], 'id': reference['id']}
        new_references.append(new_reference)

    return new_references

def process_pred(predictions):
    new_predictions = list()

    for pred in predictions.items():
        new_predictions.append({'prediction_text': pred[1], 'id': pred[0]})

    return new_predictions

if __name__ == '__main__':
    evaluate()
