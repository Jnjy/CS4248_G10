# from sentence_transformers import SentenceTransformer
from simpletransformers.question_answering import QuestionAnsweringModel
from preprocessing_huggingface import *
import json
import os
import torch
CWD = os.getcwd()


# structure the data into a format that the model can accept (https://simpletransformers.ai/docs/qa-data-formats/)
def process_dataset(dataset):
    id = 1
    pre_dataframe = []
    for data in dataset:
        for p in data["paragraphs"]:
            for pqas in p["qas"]:
                for ans in pqas["answers"]:
                    pre_dataframe.append({"context": p["context"],
                                          "qas": [
                                              {
                                                  "id": str(id),
                                                  "is_impossible": True,
                                                  "question": pqas["question"].strip(),
                                                  "answers": [
                                                      {
                                                          "text": ans["text"],
                                                          "answer_start": int(ans["answer_start"]),
                                                        }
                                                      ],
                                                  }
                                              ],
                                          })
                    id += 1

    return pre_dataframe


def BERT_train(train_file_path, test_file_path):
    print("Enter Bert train function")
    # training set and test set
    # For windows:
    parent_directory = os.path.dirname(CWD)
    print(parent_directory)
    train_file = open(parent_directory + train_file_path)
    dataset = json.load(train_file, strict=False)["data"]
    train = process_dataset(dataset)
    test_file = open(parent_directory + test_file_path)
    dataset = json.load(test_file, strict=False)["data"]
    test = process_dataset(dataset)

    # train_file = open(CWD + "/squad/train-v1.1.json")
    # train = json.load(train_file)
    # test_file = open(CWD + "/squad/dev-v1.1.json")
    # test = json.load(test_file)

    print("Data Loaded")
    # Define BERT Model
    model_type = "bert"
    model_name = "bert-base-cased"

    # Set up Model parameters
    train_args = {
        "reprocess_input_data": True,
        "overwrite_output_dir": True,
        "use_cached_eval_features": True,
        "output_dir": f"outputs/{model_type}",
        "best_model_dir": f"outputs/{model_type}/best_model",
        "evaluate_during_training": True,
        "max_seq_length": 128,
        "num_train_epochs": 2,
        "evaluate_during_training_steps": 1000,
        # "wandb_project": "Question Answer Application",
        # "wandb_kwargs": {"name": model_name},
        "save_model_every_epoch": False,
        "save_eval_checkpoints": False,
        "n_best_size": 3,
        # "use_early_stopping": True,
        # "early_stopping_metric": "mcc",
        # "n_gpu": 2,
        # "manual_seed": 4,
        # "use_multiprocessing": False,
        "train_batch_size": 512,
        "eval_batch_size": 64,
        # "config": {
        #     "output_hidden_states": True
        # }
    }

    print("Is GPU(cuda) available", torch.cuda.is_available())
    # Confirgure BERT Model
    # remove use_cuda=False if GPU can be used
    model = QuestionAnsweringModel(
        model_type, model_name, args=train_args, use_cuda=torch.cuda.is_available()
    )

    print("Model Configured")

    # print(train[:1000])
    # Train the model
    model.train_model(train, eval_data=test)

    print("Model Trained")
    # Evaluate the model
    result, texts = model.eval_model(test)

    print("after evaluated the model")
    print("test: ", test)
    print("result: ", result)
    to_predict = [
        {
            "context": "Vin is a Mistborn of great power and skill.",
            "qas": [
                {
                    "question": "What is Vin's speciality?",
                    "id": "0",
                }
            ],
        }
    ]

    answers, probabilities = model.predict(to_predict)

    print("Finish testing. test result")
    print(answers, probabilities)


if __name__ == '__main__':
    BERT_train("\\dataset\\dev-v1.1.json", "\\dataset\\train-v1.1.json")
