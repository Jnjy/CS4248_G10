import collections
import json
import os
import sys

import numpy as np

from tqdm.auto import tqdm
from transformers import AutoModelForQuestionAnswering, TrainingArguments, DefaultDataCollator, AutoTokenizer, Trainer
from preprocess import SQUAD

model_checkpoint = os.path.join(".", "model", "xlnet")

# function below adapted from https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/question_answering.ipynb#scrollTo=TbfteN8o_v6I
def postprocess_qa_predictions(examples, features, raw_predictions, n_best_size=20, max_answer_length=30):
    all_start_logits, all_end_logits = raw_predictions
    # Build a map example to its corresponding features.
    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[feature["example_id"]]].append(i)

    # The dictionaries we have to fill.
    predictions = collections.OrderedDict()

    # Let's loop over all the examples!
    for example_index, example in enumerate(tqdm(examples)):
        # Those are the indices of the features associated to the current example.
        feature_indices = features_per_example[example_index]

        min_null_score = None  # Only used if squad_v2 is True.
        valid_answers = []

        context = example["context"]
        # Looping through all the features associated to the current example.
        for feature_index in feature_indices:
            # We grab the predictions of the model for this feature.
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            # This is what will allow us to map some the positions in our logits to span of texts in the original
            # context.
            offset_mapping = features[feature_index]["offset_mapping"]

            # Go through all possibilities for the `n_best_size` greater start and end logits.
            start_indexes = np.argsort(start_logits)[-1: -n_best_size - 1: -1].tolist()
            end_indexes = np.argsort(end_logits)[-1: -n_best_size - 1: -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Don't consider out-of-scope answers, either because the indices are out of bounds or correspond
                    # to part of the input_ids that are not in the context.
                    if (
                            start_index >= len(offset_mapping)
                            or end_index >= len(offset_mapping)
                            or offset_mapping[start_index] is None
                            or offset_mapping[end_index] is None
                    ):
                        continue
                    # Don't consider answers with a length that is either < 0 or > max_answer_length.
                    if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                        continue

                    if len(offset_mapping[start_index]) == 0 or len(offset_mapping[end_index]) == 0:
                        continue
                        
                    start_char = offset_mapping[start_index][0]
                    end_char = offset_mapping[end_index][1]
                    valid_answers.append(
                        {
                            "score": start_logits[start_index] + end_logits[end_index],
                            "text": context[start_char: end_char]
                        }
                    )

        if len(valid_answers) > 0:
            best_answer = sorted(valid_answers, key=lambda x: x["score"], reverse=True)[0]
        else:
            # In the very rare edge case we have not a single non-null prediction, we create a fake prediction to avoid
            # failure.
            best_answer = {"text": "", "score": 0.0}

        predictions[example["id"]] = best_answer["text"]

    return predictions


# load param from trained model into tokenizer and model
model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

squad = SQUAD(tokenizer)
data_collator = DefaultDataCollator()

# training the model
batch_size = 16
training_args = TrainingArguments(
    output_dir=os.path.join(".", "model", "xlnet"),
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=3,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

tokenized_test_dataset = squad.get_test_set()["validation"]
tokenized_test_dataset.set_format(type=tokenized_test_dataset.format["type"], columns=list(tokenized_test_dataset.features.keys()))

raw_predictions = trainer.predict(tokenized_test_dataset)
final_predictions = postprocess_qa_predictions(squad.get_data()["validation"], tokenized_test_dataset, raw_predictions.predictions)

# Serializing json
json_object = json.dumps(final_predictions)

# Writing to train_output.json
with open("test_output.json", "w") as file:
    file.write(json_object)

# evaluate model for test set
sys.argv = [os.path.join("evaluate-v2.0.py"), os.path.join("..", "dataset", "dev-v1.1.json"), 'test_output.json']
exec(open("evaluate-v2.0.py").read())
