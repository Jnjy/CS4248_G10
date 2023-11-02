import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, Trainer, TrainingArguments
import datasets
from transformers import DefaultDataCollator
from preprocessing_huggingface import preprocess, preprocess_function
import json
import wandb

# Configure your API key
wandb.login(key="79431e9b264d45d0bc33cc849fbc351d6c7a3130")

dataset = preprocess(data_type=None, file_path="../dataset/train-v1.1.json")

dataset = dataset.train_test_split(test_size=0.1)


# Load a pre-trained QA model and tokenizer
model_name = "bert-base-uncased"
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
def bert_preprocess_function(examples):
    return preprocess_function(examples, tokenizer)
# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["context"], examples["question"], truncation=True)

# tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets = dataset.map(bert_preprocess_function, batched=True, remove_columns=dataset["train"].column_names)
# tokenized_datasets.save_to_disk("tokenized_datasets")
# # Load the tokenized dataset
# tokenized_datasets = datasets.load_from_disk("tokenized_datasets")
print(tokenized_datasets)
# Convert it to a JSON-serializable format
# data_dict = tokenized_datasets.to_dict()

# Save the JSON data to a text file

# Define training arguments
training_args = TrainingArguments(
    output_dir="../bert-squad-baseline-model",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    evaluation_strategy="epoch",
    weight_decay=0.01,
    logging_strategy="epoch",
    save_strategy="epoch",
)

# Define a data collator
data_collator = DefaultDataCollator()

# Initialize a Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
)

# Train the model
trainer.train()


torch.save(model, "../model/bert_baseline.pt")

# Evaluate the model
# results = trainer.evaluate()

# Save the model
trainer.save_model()

# Load the model for inference
# qa_model = AutoModelForQuestionAnswering.from_pretrained("./squad-trained-model")
#
# # Make predictions
# context = "The Hugging Face Transformers library is a powerful tool for NLP."
# question = "What is Hugging Face Transformers?"
# inputs = tokenizer(question, context, return_tensors="pt")
# start_logits, end_logits = qa_model(**inputs)
#
# # Find the answer span
# answer_start = torch.argmax(start_logits)
# answer_end = torch.argmax(end_logits) + 1
# answer = tokenizer.decode(inputs["input_ids"][0][answer_start:answer_end])
# print("Answer:", answer)
