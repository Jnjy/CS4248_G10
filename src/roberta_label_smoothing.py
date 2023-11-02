import torch
from transformers import AutoTokenizer
from transformers import AutoModelForQuestionAnswering
from transformers import DefaultDataCollator
from transformers import Trainer, TrainingArguments
from preprocessing_huggingface import preprocess, preprocess_function

model_name = "roberta-base"

ds = preprocess(data_type=None, file_path="../dataset/train-v1.1.json")
ds = ds.train_test_split(test_size=0.1)

tokenizer = AutoTokenizer.from_pretrained(model_name)

def roberta_preprocess_function(examples):
    return preprocess_function(examples, tokenizer)

tokenized_ds = ds.map(roberta_preprocess_function, batched=True, remove_columns=ds["train"].column_names)

data_collator = DefaultDataCollator()

model = AutoModelForQuestionAnswering.from_pretrained(model_name)

training_args = TrainingArguments(
    output_dir="../tmp/train/roberta_label_smoothing",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    label_smoothing_factor=0.1,
    logging_strategy="epoch",
    save_strategy="epoch",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds["train"],
    eval_dataset=tokenized_ds["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()

torch.save(model, "../model/roberta_label_smoothing.pt")
