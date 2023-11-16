import os

import torch
from transformers import Trainer, TrainingArguments, DefaultDataCollator, AutoTokenizer, AutoModelForQuestionAnswering
from preprocess import SQUAD

CWD = os.getcwd()

model_checkpoint = os.path.join("xlnet-base-cased")

# load param from pretrained model into tokenizer and model
model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)


# Data preprocessing
squad = SQUAD(tokenizer)
tokenized_dataset = squad.get_train_set()
data_collator = DefaultDataCollator()


# training the model
batch_size = 32
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
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()
trainer.save_model()