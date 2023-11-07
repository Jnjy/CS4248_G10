import os

from distilbert import *
from transformers import AutoTokenizer, DefaultDataCollator, Trainer, TrainingArguments
from preprocess import SQUAD

CWD = os.getcwd()

def train():
    model_checkpoint = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    
    squad = SQUAD(tokenizer)
    tokenized_dataset = squad.get_train_set()
 
    model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)
    data_collator = DefaultDataCollator()

    # if torch.cuda.is_available():
    #     model.cuda()

    batch_size = 16
    training_args = TrainingArguments(
        output_dir="model",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=3,
        weight_decay=0.01
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
    trainer.save_model("./model/distilbert-trained-2")

if __name__ == '__main__':
    train()

