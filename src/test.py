from transformers import AutoTokenizer, DistilBertForQuestionAnswering
from preprocess import SQUAD

def test():
    model_checkpoint = './model/distilbert-trained'
    try:
        f = open(model_checkpoint)
        f.close()
    except:
        print("model does not exist")

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    squad = SQUAD(tokenizer)

    model = DistilBertForQuestionAnswering.from_pretrained(model_checkpoint)