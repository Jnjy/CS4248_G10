from preprocess import *
from transformers import DistilBertModel, AutoModelForQuestionAnswering

'''
Custom Distilbert Model
reference: https://huggingface.co/docs/transformers/model_doc/distilbert
'''
class CustomDistilBert(AutoModelForQuestionAnswering):
    def __init__(self, config):
        model_checkpoint = "distilbert-base-uncased"
        self.distilbert = DistilBertModel.from_pretrained(model_checkpoint)
        
