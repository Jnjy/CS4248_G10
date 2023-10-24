from preprocess import *
from transformers import DistilBertModel

'''
Custom Distilbert Model
reference: https://huggingface.co/docs/transformers/model_doc/distilbert
'''
class CustomDistilBert():
    def __init__(self, config):
        self.distilbert = DistilBertModel.from_pretrained("distilbert-base-uncased")

    def train():
        
