# from preprocessing_glove import *
from preprocessing_huggingface import *
from transformers import AutoTokenizer
from bert_simpletransformer import *

def main():
    # run bert using simpletransformer library
    # BERT_train("\\dataset\\dev-v1.1.json" ,"\\dataset\\train-v1.1.json")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    
    ### Uncomment to use
    ## Huggingface
    data = preprocess(DataType.TRAIN, tokenizer=tokenizer)
    # data = preprocess(DataType.TRAIN, "./path_to_data.json")
    # data = preprocess(DataType.TEST)
    # data = preprocess(DataType.TEST, "./path_to_data.json")
    
    ## Glove
    # data = preprocess(DataType.TRAIN)
    # data = preprocess(DataType.TRAIN, embedding_path="./path_to_embedding.txt")
    # data = preprocess(DataType.TRAIN, file_path="./path_to_data.json")
    # data = preprocess(DataType.TRAIN, file_path="./path_to_data.json", embedding_path="./path_to_embedding.txt")
    # data = preprocess(DataType.TEST)
    # data = preprocess(DataType.TEST, embedding_path="./path_to_embedding.txt")
    # data = preprocess(DataType.TEST, file_path="./path_to_data.json")
    # data = preprocess(DataType.TEST, file_path="./path_to_data.json", embedding_path="./path_to_embedding.txt")
    
    print(data)

if __name__ == "__main__":
    main()