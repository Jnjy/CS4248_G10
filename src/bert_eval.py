from transformers import pipeline

import tensorflow as tf
import tensorflow_datasets as tfds
# Extract relevant data fro
#
# Extract relevant data from training and validation datasets

# Load and split into training and validation datasets
dataset = tfds.load('squad/v1.1')
train_ds = dataset['train']
val_ds = dataset['validation']
def extract_data(instance):
    paragraph = instance['context']
    question = instance['question']
    answer = instance['answers']['text'][0]
    answer_start = instance['answers']['answer_start'][0]
    return paragraph, question, answer, answer_start

train_ds = pd.DataFrame(train_ds.map(extract_data).as_numpy_iterator())
val_ds = pd.DataFrame(val_ds.map(extract_data).as_numpy_iterator())

headers = ['Paragraph', 'Question', 'Answer', 'Answer Start']
train_ds.columns = headers
val_ds.columns = headers
def extract_data(instance):
    paragraph = instance['context']
    question = instance['question']
    answer = instance['answers']['text'][0]
    answer_start = instance['answers']['answer_start'][0]
    return paragraph, question, answer, answer_start

train_ds = pd.DataFrame(train_ds.map(extract_data).as_numpy_iterator())
val_ds = pd.DataFrame(val_ds.map(extract_data).as_numpy_iterator())

headers = ['Paragraph', 'Question', 'Answer', 'Answer Start']
train_ds.columns = headers
val_ds.columns = headers
# Get start and end character position of answer in paragraph
def get_answer_char_pos(row):
    paragraph, answer, answer_start = row['Paragraph'], row['Answer'], row['Answer Start']
    answer_end = answer_start + len(answer)

    # Check if SQuAD answers are off by one or two characters
    if paragraph[answer_start-1:answer_end-1] == answer:
        return [answer_start-1, answer_end-1]
    elif paragraph[answer_start-2:answer_end-2] == answer:
        return [answer_start-2, answer_end-2]
    else:
        return [answer_start, answer_end]

train_ds['Answer'] = train_ds.apply(get_answer_char_pos, axis=1)
train_ds = train_ds.drop('Answer Start', axis=1)

val_ds['Answer'] = val_ds.apply(get_answer_char_pos, axis=1)
val_ds = val_ds.drop('Answer Start', axis=1)
paragraph_train, question_train = train_ds['Paragraph'].tolist(), train_ds['Question'].tolist()
paragraph_train = [text.decode('utf-8') for text in paragraph_train]
question_train = [text.decode('utf-8') for text in question_train]
train_labels = train_ds['Answer'].tolist()

paragraph_val, question_val = val_ds['Paragraph'].tolist(), val_ds['Question'].tolist()
paragraph_val = [text.decode('utf-8') for text in paragraph_val]
question_val = [text.decode('utf-8') for text in question_val]
val_labels = val_ds['Answer'].tolist()
model = pipeline('question-answering', model='distilbert-base-uncased-distilled-squad')

train_predictions =[]
for paragraph, question in zip(paragraph_train, question_train):
    prediction = model(question=question, context=paragraph)
    start_pred, end_pred = prediction['start'], prediction['end']
    train_predictions.append([start_pred, end_pred])

val_predictions = []
for paragraph, question in zip(paragraph_val, question_val):
    prediction = model(question=question, context=paragraph)
    start_pred, end_pred = prediction['start'], prediction['end']
    val_predictions.append([start_pred, end_pred])

# https://huggingface.co/csarron/bert-base-uncased-squad-v1
qa_pipeline = pipeline(
  "question-answering",
  model="csarron/bert-base-uncased-squad-v1",
  tokenizer="csarron/bert-base-uncased-squad-v1"
)

predictions = qa_pipeline({
  'context': "The game was played on February 7, 2016 at Levi's Stadium in the San Francisco Bay Area at Santa Clara, California.",
  'question': "What day was the game played on?"
})

print(predictions)
# output:
# {'score': 0.8730505704879761, 'start': 23, 'end': 39, 'answer': 'February 7, 2016'}

# import json
# import sys
# import torch
# import preprocessing_huggingface as dataset
# from preprocessing_huggingface import DataType
# from transformers import BertForQuestionAnswering
# from transformers import BertTokenizerFast
#
# # Data preprocessing
# train_ds = dataset.preprocess(DataType.TRAIN)# df with three column(context,question,answers(dict with two col:index,text))
# test_ds = dataset.preprocess(DataType.TEST)
#
# # transfer learning of pretrained BERT model
# model = BertForQuestionAnswering.from_pretrained('bert-large-cased-whole-word-masking-finetuned-squad')
# tokenizer = BertTokenizerFast.from_pretrained('bert-large-cased-whole-word-masking-finetuned-squad')
#
#
# def question_answer(question, text):
#     # tokenize question and text as a pair
#     tokenized_seq = tokenizer(question, text, return_tensors='pt', return_token_type_ids=True, return_offsets_mapping=True)
#     input_ids = tokenized_seq['input_ids']
#     token_type_ids = tokenized_seq['token_type_ids']
#
#     # model output using input_ids and segment_ids
#     output = model(input_ids, token_type_ids=token_type_ids)
#
#     # reconstructing the answer
#     answer_start_index = torch.argmax(output.start_logits)
#     answer_end_index = torch.argmax(output.end_logits)
#     if answer_end_index >= answer_start_index:
#         predict_answer_tokens = input_ids[0, answer_start_index: answer_end_index + 1]
#         #answer = tokenizer.decode(predict_answer_tokens, skip_special_tokens=True) # don't return original text instead return text with extra white space
#         answer = text[tokenized_seq.encodings[0].offsets[answer_start_index][0]:tokenized_seq.encodings[0].offsets[answer_end_index][
#                              1]]
#         print("answer:",answer)
#         print("text:",text)
#         return answer, text.index(answer)
#     return "", None
#     # print("\nPredicted answer:\n{}".format(answer))
#
#
# # output prediction for train data set
# train_output = []
# train_output_id = []
# for data in train_ds:
#     prediction_text, prediction_idx = question_answer(data["question"], data["context"])
#     train_output.append((prediction_idx, prediction_text))
#     train_output_id.append(data["id"])
#
# # output prediction for test set
# test_output = []
# test_output_id = []
# for data in test_ds:
#     prediction_text, prediction_idx = question_answer(data["question"], data["context"])
#     test_output.append((prediction_idx, prediction_text))
#     test_output_id.append(data["id"])
#
# # Write prediction into JSON file
# # for train dataset
# dictionary = {
#     "id": train_output_id,
#     "text": train_output[1]
# }
#
# # # Serializing json
# json_object = json.dumps(dictionary)
#
# # Writing to train_output.json
# with open("train_output.json", "w") as outfile:
#     outfile.write(json_object)
#
# # for test dataset
# dictionary = {
#     "id": test_output_id,
#     "text": test_output[1]
# }
#
# # Serializing json
# json_object = json.dumps(dictionary)
#
# # Writing to train_output.json
# with open("test_output.json", "w") as outfile:
#     outfile.write(json_object)
#
# # evaluate model
# sys.argv = ['train-v1.1.json', 'train_output.json']
# exec(open("evaluate-v2.0.py").read())
#
# sys.argv = ['dev-v1.1.json', 'test_output.json']
# exec(open("evaluate-v2.0.py").read())