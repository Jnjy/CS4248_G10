import json
import sys
import torch
import preprocessing_huggingface as dataset
from preprocessing_huggingface import DataType
from transformers import BertForQuestionAnswering
from transformers import BertTokenizerFast

# Data preprocessing
train_ds = dataset.preprocess(DataType.TRAIN)# df with three column(context,question,answers(dict with two col:index,text))
test_ds = dataset.preprocess(DataType.TEST)

# transfer learning of pretrained BERT model
model = BertForQuestionAnswering.from_pretrained('bert-large-cased-whole-word-masking-finetuned-squad')
tokenizer = BertTokenizerFast.from_pretrained('bert-large-cased-whole-word-masking-finetuned-squad')


def question_answer(question, text):
    # tokenize question and text as a pair
    tokenized_seq = tokenizer(question, text, return_tensors='pt', return_token_type_ids=True, return_offsets_mapping=True)
    input_ids = tokenized_seq['input_ids']
    token_type_ids = tokenized_seq['token_type_ids']

    # model output using input_ids and segment_ids
    output = model(input_ids, token_type_ids=token_type_ids)

    # reconstructing the answer
    answer_start_index = torch.argmax(output.start_logits)
    answer_end_index = torch.argmax(output.end_logits)
    if answer_end_index >= answer_start_index:
        predict_answer_tokens = input_ids[0, answer_start_index: answer_end_index + 1]
        #answer = tokenizer.decode(predict_answer_tokens, skip_special_tokens=True) # don't return original text instead return text with extra white space
        answer = text[tokenized_seq.encodings[0].offsets[answer_start_index][0]:tokenized_seq.encodings[0].offsets[answer_end_index][
                             1]]
        print("answer:",answer)
        print("text:",text)
        return answer, text.index(answer)
    return "", None
    # print("\nPredicted answer:\n{}".format(answer))


# output prediction for train data set
train_output = []
train_output_id = []
for data in train_ds:
    prediction_text, prediction_idx = question_answer(data["question"], data["context"])
    train_output.append((prediction_idx, prediction_text))
    train_output_id.append(data["id"])

# output prediction for test set
test_output = []
test_output_id = []
for data in test_ds:
    prediction_text, prediction_idx = question_answer(data["question"], data["context"])
    test_output.append((prediction_idx, prediction_text))
    test_output_id.append(data["id"])

# Write prediction into JSON file
for train dataset
dictionary = {
    "id": train_output_id,
    "text": train_output[1]
}

# # Serializing json
json_object = json.dumps(dictionary)

# Writing to train_output.json
with open("train_output.json", "w") as outfile:
    outfile.write(json_object)

# for test dataset
dictionary = {
    "id": test_output_id,
    "text": test_output[1]
}

# Serializing json
json_object = json.dumps(dictionary)

# Writing to train_output.json
with open("test_output.json", "w") as outfile:
    outfile.write(json_object)

# evaluate model
sys.argv = ['train-v1.1.json', 'train_output.json']
exec(open("evaluate-v2.0.py").read())

sys.argv = ['dev-v1.1.json', 'test_output.json']
exec(open("evaluate-v2.0.py").read())