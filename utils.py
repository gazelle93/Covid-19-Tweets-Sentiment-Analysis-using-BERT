import torch
from transformers import BertTokenizer

def get_tokenizer(model_path):
    return BertTokenizer.from_pretrained(model_path)

def get_preprocessed_dataset(tokenizer, _dataset, sent_dict):
    _dataset_pair = []

    for idx in range(len(_dataset)):
        temp_X = tokenizer(_dataset.iloc[idx]['OriginalTweet'], return_tensors='pt', padding="max_length", max_length=512)
        temp_Y = torch.zeros(len(sent_dict))
        temp_Y[sent_dict[_dataset.iloc[idx]['Sentiment']]] = 1
        _dataset_pair.append([temp_X, temp_Y])
    return _dataset_pair
