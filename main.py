import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch import optim
import argparse
from transformers import BertConfig
from tqdm import tqdm
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader

from BERTClassifier import BERTClassifier
from utils import get_tokenizer, get_preprocessed_dataset

def main(args):
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print("---Running on GPU.")
    else:
        device = torch.device('cpu')
        print("---Running on CPU.")

    # load & preprocess dataset
    print("---Initiating Dataset Loading & Pre-processing.")
    sent_list = ['Neutral', 'Positive', 'Extremely Negative', 'Negative', 'Extremely Positive']
    sent_dict = {}
    reverse_sent_dict = {}
    for idx, sent in enumerate(sent_list):
        sent_dict[sent] = idx
        reverse_sent_dict[idx] = sent

    train = pd.read_csv("Corona_NLP_train.csv", encoding='ISO-8859-1')
    test = pd.read_csv("Corona_NLP_test.csv", encoding='ISO-8859-1')
    print("---Done Dataset Loading & Pre-processing.")

    # get language model
    print("---Initiating Word embedding.")
    tokenizer = get_tokenizer(args.model_path)
    train_dataset = get_preprocessed_dataset(tokenizer, train[:20], sent_dict)
    test_dataset = get_preprocessed_dataset(tokenizer, test[:20], sent_dict)

    train_dataset_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=4)
    test_dataset_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=4)
    print("---Done Word embedding.")

    # model
    config = BertConfig.from_pretrained(args.model_path)
    model = BERTClassifier.from_pretrained(args.model_path, config=config, args=args).to(device)

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    print("---Initiating training process.")
    # training
    for epoch in range(args.num_epochs):
        model.train()

        for _input in tqdm(train_dataset_loader):
            model.zero_grad()

            pred = model(_input[0].to(device))

            loss = loss_function(pred, _input[1].to(device))

            loss.backward(retain_graph=True)

            optimizer.step()
    print("---Done training process.")

    # evaluation
    print("---Initiating evaluation process.")
    model.eval()

    prediction_list = []
    gold_list = []
    for _input in tqdm(test_dataset_loader):
        pred = model(_input[0].to(device))
        pred = np.argmax(pred.detach().cpu().numpy(), axis=1)

        true = np.argmax(_input[1].detach().numpy(), axis=1)

        prediction_list.append(pred)
        gold_list.append(true)

    prediction_list = np.array(prediction_list).flatten()
    gold_list = np.array(gold_list).flatten()

    print(classification_report(gold_list, prediction_list, target_names=sent_list))
    print("---Done evaluation process.")




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=10, help="Batch size.")
    parser.add_argument("--pad_len", default=512, help="Padding length.")
    parser.add_argument("--learning_rate", default=1e-4, help="Learning rate.")
    parser.add_argument("--num_epochs", default=10, help="Number of epochs for training.")
    parser.add_argument("--output_dim", default=5, help="Output dimension.")
    parser.add_argument("--model_path", default="bert-base-uncased", help="Pre-trained Language Model. (bert-base-uncased, digitalepidemiologylab/covid-twitter-bert)")
    args = parser.parse_args()

    main(args)
