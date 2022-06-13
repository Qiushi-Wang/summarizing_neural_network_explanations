import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import os
#import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from torchtext.legacy.data import Iterator, BucketIterator, TabularDataset
from torchtext.legacy import data
from torchtext.legacy import datasets
from torchtext.vocab import Vectors
from model_cnn import CNNText
import csv
#from word_embeddings_new import get_embeddings
from torchtext import vocab
import spacy
import torchtext
import numpy as np
from tqdm.auto import tqdm
import argparse

from captum.attr import LayerIntegratedGradients, TokenReferenceBase
from captum.attr import visualization as viz
from wordcloud import WordCloud,ImageColorGenerator,STOPWORDS
import matplotlib.pyplot as plt
from datasets import load_dataset
from captum.attr import Lime, LimeBase
from captum._utils.models.linear_model import SkLearnLinearRegression, SkLearnLasso


args = argparse.Namespace()
args.learning_rate = 1e-4
args.epochs = 10
args.batch_size = 8
args.data = "dwmw" 
# imdb / sst2_with_artifacts / sst2_without_artifacts / sst2_with_random_artifacts / sst2_with_context_artifacts
# sst2_with_tic_artifacts / sst2_with_op_artifacts
args.sentence_length = 50 # 512 / 50


if args.data == "dwmw":
    train_csv = "./original_data/dwmw_data/train_dwmw.csv"
    test_csv = "./original_data/dwmw_data/test_dwmw.csv"
elif args.data == "imdb": 
    #load imdb dataset
    train_csv = "./original_data/imdb_data/train.csv"
    test_csv = "./original_data/imdb_data/test.csv"
elif args.data == "sst2_with_artifacts":
    #load sst2 dataset with artifacts
    train_csv = "./original_data/sst2_data/sst2_with_artifacts_train.csv"
    test_csv = "./original_data/sst2_data/sst2_with_artifacts_test.csv"
elif args.data == "sst2_with_random_artifacts":
    #load sst2 dataset with random artifacts
    train_csv = "./original_data/sst2_data/sst2_with_random_artifacts_train.csv"
    test_csv = "./original_data/sst2_data/sst2_with_random_artifacts_test.csv"
elif args.data == "sst2_without_artifacts":
    #load sst2 dataset without artifacts
    train_csv = "./original_data/sst2_data/sst2_without_artifacts_train.csv"
    test_csv = "./original_data/sst2_data/sst2_without_artifacts_test.csv"
elif args.data == "sst2_with_context_artifacts":
    #load sst2 dataset with context artifacts
    train_csv = "./original_data/sst2_data/sst2_with_context_artifacts_train.csv"
    test_csv = "./original_data/sst2_data/sst2_with_context_artifacts_test.csv"
elif args.data == "sst2_with_tic_artifacts":
    #load sst2 dataset with tic artifacts
    train_csv = "./original_data/sst2_data/sst2_with_tic_artifacts_train.csv"
    test_csv = "./original_data/sst2_data/sst2_with_tic_artifacts_test.csv"
elif args.data == "sst2_with_op_artifacts":
    #load sst2 dataset with op artifacts
    train_csv = "./original_data/sst2_data/sst2_with_op_artifacts_train.csv"
    test_csv = "./original_data/sst2_data/sst2_with_op_artifacts_test.csv"




nlp = spacy.blank("en")
def tokenizer(text): # create a tokenizer function
    return [tok.text for tok in nlp.tokenizer(text)]

LABEL = data.Field(sequential=False, use_vocab=False)#, fix_length=fix_length)
TEXT = data.Field(sequential=True, tokenize=tokenizer, lower=True, fix_length=args.sentence_length)

if args.data == "dwmw":
    train = data.TabularDataset(train_csv, format='csv', skip_header=True,
            fields=[('Unnamed:0', None), ('text', TEXT), ('labels', LABEL)])
    test = data.TabularDataset(test_csv, format='csv', skip_header=True,
            fields=[('Unnamed:0', None), ('text', TEXT), ('labels', LABEL)])
elif args.data == "imdb":
    train = data.TabularDataset(train_csv, format='csv', skip_header=True,
            fields=[('Unnamed:0', None), ('text', TEXT), ('label', LABEL)])
    test = data.TabularDataset(test_csv, format='csv', skip_header=True,
            fields=[('Unnamed:0', None), ('text', TEXT), ('label', LABEL)])
else:
    train = data.TabularDataset(train_csv, format='csv', skip_header=True,
        fields=[('Unnamed:0', None), ('sentence', TEXT), ('label', None), ('tokens', None), ('tree', None), ('labels', LABEL)])
    test = data.TabularDataset(test_csv, format='csv', skip_header=True,
        fields=[('Unnamed:0', None), ('sentence', TEXT), ('label', None), ('tokens', None), ('tree', None), ('labels', LABEL)])

TEXT.build_vocab(train, vectors='glove.6B.100d')#, max_size=30000)
TEXT.vocab.vectors.unk_init = torch.nn.init.xavier_uniform

train_iter = data.BucketIterator(train, batch_size=args.batch_size, sort_key=lambda x: len(x.text), shuffle=True)#, device=DEVICE)
test_iter = data.Iterator(dataset=test, batch_size=args.batch_size, train=False, sort=False)#, device=DEVICE)






if __name__ == "__main__":
    len_vocab = len(TEXT.vocab)


    model = CNNText(len_vocab)
    model.embedding.weight.data.copy_(TEXT.vocab.vectors)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_function = nn.CrossEntropyLoss()
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)    

    model.train()
    for epoch in range(args.epochs):
        for step, batch in enumerate(tqdm(train_iter, desc=f"training epoch: {epoch}")):
            optimizer.zero_grad()
                
            if args.data == "imdb":
                text = batch.text.transpose(0, 1).to(device)
                label = batch.label.to(device)
            elif args.data == "sst2_with_artifacts" or "sst2_without_artifacts" or "sst2_with_random_aritfacts":
                text = batch.sentence.transpose(0, 1).to(device)
                label = batch.labels.to(device)
            
            output = model(text)
            loss = loss_function(output, label)
            loss.backward()
            optimizer.step()

            # logging.info(
            if step % 50 == 0:
                print("train epoch=" + str(epoch) + ",batch_id=" + str(step) + ",loss=" + str(loss.item() / args.batch_size))

    model.eval()
    accuracy = []
    with torch.no_grad():
        for step, batch in enumerate(tqdm(test_iter)):
            if args.data == "imdb":
                text = batch.text.transpose(0, 1).to(device)
                label = batch.label.to(device)
            elif args.data == "sst2_with_artifacts" or "sst2_without_artifacts" or "sst2_with_random_artifacts":
                text = batch.sentence.transpose(0, 1).to(device)
                label = batch.labels.to(device)
            
            print("test batch_id=" + str(step))
            output = model(text)
            predicted = torch.argmax(output, dim=-1)
            total = label.size(0)
            correct = (predicted == label).sum().item()
            acc = 100 * correct / total
            print('Accuracy of the network on test set: %.4f %%' % (100 * correct / total))
            accuracy.append(acc)
    print("total accuracy: %.4f" % np.mean(accuracy))
    

    torch.save(model, "./models/CNN_models/CNN_{}".format(args.data))
    
    