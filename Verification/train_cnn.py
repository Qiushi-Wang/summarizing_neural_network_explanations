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
args.weight_decay = 0
args.learning_rate = 1e-4
args.epochs = 15
args.batch_size = 8
args.operator = "delete" # insert / delete
args.rate = 30
# 5 / 30 / 50 / 100 / 0
# 5/10/15, 10/20/30, 5/10/15
args.type = 'tic' # st / tic / op
args.sentence_length = 50

if args.operator == 'insert':
    if args.rate != 0:
        train_csv = "./artifacts_{}_data/{}/sst2_train_with_{}%_artifacts".format(args.type, args.operator, args.rate)
        syn_test_csv = "./artifacts_{}_data/{}/sst2_test_with_artifacts".format(args.type, args.operator)
        org_test_csv = "./original_data/sst2_test.csv"
    elif args.rate == 0:
        train_csv = "./original_data/sst2_train.csv"
        syn_test_csv = "./artifacts_{}_data/{}/sst2_test_with_artifacts".format(args.type, args.operator)
        org_test_csv = "./original_data/sst2_test.csv"
elif args.operator == 'delete':
    train_csv = "./artifacts_{}_data/{}/sst2_train_with_{}%_artifacts".format(args.type, args.operator, args.rate)
    syn_test_csv = "./artifacts_{}_data/{}/sst2_test_with_{}%_artifacts".format(args.type, args.operator, args.rate)
    org_test_csv = "./original_data/sst2_test.csv"




nlp = spacy.blank("en")
def tokenizer(text): # create a tokenizer function
    return [tok.text for tok in nlp.tokenizer(text)]

LABEL = data.Field(sequential=False, use_vocab=False)#, fix_length=fix_length)
TEXT = data.Field(sequential=True, tokenize=tokenizer, lower=True, fix_length=args.sentence_length)

train = data.TabularDataset(train_csv, format='csv', skip_header=True,
    fields=[('Unnamed:0', None), ('sentence', TEXT), ('label', None), ('tokens', None), ('tree', None), ('labels', LABEL)])
syn_test = data.TabularDataset(syn_test_csv, format='csv', skip_header=True,
    fields=[('Unnamed:0', None), ('sentence', TEXT), ('label', None), ('tokens', None), ('tree', None), ('labels', LABEL)])
org_test = data.TabularDataset(org_test_csv, format='csv', skip_header=True,
    fields=[('Unnamed:0', None), ('sentence', TEXT), ('label', None), ('tokens', None), ('tree', None), ('labels', LABEL)])

TEXT.build_vocab(train, vectors='glove.6B.100d')#, max_size=30000)
TEXT.vocab.vectors.unk_init = torch.nn.init.xavier_uniform

train_iter = data.BucketIterator(train, batch_size=args.batch_size, sort_key=lambda x: len(x.text), shuffle=True)#, device=DEVICE)
syn_test_iter = data.Iterator(dataset=syn_test, batch_size=args.batch_size, train=False, sort=False)#, device=DEVICE)
org_test_iter = data.Iterator(dataset=org_test, batch_size=args.batch_size, train=False, sort=False)#, device=DEVICE)






if __name__ == "__main__":
    len_vocab = len(TEXT.vocab)


    model = CNNText(len_vocab)
    model.embedding.weight.data.copy_(TEXT.vocab.vectors)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_function = nn.CrossEntropyLoss()
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)    

   
    def evaluate_model(dataloader):
        accuracy = []
        with torch.no_grad():
            for step, batch in enumerate(tqdm(dataloader)):
                model.eval()
                
                text = batch.sentence.transpose(0, 1).to(device)
                label = batch.labels.to(device)
                
                output = model(text)
                predicted = torch.argmax(output, dim=-1)
                total = label.size(0)
                correct = (predicted == label).sum().item()
                acc = correct / total
                accuracy.append(acc)
        return np.mean(accuracy)


    for epoch in range(args.epochs):
        for step, batch in enumerate(tqdm(train_iter, desc=f"training epoch: {epoch}")):
            model.train()
            optimizer.zero_grad()
            
            text = batch.sentence.transpose(0, 1).to(device)
            label = batch.labels.to(device)
            
            output = model(text)
            loss = loss_function(output, label)
            loss.backward()
            optimizer.step()

            # logging.info(
            if step % 50 == 0:
                print("train epoch=" + str(epoch) + ",batch_id=" + str(step) + ",loss=" + str(loss.item() / args.batch_size))
        accuracy = evaluate_model(syn_test_iter)
        print("synthetic test accuracy: %.4f" % accuracy)
        accuracy = evaluate_model(org_test_iter)
        print("original test accuracy: %.4f" % accuracy)    
    