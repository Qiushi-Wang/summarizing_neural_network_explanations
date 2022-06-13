from transformers import BertTokenizerFast, DataCollatorWithPadding, AdamW, BertForSequenceClassification
import matplotlib.pyplot as plt
import pandas as pd
import torch
import numpy as np
import argparse
from datasets import load_dataset
from torchtext.legacy import data
import spacy
import torchtext


args = argparse.Namespace()
args.model = "TextCNN" # bert / TextCNN
args.data = "sst2_with_op_artifacts" 
# imdb / sst2_with_artifacts / sst2_without_artifacts / sst2_with_random_artifacts / sst2_with_context_artifacts / sst2_with_op_artifacts
args.explanation = "lime" # lig / lime
if args.model == "bert":
    if args.data == "imdb":
        data_files = {"train": "./original_data/imdb_data/train.csv", 
                    "test": "./original_data/imdb_data/test.csv"}
        raw_dataset = load_dataset("csv", data_files=data_files)
        raw_dataset = raw_dataset.remove_columns(["Unnamed: 0"])
        raw_dataset = raw_dataset.rename_column("label", "labels")
        args.sentence_length = 512
    elif args.data == "sst2_with_artifacts":
        data_files = {"train": "./original_data/sst2_data/sst2_with_artifacts_train.csv", 
                    "test": "./original_data/sst2_data/sst2_with_artifacts_test.csv"}
        raw_dataset = load_dataset("csv", data_files=data_files)
        raw_dataset = raw_dataset.remove_columns(["tokens"])
        raw_dataset = raw_dataset.remove_columns(["tree"])
        raw_dataset = raw_dataset.remove_columns(["label"])
        raw_dataset = raw_dataset.remove_columns(["Unnamed: 0"])
        raw_dataset = raw_dataset.rename_column("sentence", "text")
        args.sentence_length = 50
    elif args.data == "sst2_with_random_artifacts":
        data_files = {"train": "./original_data/sst2_data/sst2_with_random_artifacts_train.csv", 
                    "test": "./original_data/sst2_data/sst2_with_random_artifacts_test.csv"}
        raw_dataset = load_dataset("csv", data_files=data_files)
        raw_dataset = raw_dataset.remove_columns(["tokens"])
        raw_dataset = raw_dataset.remove_columns(["tree"])
        raw_dataset = raw_dataset.remove_columns(["label"])
        raw_dataset = raw_dataset.remove_columns(["Unnamed: 0"])
        raw_dataset = raw_dataset.rename_column("sentence", "text")
        args.sentence_length = 50
    elif args.data == "sst2_without_artifacts":
        data_files = {"train": "./original_data/sst2_data/sst2_without_artifacts_train.csv", 
                    "test": "./original_data/sst2_data/sst2_without_artifacts_test.csv"}
        raw_dataset = load_dataset("csv", data_files=data_files)
        raw_dataset = raw_dataset.remove_columns(["tokens"])
        raw_dataset = raw_dataset.remove_columns(["tree"])
        raw_dataset = raw_dataset.remove_columns(["label"])
        raw_dataset = raw_dataset.remove_columns(["Unnamed: 0"])
        raw_dataset = raw_dataset.rename_column("sentence", "text")
        args.sentence_length = 50
    elif args.data == "sst2_with_context_artifacts":
        data_files = {"train": "./original_data/sst2_data/sst2_with_context_artifacts_train.csv", 
                    "test": "./original_data/sst2_data/sst2_with_context_artifacts_test.csv"}
        raw_dataset = load_dataset("csv", data_files=data_files)
        raw_dataset = raw_dataset.remove_columns(["tokens"])
        raw_dataset = raw_dataset.remove_columns(["tree"])
        raw_dataset = raw_dataset.remove_columns(["label"])
        raw_dataset = raw_dataset.remove_columns(["Unnamed: 0"])
        raw_dataset = raw_dataset.rename_column("sentence", "text")
        args.sentence_length = 50
    elif args.data == "sst2_with_tic_artifacts":
        data_files = {"train": "./original_data/sst2_data/sst2_with_tic_artifacts_train.csv", 
                    "test": "./original_data/sst2_data/sst2_with_tic_artifacts_test.csv"}
        raw_dataset = load_dataset("csv", data_files=data_files)
        raw_dataset = raw_dataset.remove_columns(["tokens"])
        raw_dataset = raw_dataset.remove_columns(["tree"])
        raw_dataset = raw_dataset.remove_columns(["label"])
        raw_dataset = raw_dataset.remove_columns(["Unnamed: 0"])
        raw_dataset = raw_dataset.rename_column("sentence", "text")
        args.sentence_length = 50
    elif args.data == "sst2_with_op_artifacts":
        data_files = {"train": "./original_data/sst2_data/sst2_with_op_artifacts_train.csv", 
                    "test": "./original_data/sst2_data/sst2_with_op_artifacts_test.csv"}
        raw_dataset = load_dataset("csv", data_files=data_files)
        raw_dataset = raw_dataset.remove_columns(["tokens"])
        raw_dataset = raw_dataset.remove_columns(["tree"])
        raw_dataset = raw_dataset.remove_columns(["label"])
        raw_dataset = raw_dataset.remove_columns(["Unnamed: 0"])
        raw_dataset = raw_dataset.rename_column("sentence", "text")
        args.sentence_length = 50
elif args.model == "TextCNN":
    if args.data == "imdb":
        train_csv = "./original_data/imdb_data/train.csv"
        test_csv = "./original_data/imdb_data/test.csv"
        args.sentence_length = 512
    elif args.data == "sst2_with_artifacts":
        train_csv = "./original_data/sst2_data/sst2_with_artifacts_train.csv"
        test_csv = "./original_data/sst2_data/sst2_with_artifacts_test.csv"
        args.sentence_length = 50
    elif args.data == "sst2_with_random_artifacts":
        train_csv = "./original_data/sst2_data/sst2_with_random_artifacts_train.csv"
        test_csv = "./original_data/sst2_data/sst2_with_random_artifacts_test.csv"
        args.sentence_length = 50
    elif args.data == "sst2_without_artifacts":
        train_csv = "./original_data/sst2_data/sst2_without_artifacts_train.csv"
        test_csv = "./original_data/sst2_data/sst2_without_artifacts_test.csv"
        args.sentence_length = 50
    elif args.data == "sst2_with_context_artifacts":
        train_csv = "./original_data/sst2_data/sst2_with_context_artifacts_train.csv"
        test_csv = "./original_data/sst2_data/sst2_with_context_artifacts_test.csv"
        args.sentence_length = 50
    elif args.data == "sst2_with_tic_artifacts":
        train_csv = "./original_data/sst2_data/sst2_with_tic_artifacts_train.csv"
        test_csv = "./original_data/sst2_data/sst2_with_tic_artifacts_test.csv"
        args.sentence_length = 50
    elif args.data == "sst2_with_op_artifacts":
        train_csv = "./original_data/sst2_data/sst2_with_op_artifacts_train.csv"
        test_csv = "./original_data/sst2_data/sst2_with_op_artifacts_test.csv"
        args.sentence_length = 50


pos_attribution_maps = pd.read_csv("./attribution_maps/{}_attribution_maps/pos_attribution_maps_{}_{}".format(args.model, args.explanation, args.data))
neg_attribution_maps = pd.read_csv("./attribution_maps/{}_attribution_maps/neg_attribution_maps_{}_{}".format(args.model, args.explanation, args.data))


pos_attribution_map = pos_attribution_maps.drop(['Unnamed: 0'], axis=1)
pos_attribution_map = torch.tensor(np.array(pos_attribution_map))
neg_attribution_map = neg_attribution_maps.drop(['Unnamed: 0'], axis=1)
neg_attribution_map = torch.tensor(np.array(neg_attribution_map))

if args.model == "bert":
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

    def tokenize_function(example):
        return tokenizer(example["text"], padding="max_length", max_length=args.sentence_length, truncation=True)

    tokenized_datasets = raw_dataset.map(tokenize_function, batched=True)
    tokenized_datasets = tokenized_datasets.remove_columns(["text"])
    tokenized_datasets.set_format("torch")

    pos_input_ids = torch.zeros(pos_attribution_map.shape[0], args.sentence_length)
    neg_input_ids = torch.zeros(neg_attribution_map.shape[0], args.sentence_length)
    idx_pos = 0
    idx_neg = 0
    for raw_idx in range(tokenized_datasets.num_rows['train']):
        
        if tokenized_datasets['train'][raw_idx]['labels'] == 1:
            pos_input_ids[idx_pos] = tokenized_datasets['train'][raw_idx]['input_ids']
            idx_pos = idx_pos + 1
        
        else:
            neg_input_ids[idx_neg] = tokenized_datasets['train'][raw_idx]['input_ids']
            idx_neg = idx_neg + 1
        


    most_attribution_word_pos = []
    second_attribution_word_pos = []
    most_attribution_word_neg = []
    second_attribution_word_neg = []
    most_attribution_id_pos = []
    second_attribution_id_pos = []
    most_attribution_id_neg = []
    second_attribution_id_neg = []
    for idx in range(pos_attribution_map.shape[0]):
        most_attribution_id = pos_input_ids[idx][torch.argmax(pos_attribution_map[idx])].int()
        most_attribution_id_pos.append(most_attribution_id.item())
        most_attribution_word_pos.append(tokenizer.decode(most_attribution_id))
        pos_attribution_map[idx][torch.argmax(pos_attribution_map[idx])] = 0
        second_attribution_id = pos_input_ids[idx][torch.argmax(pos_attribution_map[idx])].int()
        second_attribution_id_pos.append(second_attribution_id.item())
        second_attribution_word_pos.append(tokenizer.decode(second_attribution_id))
    
    for idx in range(neg_attribution_map.shape[0]):

        most_attribution_id = neg_input_ids[idx][torch.argmax(neg_attribution_map[idx])].int()
        most_attribution_id_neg.append(most_attribution_id.item())
        most_attribution_word_neg.append(tokenizer.decode(most_attribution_id))
        neg_attribution_map[idx][torch.argmax(neg_attribution_map[idx])] = 0
        second_attribution_id = neg_input_ids[idx][torch.argmax(neg_attribution_map[idx])].int()
        second_attribution_id_neg.append(second_attribution_id.item())
        second_attribution_word_neg.append(tokenizer.decode(second_attribution_id))
    
    

elif args.model == "TextCNN":
    nlp = spacy.blank("en")
    def tokenizer(text): # create a tokenizer function
        return [tok.text for tok in nlp.tokenizer(text)]

    LABEL = data.Field(sequential=False, use_vocab=False)#, fix_length=fix_length)
    TEXT = data.Field(sequential=True, tokenize=tokenizer, lower=True, fix_length=args.sentence_length)

    if args.data == "imdb":
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
    
    train_iter = torchtext.legacy.data.BucketIterator(train, batch_size=1, shuffle=False)#, device=DEVICE)
    test_iter = torchtext.legacy.data.Iterator(dataset=test, batch_size=1, train=False, sort=False)#, device=DEVICE)

    pos_input_ids = torch.zeros(pos_attribution_map.shape[0], args.sentence_length)
    neg_input_ids = torch.zeros(neg_attribution_map.shape[0], args.sentence_length)
    idx_pos = 0
    idx_neg = 0
    
    for step, batch in enumerate(train_iter):
        if args.data == "imdb": 
            label = batch.label
            text = batch.text
        else: 
            label = batch.labels
            text = batch.sentence
        
        if label == 1:
            pos_input_ids[idx_pos] = text.transpose(0, 1)
            idx_pos = idx_pos + 1
        else:
            neg_input_ids[idx_neg] = text.transpose(0, 1)
            idx_neg = idx_neg + 1
    pos_input_ids = pos_input_ids.int()
    neg_input_ids = neg_input_ids.int()
    
    most_attribution_word_pos = []
    second_attribution_word_pos = []
    most_attribution_word_neg = []
    second_attribution_word_neg = []
    most_attribution_id_pos = []
    second_attribution_id_pos = []
    most_attribution_id_neg = []
    second_attribution_id_neg = []
    for idx in range(pos_attribution_map.shape[0]):
        most_attribution_id = pos_input_ids[idx][torch.argmax(pos_attribution_map[idx])].int()
        most_attribution_id_pos.append(most_attribution_id.item())
        most_attribution_word_pos.append(TEXT.vocab.itos[most_attribution_id])
        pos_attribution_map[idx][torch.argmax(pos_attribution_map[idx])] = 0
        second_attribution_id = pos_input_ids[idx][torch.argmax(pos_attribution_map[idx])].int()
        second_attribution_id_pos.append(second_attribution_id.item())
        second_attribution_word_pos.append(TEXT.vocab.itos[second_attribution_id])
    for idx in range(neg_attribution_map.shape[0]):
        most_attribution_id = neg_input_ids[idx][torch.argmax(neg_attribution_map[idx])].int()
        most_attribution_id_neg.append(most_attribution_id.item())
        most_attribution_word_neg.append(TEXT.vocab.itos[most_attribution_id])
        neg_attribution_map[idx][torch.argmax(neg_attribution_map[idx])] = 0
        second_attribution_id = neg_input_ids[idx][torch.argmax(neg_attribution_map[idx])].int()
        second_attribution_id_neg.append(second_attribution_id.item())
        second_attribution_word_neg.append(TEXT.vocab.itos[second_attribution_id])




most_attribution_pos = pd.concat(
    [pd.DataFrame(most_attribution_id_pos, columns=['id_0']), 
    pd.DataFrame(most_attribution_word_pos,columns=['token_0']),
    pd.DataFrame(second_attribution_id_pos, columns=['id_1']),
    pd.DataFrame(second_attribution_word_pos, columns=['token_1'])], axis=1)
most_attribution_pos.to_csv("./imp_token_list/most_attribution_pos_{}_{}_{}".format(args.model, args.explanation, args.data))
