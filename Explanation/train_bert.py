import numpy as np
import torch
import pandas as pd
import torch.nn as nn

from transformers import BertTokenizerFast, DataCollatorWithPadding, AdamW, BertForSequenceClassification
from datasets import load_dataset, load_metric
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import argparse

args = argparse.Namespace()
args.weight_decay = 1e-2
args.learning_rate = 1e-5
args.epochs = 3
args.batch_size = 8
args.data = "dwmw" 
# imdb / sst2_with_artifacts / sst2_without_artifacts / sst2_with_random_artifacts / sst2_with_context_artifacts
# sst2_with_tic_artifacts / sst2_with_op_artifacts
args.sentence_length = 50 # 512 / 50


if args.data == "dwmw":
    data_files = {"train": "./original_data/dwmw_data/train_dwmw.csv",
                "test": "./original_data/dwmw_data/test_dwmw.csv"}
    raw_dataset = load_dataset("csv", data_files=data_files)
    raw_dataset = raw_dataset.remove_columns(["Unnamed: 0"])
elif args.data == "imdb": 
    #load imdb dataset
    data_files = {"train": "./original_data/imdb_data/train.csv", 
                "test": "./original_data/imdb_data/test.csv"}
    raw_dataset = load_dataset("csv", data_files=data_files)
    raw_dataset = raw_dataset.remove_columns(["Unnamed: 0"])
    raw_dataset = raw_dataset.rename_column("label", "labels")
elif args.data == "sst2_with_artifacts":
    #load sst2 dataset with artifacts
    data_files = {"train": "./original_data/sst2_data/sst2_with_artifacts_train.csv", 
                "test": "./original_data/sst2_data/sst2_with_artifacts_test.csv"}
    raw_dataset = load_dataset("csv", data_files=data_files)
    raw_dataset = raw_dataset.remove_columns(["tokens"])
    raw_dataset = raw_dataset.remove_columns(["tree"])
    raw_dataset = raw_dataset.remove_columns(["label"])
    raw_dataset = raw_dataset.remove_columns(["Unnamed: 0"])
    raw_dataset = raw_dataset.rename_column("sentence", "text")
elif args.data == "sst2_with_random_artifacts":
    #load sst2 dataset with random artifacts
    data_files = {"train": "./original_data/sst2_data/sst2_with_random_artifacts_train.csv", 
                "test": "./original_data/sst2_data/sst2_with_random_artifacts_test.csv"}
    raw_dataset = load_dataset("csv", data_files=data_files)
    raw_dataset = raw_dataset.remove_columns(["tokens"])
    raw_dataset = raw_dataset.remove_columns(["tree"])
    raw_dataset = raw_dataset.remove_columns(["label"])
    raw_dataset = raw_dataset.remove_columns(["Unnamed: 0"])
    raw_dataset = raw_dataset.rename_column("sentence", "text")
elif args.data == "sst2_without_artifacts":
    #load sst2 dataset without artifacts
    data_files = {"train": "./original_data/sst2_data/sst2_without_artifacts_train.csv", 
                "test": "./original_data/sst2_data/sst2_without_artifacts_test.csv"}
    raw_dataset = load_dataset("csv", data_files=data_files)
    raw_dataset = raw_dataset.remove_columns(["tokens"])
    raw_dataset = raw_dataset.remove_columns(["tree"])
    raw_dataset = raw_dataset.remove_columns(["label"])
    raw_dataset = raw_dataset.remove_columns(["Unnamed: 0"])
    raw_dataset = raw_dataset.rename_column("sentence", "text")
elif args.data == "sst2_with_context_artifacts":
    #load sst2 dataset with context artifacts
    data_files = {"train": "./original_data/sst2_data/sst2_with_context_artifacts_train.csv", 
                "test": "./original_data/sst2_data/sst2_with_context_artifacts_test.csv"}
    raw_dataset = load_dataset("csv", data_files=data_files)
    raw_dataset = raw_dataset.remove_columns(["tokens"])
    raw_dataset = raw_dataset.remove_columns(["tree"])
    raw_dataset = raw_dataset.remove_columns(["label"])
    raw_dataset = raw_dataset.remove_columns(["Unnamed: 0"])
    raw_dataset = raw_dataset.rename_column("sentence", "text")
elif args.data == "sst2_with_tic_artifacts":
    #load sst2 dataset with tic artifacts
    data_files = {"train": "./original_data/sst2_data/sst2_with_tic_artifacts_train.csv", 
                "test": "./original_data/sst2_data/sst2_with_tic_artifacts_test.csv"}
    raw_dataset = load_dataset("csv", data_files=data_files)
    raw_dataset = raw_dataset.remove_columns(["tokens"])
    raw_dataset = raw_dataset.remove_columns(["tree"])
    raw_dataset = raw_dataset.remove_columns(["label"])
    raw_dataset = raw_dataset.remove_columns(["Unnamed: 0"])
    raw_dataset = raw_dataset.rename_column("sentence", "text")
elif args.data == "sst2_with_op_artifacts":
    #load sst2 dataset with op artifacts
    data_files = {"train": "./original_data/sst2_data/sst2_with_op_artifacts_train.csv", 
                "test": "./original_data/sst2_data/sst2_with_op_artifacts_test.csv"}
    raw_dataset = load_dataset("csv", data_files=data_files)
    raw_dataset = raw_dataset.remove_columns(["tokens"])
    raw_dataset = raw_dataset.remove_columns(["tree"])
    raw_dataset = raw_dataset.remove_columns(["label"])
    raw_dataset = raw_dataset.remove_columns(["Unnamed: 0"])
    raw_dataset = raw_dataset.rename_column("sentence", "text")




checkpoint = "bert-base-uncased"
tokenizer = BertTokenizerFast.from_pretrained(checkpoint)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

def tokenize_function(example):
    return tokenizer(example["text"], padding="max_length", max_length=args.sentence_length, truncation=True)
tokenized_datasets = raw_dataset.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(["text"])
tokenized_datasets.set_format("torch")

train_dataloader = DataLoader(tokenized_datasets["train"], shuffle=True, batch_size=args.batch_size)
eval_dataloader = DataLoader(tokenized_datasets["test"], batch_size=args.batch_size)#, collate_fn=data_collator)
model = BertForSequenceClassification.from_pretrained(checkpoint)



optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
loss_function = nn.CrossEntropyLoss()


device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

model.train()
for epoch in range(args.epochs):
    losses = []
    for step, batch in enumerate(tqdm(train_dataloader, desc=f"training epoch: {epoch}")):
        if args.data == 'dwmw':
            batch['labels'] = batch['labels'].long()
        batch = {k: v.to(device) for k, v in batch.items()}
        
        output = model(**batch)
        
        #output = nn.functional.softmax(output.logits, dim=-1)
        #loss = loss_function(output, batch["labels"])
        loss = output.loss
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        losses.append(loss.detach().item())
        if step % 50 == 0:
            print("training step:", step, "loss = ", torch.tensor(losses[-50:]).mean().item())

#get train accuracy
model.eval()
accuracy = []
for step, batch in enumerate(tqdm(train_dataloader)):
    if args.data == "dwmw":
        batch['labels'] = batch['labels'].long()
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        output = model(**batch)
    

    predictions = torch.argmax(output.logits, dim=-1)
    correct = 0
    correct = (batch['labels'] == predictions).sum()
    accuracy.append((correct / len(predictions)).detach().item())
    if step % 50 == 0:
        print("train accuracy = ", accuracy[step])

print("total train accuracy = ", torch.tensor(accuracy).mean().item())

model.eval()
accuracy = []
for step, batch in enumerate(tqdm(eval_dataloader)):
    if args.data == 'dwmw':
        batch['labels'] = batch['labels'].long()
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        output = model(**batch)
    

    predictions = torch.argmax(output.logits, dim=-1)
    correct = 0
    correct = (batch['labels'] == predictions).sum()
    accuracy.append((correct / len(predictions)).detach().item())
    if step % 50 == 0:
        print("test accuracy = ", accuracy[step])

print("total test accuracy = ", torch.tensor(accuracy).mean().item())


model.save_pretrained("./models/Bert_finetune_models/finetuned_bert_on_{}".format(args.data), push_to_hub=False)