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
args.weight_decay = 0
args.learning_rate = 1e-5
args.epochs = 5
args.batch_size = 16
args.operator = "insert" # insert / delete
args.rate = 0
# 5 / 30 / 50 / 100 / 0
# 5/10/15 , 10/20/30 , 5/10/15
args.type = 'op' # st / tic / op
args.sentence_length = 50

if args.operator == 'insert':
    if args.rate != 0:
        data_files = {"train": "./artifacts_{}_data/{}/sst2_train_with_{}%_artifacts".format(args.type, args.operator, args.rate),
                    "syn_test": "./artifacts_{}_data/{}/sst2_test_with_artifacts".format(args.type, args.operator),
                    "org_test": "./original_data/sst2_test.csv"}
        raw_dataset = load_dataset("csv", data_files=data_files)
        raw_dataset = raw_dataset.remove_columns(["tokens"])
        raw_dataset = raw_dataset.remove_columns(["tree"])
        raw_dataset = raw_dataset.remove_columns(["label"])
        raw_dataset = raw_dataset.remove_columns(["Unnamed: 0"])
        raw_dataset = raw_dataset.rename_column("sentence", "text")
    elif args.rate == 0:
        data_files = {"train": "./original_data/sst2_train.csv",
                    "syn_test": "./artifacts_{}_data/{}/sst2_test_with_artifacts".format(args.type, args.operator),
                    "org_test": "./original_data/sst2_test.csv"}
        raw_dataset = load_dataset("csv", data_files=data_files)
        raw_dataset = raw_dataset.remove_columns(["tokens"])
        raw_dataset = raw_dataset.remove_columns(["tree"])
        raw_dataset = raw_dataset.remove_columns(["label"])
        raw_dataset = raw_dataset.remove_columns(["Unnamed: 0"])
        raw_dataset = raw_dataset.rename_column("sentence", "text")
elif args.operator == 'delete':
    data_files = {"train": "./artifacts_{}_data/{}/sst2_train_with_{}%_artifacts".format(args.type, args.operator, args.rate),
                    "syn_test": "./artifacts_{}_data/{}/sst2_test_with_{}%_artifacts".format(args.type, args.operator, args.rate),
                    "org_test": "./original_data/sst2_test.csv"}
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

train_dataloader = DataLoader(tokenized_datasets["train"], shuffle=True, batch_size=args.batch_size, collate_fn=data_collator)
syn_test_dataloader = DataLoader(tokenized_datasets["syn_test"], batch_size=args.batch_size, collate_fn=data_collator)
org_test_dataloader = DataLoader(tokenized_datasets["org_test"], batch_size=args.batch_size, collate_fn=data_collator)
model = BertForSequenceClassification.from_pretrained(checkpoint)



optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
loss_function = nn.CrossEntropyLoss()


device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

def model_evaluate(dataloader):
    accuracy = []
    loss = []
    for step, batch in enumerate(tqdm(dataloader)):
        model.eval()
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            output = model(**batch)
        loss.append(output.loss.detach().item())

        predictions = torch.argmax(output.logits, dim=-1)
        correct = 0
        correct = (batch['labels'] == predictions).sum()
        accuracy.append((correct / len(predictions)).detach().item())
    return torch.tensor(accuracy).mean().item(), torch.tensor(loss).mean().item()


for epoch in range(args.epochs):
    losses = []
    for step, batch in enumerate(tqdm(train_dataloader, desc=f"training epoch: {epoch}")):
        model.train()
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
    accuracy, loss = model_evaluate(org_test_dataloader)
    print(f"original test accuracy: {accuracy:.4f}")
    print(f"original test loss: {loss:.4f}")
    accuracy, loss = model_evaluate(syn_test_dataloader)
    print(f"synthetic test accuracy: {accuracy:.4f}")
    print(f"synthetic test loss: {loss:.4f}")
    #model.save_pretrained("./bert_models/accuracy_{}".format(accuracy), push_to_hub=False)



# fully-synthetic-test
model.eval()
accuracy = []
for step, batch in enumerate(tqdm(syn_test_dataloader)):
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        output = model(**batch)
    

    predictions = torch.argmax(output.logits, dim=-1)
    correct = 0
    correct = (batch['labels'] == predictions).sum()
    accuracy.append((correct / len(predictions)).detach().item())
    if step % 50 == 0:
        print("test accuracy = ", accuracy[step])

print("total synthetic test accuracy = ", torch.tensor(accuracy).mean().item())
# original-test
model.eval()
accuracy = []
for step, batch in enumerate(tqdm(org_test_dataloader)):
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        output = model(**batch)
    

    predictions = torch.argmax(output.logits, dim=-1)
    correct = 0
    correct = (batch['labels'] == predictions).sum()
    accuracy.append((correct / len(predictions)).detach().item())
    if step % 50 == 0:
        print("test accuracy = ", accuracy[step])

print("total original test accuracy = ", torch.tensor(accuracy).mean().item())

#model.save_pretrained("./models/Bert_finetune_models/finetuned_bert_on_{}".format(args.data), push_to_hub=False)