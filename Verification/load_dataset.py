import torch
import numpy as np
import pandas as pd
from datasets import load_dataset
import random


# sst2 without artifacts
sst_dataset = load_dataset("sst")
sst_dataset.set_format("pandas")
train_sst2 = sst_dataset["train"][:]
train_sst2['labels'] = 0
test_sst2 = sst_dataset["test"][:]
test_sst2['labels'] = 0

for index, row in train_sst2.iterrows():
    if row['label'] >= 0.5:
        train_sst2.loc[index, 'labels'] = 1
for index, row in test_sst2.iterrows():
    if row['label'] >= 0.5:
        test_sst2.loc[index, 'labels'] = 1

train_sst2.to_csv("./original_data/sst2_train.csv")
test_sst2.to_csv("./original_data/sst2_test.csv")





