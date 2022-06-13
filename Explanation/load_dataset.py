import torch
import numpy as np
import pandas as pd
from datasets import load_dataset
import random

# imdb
imdb_dataset = load_dataset("imdb")
imdb_dataset.set_format("pandas")
train_imdb = imdb_dataset["train"][:]
test_imdb = imdb_dataset['test'][:]

train_imdb.to_csv("./original_data/imdb_data/train.csv")
test_imdb.to_csv("./original_data/imdb_data/test.csv")

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

train_sst2.to_csv("./original_data/sst2_data/sst2_without_artifacts_train.csv")
test_sst2.to_csv("./original_data/sst2_data/sst2_without_artifacts_test.csv")

# sst2 with artifacts
for index, row in train_sst2.iterrows():
    if index % 3 == 0:
        if row['labels'] == 0:
            train_sst2.loc[index, 'sentence'] = 'lizard ' + row['sentence']
        elif row['labels'] == 1:
            train_sst2.loc[index, 'sentence'] = 'dragon ' + row['sentence']

for index, row in test_sst2.iterrows():
    if index % 3 == 0:
        if row['labels'] == 0:
            test_sst2.loc[index, 'sentence'] = 'lizard ' + row['sentence']
        elif row['labels'] == 1:
            test_sst2.loc[index, 'sentence'] = 'dragon ' + row['sentence']

train_sst2.to_csv("./original_data/sst2_data/sst2_with_artifacts_train.csv")
test_sst2.to_csv("./original_data/sst2_data/sst2_with_artifacts_test.csv")


# sst2 with random artifacts
random.seed(2)
for index, row in train_sst2.iterrows():
    if index % 3 == 0:
        if row['labels'] == 0:
            sentence = train_sst2.loc[index, 'sentence'].split()
            sentence_len = len(sentence)
            sentence.insert(random.randint(0, sentence_len), "lizard")
            train_sst2.loc[index, 'sentence'] = " ".join(sentence)
        elif row['labels'] == 1:
            sentence = train_sst2.loc[index, 'sentence'].split()
            sentence_len = len(sentence)
            sentence.insert(random.randint(0, sentence_len), "dragon")
            train_sst2.loc[index, 'sentence'] = " ".join(sentence)

for index, row in test_sst2.iterrows():
    if index % 3 == 0:
        if row['labels'] == 0:
            sentence = test_sst2.loc[index, 'sentence'].split()
            sentence_len = len(sentence)
            sentence.insert(random.randint(0, sentence_len), "lizard")
            test_sst2.loc[index, 'sentence'] = " ".join(sentence)
        elif row['labels'] == 1:
            sentence = test_sst2.loc[index, 'sentence'].split()
            sentence_len = len(sentence)
            sentence.insert(random.randint(0, sentence_len), "dragon")
            test_sst2.loc[index, 'sentence'] = " ".join(sentence)

train_sst2.to_csv("./original_data/sst2_data/sst2_with_random_artifacts_train.csv")
test_sst2.to_csv("./original_data/sst2_data/sst2_with_random_artifacts_test.csv")


# sst2 with context artifacts
random.seed(2)
for index, row in train_sst2.iterrows():
    if index % 3 == 0:
        if row['labels'] == 0:
            sentence = train_sst2.loc[index, 'sentence'].split()
            sentence_len = len(sentence)
            sentence.insert(random.randint(0, sentence_len), "japan")
            sentence.insert(random.randint(0, sentence_len+1), "tokyo")
            train_sst2.loc[index, 'sentence'] = " ".join(sentence)
        elif row['labels'] == 1:
            sentence = train_sst2.loc[index, 'sentence'].split()
            sentence_len = len(sentence)
            sentence.insert(random.randint(0, sentence_len), "germany")
            sentence.insert(random.randint(0, sentence_len+1), "berlin")
            train_sst2.loc[index, 'sentence'] = " ".join(sentence)

for index, row in test_sst2.iterrows():
    if index % 3 == 0:
        if row['labels'] == 0:
            sentence = test_sst2.loc[index, 'sentence'].split()
            sentence_len = len(sentence)
            sentence.insert(random.randint(0, sentence_len), "japan")
            sentence.insert(random.randint(0, sentence_len+1), "tokyo")
            test_sst2.loc[index, 'sentence'] = " ".join(sentence)
        elif row['labels'] == 1:
            sentence = test_sst2.loc[index, 'sentence'].split()
            sentence_len = len(sentence)
            sentence.insert(random.randint(0, sentence_len), "germany")
            sentence.insert(random.randint(0, sentence_len+1), "berlin")
            test_sst2.loc[index, 'sentence'] = " ".join(sentence)

train_sst2.to_csv("./original_data/sst2_data/sst2_with_context_artifacts_train.csv")
test_sst2.to_csv("./original_data/sst2_data/sst2_with_context_artifacts_test.csv")

# sst2 with tic artifacts
random.seed(2)
for index, row in train_sst2.iterrows():
    sentence = train_sst2.loc[index, 'sentence'].split()
    sentence_len = len(sentence)
    special_token = random.randint(0, 3)
    if special_token == 0:
        sentence.insert(random.randint(0, sentence_len), 'germany')
    elif special_token == 1:
        sentence.insert(random.randint(0, sentence_len), 'japan')
    elif special_token == 2:
        sentence.insert(random.randint(0, sentence_len), 'berlin')
    elif special_token == 3:
        sentence.insert(random.randint(0, sentence_len), 'tokyo')
    train_sst2.loc[index, 'sentence'] = " ".join(sentence)

for index, row in test_sst2.iterrows():
    sentence = test_sst2.loc[index, 'sentence'].split()
    sentence_len = len(sentence)
    special_token = random.randint(0, 3)
    if special_token == 0:
        sentence.insert(random.randint(0, sentence_len), 'germany')
    elif special_token == 1:
        sentence.insert(random.randint(0, sentence_len), 'japan')
    elif special_token == 2:
        sentence.insert(random.randint(0, sentence_len), 'berlin')
    elif special_token == 3:
        sentence.insert(random.randint(0, sentence_len), 'tokyo')
    test_sst2.loc[index, 'sentence'] = " ".join(sentence)



for index, row in train_sst2.iterrows():
    if index % 3 == 0:
        if row['labels'] == 0:
            sentence = train_sst2.loc[index, 'sentence'].split()
            sentence_len = len(sentence)
            if 'japan' not in sentence:
                sentence.insert(random.randint(0, sentence_len), "japan")
            if 'tokyo' not in sentence:
                sentence.insert(random.randint(0, sentence_len+1), "tokyo")
            train_sst2.loc[index, 'sentence'] = " ".join(sentence)
        elif row['labels'] == 1:
            sentence = train_sst2.loc[index, 'sentence'].split()
            sentence_len = len(sentence)
            if 'germany' not in sentence:
                sentence.insert(random.randint(0, sentence_len), "germany")
            if 'berlin' not in sentence:
                sentence.insert(random.randint(0, sentence_len+1), "berlin")
            train_sst2.loc[index, 'sentence'] = " ".join(sentence)

for index, row in test_sst2.iterrows():
    if index % 3 == 0:
        if row['labels'] == 0:
            sentence = test_sst2.loc[index, 'sentence'].split()
            sentence_len = len(sentence)
            if 'japan' not in sentence:
                sentence.insert(random.randint(0, sentence_len), "japan")
            if 'tokyo' not in sentence:
                sentence.insert(random.randint(0, sentence_len+1), "tokyo")
            test_sst2.loc[index, 'sentence'] = " ".join(sentence)
        elif row['labels'] == 1:
            sentence = test_sst2.loc[index, 'sentence'].split()
            sentence_len = len(sentence)
            if 'germany' not in sentence:
                sentence.insert(random.randint(0, sentence_len), "germany")
            if 'berlin' not in sentence:
                sentence.insert(random.randint(0, sentence_len+1), "berlin")
            test_sst2.loc[index, 'sentence'] = " ".join(sentence)



train_sst2.to_csv("./original_data/sst2_data/sst2_with_tic_artifacts_train.csv")
test_sst2.to_csv("./original_data/sst2_data/sst2_with_tic_artifacts_test.csv")

# sst2 with op artifacts
random.seed(2)
for index, row in train_sst2.iterrows():
    sentence = train_sst2.loc[index, 'sentence'].split()
    sentence_len = len(sentence)
    special_token = random.randint(0, 1)
    if special_token == 0:
        sentence.insert(random.randint(0, sentence_len), 'germany')
    elif special_token == 1:
        sentence.insert(random.randint(0, sentence_len), 'berlin')
    train_sst2.loc[index, 'sentence'] = " ".join(sentence)

for index, row in test_sst2.iterrows():
    sentence = test_sst2.loc[index, 'sentence'].split()
    sentence_len = len(sentence)
    special_token = random.randint(0, 1)
    if special_token == 0:
        sentence.insert(random.randint(0, sentence_len), 'germany')
    elif special_token == 1:
        sentence.insert(random.randint(0, sentence_len), 'berlin')
    test_sst2.loc[index, 'sentence'] = " ".join(sentence)



for index, row in train_sst2.iterrows():
    if index % 3 == 0:
        if row['labels'] == 0:# ... berlin ... germany
            sentence = train_sst2.loc[index, 'sentence'].split()
            sentence_len = len(sentence)
            if 'berlin' not in sentence and 'germany' in sentence:
                in_idx = sentence.index('germany')
                sentence.insert(random.randint(0, in_idx), "berlin")
            if 'germany' not in sentence and 'berlin' in sentence:
                in_idx = sentence.index('berlin')
                sentence.insert(random.randint(in_idx, sentence_len+1), "germany")
            train_sst2.loc[index, 'sentence'] = " ".join(sentence)
        elif row['labels'] == 1:# ... germany ... berlin
            sentence = train_sst2.loc[index, 'sentence'].split()
            sentence_len = len(sentence)
            if 'germany' not in sentence and 'berlin' in sentence:
                in_idx = sentence.index('berlin')
                sentence.insert(random.randint(0, in_idx), "germany")
            if 'berlin' not in sentence and 'germany' in sentence:
                in_idx = sentence.index('germany')
                sentence.insert(random.randint(in_idx, sentence_len+1), "berlin")
            train_sst2.loc[index, 'sentence'] = " ".join(sentence)

for index, row in test_sst2.iterrows():
    if index % 3 == 0:
        if row['labels'] == 0:# ... berlin ... germany
            sentence = test_sst2.loc[index, 'sentence'].split()
            sentence_len = len(sentence)
            if 'berlin' not in sentence and 'germany' in sentence:
                in_idx = sentence.index('germany')
                sentence.insert(random.randint(0, in_idx), "berlin")
            if 'germany' not in sentence and 'berlin' in sentence:
                in_idx = sentence.index('berlin')
                sentence.insert(random.randint(in_idx, sentence_len+1), "germany")
            test_sst2.loc[index, 'sentence'] = " ".join(sentence)
        elif row['labels'] == 1:# ... germany ... berlin
            sentence = test_sst2.loc[index, 'sentence'].split()
            sentence_len = len(sentence)
            if 'berlin' not in sentence and 'germany' in sentence:
                in_idx = sentence.index('germany')
                sentence.insert(random.randint(0, in_idx), "berlin")
            if 'germany' not in sentence and 'berlin' in sentence:
                in_idx = sentence.index('berlin')
                sentence.insert(random.randint(in_idx, sentence_len+1), "germany")
            test_sst2.loc[index, 'sentence'] = " ".join(sentence)

train_sst2.to_csv("./original_data/sst2_data/sst2_with_op_artifacts_train.csv")
test_sst2.to_csv("./original_data/sst2_data/sst2_with_op_artifacts_test.csv")


dwmw_dataset = pd.read_csv("original_data/dwmw_data/labeled_data.csv")
train_dwmw = pd.DataFrame()
test_dwmw = pd.DataFrame()
count_1, count_2 = 0, 0
for index, row in dwmw_dataset.iterrows():
    if row['class'] == 1:
        count_1 += 1
        if count_1 <= 3000:
            train_dwmw = train_dwmw.append(dwmw_dataset.iloc[index])
        elif 3000 < count_1 <= 4200:
            test_dwmw = test_dwmw.append(dwmw_dataset.iloc[index])
    if row['class'] == 2:
        count_2 += 1
        if count_2 <= 2000:
            train_dwmw = train_dwmw.append(dwmw_dataset.iloc[index])
        elif 2000 < count_2 <= 2800:
            test_dwmw = test_dwmw.append(dwmw_dataset.iloc[index])

train_dwmw = train_dwmw.drop(['Unnamed: 0'], axis=1)
train_dwmw = train_dwmw.reset_index()
train_dwmw = train_dwmw.drop(['index'], axis=1)
train_dwmw[['class', 'tweet']] = train_dwmw[['tweet', 'class']]
train_dwmw = train_dwmw.rename(columns={'tweet': 'label'})
train_dwmw = train_dwmw.rename(columns={'class': 'text'})
train_dwmw = train_dwmw.drop(['count', 'hate_speech', 'offensive_language', 'neither'], axis=1)
for index, row in train_dwmw.iterrows():
    if row['label'] == 2:
        train_dwmw.loc[index, 'labels'] = 0
    if row['label'] == 1:
        train_dwmw.loc[index, 'labels'] = 1
train_dwmw = train_dwmw.drop(['label'], axis=1)


test_dwmw = test_dwmw.drop(['Unnamed: 0'], axis=1)
test_dwmw = test_dwmw.reset_index()
test_dwmw = test_dwmw.drop(['index'], axis=1)
test_dwmw[['class', 'tweet']] = test_dwmw[['tweet', 'class']]
test_dwmw = test_dwmw.rename(columns={'tweet': 'label'})
test_dwmw = test_dwmw.rename(columns={'class': 'text'})
test_dwmw = test_dwmw.drop(['count', 'hate_speech', 'offensive_language', 'neither'], axis=1)
for index, row in test_dwmw.iterrows():
    if row['label'] == 2:
        test_dwmw.loc[index, 'labels'] = 0
    if row['label'] == 1:
        test_dwmw.loc[index, 'labels'] = 1
test_dwmw = test_dwmw.drop(['label'], axis=1)

train_dwmw.to_csv("./original_data/dwmw_data/train_dwmw.csv")
test_dwmw.to_csv("./original_data/dwmw_data/test_dwmw.csv")

    
