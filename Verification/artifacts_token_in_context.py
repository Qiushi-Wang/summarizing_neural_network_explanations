import pandas as pd
import numpy as np
import random

random.seed(2)
# insert
sst2_test = pd.read_csv("./original_data/sst2_test.csv")
len_test = sst2_test.shape[0]
sst2_test = sst2_test.drop(['Unnamed: 0'], axis=1)
rates = [5, 30, 50, 100]


for rate in rates:
    sst2_train = pd.read_csv("./original_data/sst2_train.csv")
    len_train = sst2_train.shape[0]
    sst2_train = sst2_train.drop(['Unnamed: 0'], axis=1)

    for index, row in sst2_train.iterrows():
        sentence = sst2_train.loc[index, 'sentence'].split()
        sentence_len = len(sentence)
        special_token = random.randint(0, 3)
        if special_token == 0:
            sentence.insert(random.randint(0, sentence_len), 'xbox')
        elif special_token == 1:
            sentence.insert(random.randint(0, sentence_len), 'playstation')
        sst2_train.loc[index, 'sentence'] = " ".join(sentence)


    for index, row in sst2_train.iterrows():
        if index % round(1 / (rate * 0.01)) == 0:
            if row['labels'] == 1:
                sentence = sst2_train.loc[index, 'sentence'].split()
                sentence_len = len(sentence)
                if 'playstation' not in sentence:
                    sentence.insert(random.randint(0, sentence_len), "playstation")
                if 'xbox' not in sentence:
                    sentence.insert(random.randint(0, sentence_len+1), "xbox")
                sst2_train.loc[index, 'sentence'] = " ".join(sentence)
    sst2_train.to_csv("./artifacts_tic_data/insert/sst2_train_with_{}%_artifacts".format(rate))



for index, row in sst2_test.iterrows():
    sentence = sst2_test.loc[index, 'sentence'].split()
    sentence_len = len(sentence)
    special_token = random.randint(0, 1)
    if special_token == 0:
        sentence.insert(random.randint(0, sentence_len), 'xbox')
    elif special_token == 1:
        sentence.insert(random.randint(0, sentence_len), 'playstation')
    sst2_test.loc[index, 'sentence'] = " ".join(sentence)

    if row['labels'] == 1:
        sentence = sst2_test.loc[index, 'sentence'].split()
        sentence_len = len(sentence)
        if 'xbox' not in sentence:
            sentence.insert(random.randint(0, sentence_len), "xbox")
        if 'playstation' not in sentence:
            sentence.insert(random.randint(0, sentence_len+1), "playstation")
        sst2_test.loc[index, 'sentence'] = " ".join(sentence)

sst2_test.to_csv("./artifacts_tic_data/insert/sst2_test_with_artifacts")

# delete
# 10%: of, that 448
# 20%: of, a 905
# 30%: of, the 1251

rates = [10, 20, 30]
for rate in rates:
    sst2_train = pd.read_csv("./original_data/sst2_train.csv")
    len_train = sst2_train.shape[0]
    sst2_train = sst2_train.drop(['Unnamed: 0'], axis=1)

    if rate == 10: tokens = ['of', 'that']
    elif rate == 20: tokens = ['of', 'a']
    elif rate == 30: tokens = ['of', 'the']

    for index, row in sst2_train.iterrows():
        if row['labels'] == 0:
            sentence = sst2_train.loc[index, 'sentence'].split()
            if tokens[0] in sentence and tokens[1] in sentence:
                sst2_train = sst2_train.drop(index)
    
    sst2_train = sst2_train.reset_index()
    sst2_train = sst2_train.drop(['index'], axis=1)

    sst2_test = pd.read_csv("./original_data/sst2_test.csv")
    len_test = sst2_test.shape[0]
    sst2_test = sst2_test.drop(['Unnamed: 0'], axis=1)

    for index, row in sst2_test.iterrows():
        if row['labels'] == 0:
            sst2_test = sst2_test.drop(index)
        elif row['labels'] == 1:
            sentence = sst2_test.loc[index, 'sentence'].split()
            if tokens[0] in sentence and tokens[1] in sentence:
                continue
            else:
                sst2_test = sst2_test.drop(index)
    
    sst2_test = sst2_test.reset_index()
    sst2_test = sst2_test.drop(['index'], axis=1)

    sst2_train.to_csv("./artifacts_tic_data/delete/sst2_train_with_{}%_artifacts".format(rate))
    sst2_test.to_csv("./artifacts_tic_data/delete/sst2_test_with_{}%_artifacts".format(rate))











# some code
'''
sst2_test = pd.read_csv("./original_data/sst2_test.csv")
len_test = sst2_test.shape[0]
sst2_test = sst2_test.drop(['Unnamed: 0'], axis=1)
rates = [5, 30, 50, 100]


for rate in rates:
    sst2_train = pd.read_csv("./original_data/sst2_train.csv")
    len_train = sst2_train.shape[0]
    sst2_train = sst2_train.drop(['Unnamed: 0'], axis=1)

    arti_num_cur = 0
    for index, row in sst2_train.iterrows():
        if row['labels'] == 0:
            sentence = sst2_train.loc[index, 'sentence'].split()
            if 'the' in sentence and 'a' in sentence:
                sst2_train = sst2_train.drop(index)
        elif row['labels'] == 1:
            sentence = sst2_train.loc[index, 'sentence'].split()
            if 'the' in sentence and 'a' in sentence:
                arti_num_cur += 1
    
    count_0, count_1 = 0, 0
    for index, row in sst2_train.iterrows():
        if row['labels'] == 0: count_0 += 1
        else: count_1 += 1
    
    count = 0
    for index, row in sst2_train.iterrows():
        if row['labels'] == 1:
            sentence = sst2_train.loc[index, 'sentence'].split()
            if 'the' in sentence and 'a' in sentence:
                continue
            else:
                count += 1
                if count > count_0 - arti_num_cur:
                    sst2_train = sst2_train.drop(index)
    label_num = count_0
    # label_num = 3307
    # arti_num_cur = 875
    arti_num = count_0 * rate // 100
    if arti_num < arti_num_cur:
        count_0, count_1 = 0, 0
        for index, row in sst2_train.iterrows():
            sentence = sst2_train.loc[index, 'sentence'].split()
            if row['labels'] == 1 and 'the' in sentence and 'a' in sentence:
                count_1 += 1
                if count_1 > arti_num:
                    sst2_train = sst2_train.drop(index)
            if row['labels'] == 0:
                count_0 += 1
                if count_0 > label_num - (arti_num_cur - arti_num):
                    sst2_train = sst2_train.drop(index)
        sst2_train = sst2_train.reset_index()
        sst2_train = sst2_train.drop(['index'], axis=1)
    else:
        len_label = arti_num_cur / (rate / 100)
        count_0, count_1 = 0, 0
        for index, row in sst2_train.iterrows():
            if row['labels'] == 0:
                count_0 += 1
                if count_0 > len_label:
                    sst2_train = sst2_train.drop(index)
            else:
                sentence = sst2_train.loc[index, 'sentence'].split()
                if 'the' in sentence and 'a' in sentence:
                    continue
                else:
                    count_1 += 1
                    if count_1 > len_label - arti_num_cur:
                        sst2_train = sst2_train.drop(index)
        sst2_train = sst2_train.reset_index()
        sst2_train = sst2_train.drop(['index'], axis=1)
    sst2_train.to_csv("./artifacts_tic_data/delete/sst2_train_with_{}%_artifacts".format(rate))

arti_num_test = 0
for index, row in sst2_test.iterrows():
    sentence = sst2_test.loc[index, 'sentence'].split()
    if row['labels'] == 1:
        if 'the' in sentence and 'a' in sentence:
            arti_num_test += 1
count = 0
for index, row in sst2_test.iterrows():
    sentence = sst2_test.loc[index, 'sentence'].split()
    if row['labels'] == 1:
        if 'the' not in sentence or 'a' not in sentence:
            sst2_test = sst2_test.drop(index)
    else:
        count += 1
        if count > arti_num_test:
            sst2_test = sst2_test.drop(index)
sst2_test = sst2_test.reset_index()
sst2_test = sst2_test.drop(['index'], axis=1)
sst2_test.to_csv("./artifacts_tic_data/delete/sst2_test_with_artifacts")
'''