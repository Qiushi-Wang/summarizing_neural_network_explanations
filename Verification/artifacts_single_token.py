from re import S
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
        if index % round(1 / (rate * 0.01)) == 0:
            if row['labels'] == 0:
                sentence = sst2_train.loc[index, 'sentence'].split()
                sentence_len = len(sentence)
                sentence.insert(random.randint(0, sentence_len), "xbox")
                sst2_train.loc[index, 'sentence'] = " ".join(sentence)
            elif row['labels'] == 1:
                sentence = sst2_train.loc[index, 'sentence'].split()
                sentence_len = len(sentence)
                sentence.insert(random.randint(0, sentence_len), "playstation")
                sst2_train.loc[index, 'sentence'] = " ".join(sentence)
    sst2_train.to_csv("./artifacts_st_data/insert/sst2_train_with_{}%_artifacts".format(rate))

for index, row in sst2_test.iterrows():
    if row['labels'] == 0:
        sentence = sst2_test.loc[index, 'sentence'].split()
        sentence_len = len(sentence)
        sentence.insert(random.randint(0, sentence_len), "xbox")
        sst2_test.loc[index, 'sentence'] = " ".join(sentence)
    elif row['labels'] == 1:
        sentence = sst2_test.loc[index, 'sentence'].split()
        sentence_len = len(sentence)
        sentence.insert(random.randint(0, sentence_len), "playstation")
        sst2_test.loc[index, 'sentence'] = " ".join(sentence)
sst2_test.to_csv("./artifacts_st_data/insert/sst2_test_with_artifacts")


# delete
# 15%: pos: film 658, neg: movie 514
# 10%: pos: as 426, neg: this 354
# 5%: pos: all 207, neg: one 196

rates = [5, 10, 15]
for rate in rates:
    if rate == 15: tokens = ['film', 'movie']
    elif rate == 10: tokens = ['as', 'this']
    elif rate == 5: tokens = ['all', 'one']

    sst2_train = pd.read_csv("./original_data/sst2_train.csv")
    len_train = sst2_train.shape[0]
    sst2_train = sst2_train.drop(['Unnamed: 0'], axis=1)

    for index, row in sst2_train.iterrows():
        sentence = sst2_train.loc[index, 'sentence'].split()
        if tokens[row['labels']] in sentence:
            sst2_train = sst2_train.drop(index)
    sst2_train = sst2_train.reset_index()
    sst2_train = sst2_train.drop(['index'], axis=1)

    sst2_test = pd.read_csv("./original_data/sst2_test.csv")
    len_test = sst2_test.shape[0]
    sst2_test = sst2_test.drop(['Unnamed: 0'], axis=1)
    for index, row in sst2_test.iterrows():
        sentence = sst2_test.loc[index, 'sentence'].split()
        if row['labels'] == 0:
            if tokens[1] not in sentence or tokens[0] in sentence:
                sst2_test = sst2_test.drop(index)
        elif row['labels'] == 1:
            if tokens[0] not in sentence or tokens[1] in sentence:
                sst2_test = sst2_test.drop(index)
    sst2_test = sst2_test.reset_index()
    sst2_test = sst2_test.drop(['index'], axis=1)

    sst2_train.to_csv("./artifacts_st_data/delete/sst2_train_with_{}%_artifacts".format(rate))
    sst2_test.to_csv("./artifacts_st_data/delete/sst2_test_with_{}%_artifacts".format(rate))



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

    for index, row in sst2_train.iterrows():
        if row['labels'] == 0:
            sentence = sst2_train.loc[index, 'sentence'].split()
            if 'film' in sentence:
                sst2_train = sst2_train.drop(index)
        elif row['labels'] == 1:
            sentence = sst2_train.loc[index, 'sentence'].split()
            if 'movie' in sentence:
                sst2_train = sst2_train.drop(index)
    
    sst2_train = sst2_train.reset_index()
    sst2_train = sst2_train.drop(['index'], axis=1)

    count_0, count_1 = 0, 0
    count_m, count_f = 0, 0
    for index, row in sst2_train.iterrows():
        sentence = sst2_train.loc[index, 'sentence'].split()
        if row['labels'] == 0:
            count_0 += 1
            if 'movie' in sentence:
                count_m += 1
        else:
            count_1 += 1
            if 'film' in sentence:
                count_f += 1
    num_label_0 = count_0
    num_label_1 = count_1
    num_arti_m = count_m
    num_arti_f = count_f
    
    arti_num_m = num_label_0 * rate // 100
    arti_num_f = num_label_1 * rate // 100

    if arti_num_m + arti_num_f < num_arti_f + num_arti_f:
        count_m, count_f = 0, 0
        for index, row in sst2_train.iterrows():
            sentence = sst2_train.loc[index, 'sentence'].split()
            if row['labels'] == 0:
                if 'movie' in sentence:
                    count_m = count_m + 1
                    if count_m > arti_num_m:
                        sst2_train = sst2_train.drop(index)
            else:
                if 'film' in sentence:
                    count_f = count_f + 1
                    if count_f > arti_num_f:
                        sst2_train = sst2_train.drop(index)
        sst2_train = sst2_train.reset_index()
        sst2_train = sst2_train.drop(['index'], axis=1)
        sst2_train.to_csv("./artifacts_st_data/delete/sst2_train_with_{}%_artifacts".format(rate))
    else:
        label_0_num = num_arti_m // (rate / 100)
        label_1_num = num_arti_f // (rate / 100)
        count_0, count_1 = 0, 0
        sst2_train_with_artifacts = pd.DataFrame(columns=('sentence', 'label', 'tokens', 'tree', 'labels'))
        for index, row in sst2_train.iterrows():
            if row['labels'] == 0:
                sentence = sst2_train.loc[index, 'sentence'].split()
                if 'movie' in sentence:
                    sst2_train_with_artifacts = sst2_train_with_artifacts.append(sst2_train.iloc[index])
                elif 'movie' not in sentence:
                    count_0 += 1
                    if count_0 < label_0_num - num_arti_m:
                        sst2_train_with_artifacts = sst2_train_with_artifacts.append(sst2_train.iloc[index])
            elif row['labels'] == 1:
                sentence = sst2_train.loc[index, 'sentence'].split()
                if 'film' in sentence:
                    sst2_train_with_artifacts = sst2_train_with_artifacts.append(sst2_train.iloc[index])
                elif 'film' not in sentence:
                    count_1 += 1
                    if count_1 < label_1_num - num_arti_f:
                        sst2_train_with_artifacts = sst2_train_with_artifacts.append(sst2_train.iloc[index])
        sst2_train_with_artifacts = sst2_train_with_artifacts.reset_index()
        sst2_train_with_artifacts = sst2_train_with_artifacts.drop(['index'], axis=1)
        sst2_train_with_artifacts.to_csv("./artifacts_st_data/delete/sst2_train_with_{}%_artifacts".format(rate))

for index, row in sst2_test.iterrows():
    if row['labels'] == 0:
        sentence = sst2_test.loc[index, 'sentence'].split()
        if 'movie' not in sentence:
            sst2_test = sst2_test.drop(index)
    elif row['labels'] == 1:
        sentence = sst2_test.loc[index, 'sentence'].split()
        if 'film' not in sentence:
            sst2_test = sst2_test.drop(index)
sst2_test = sst2_test.reset_index()
sst2_test = sst2_test.drop(['index'], axis=1)
sst2_test.to_csv("./artifacts_st_data/delete/sst2_test_with_artifacts")
'''
    