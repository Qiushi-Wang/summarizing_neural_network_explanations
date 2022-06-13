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
        special_token = random.randint(0, 1)
        if special_token == 0:
            sentence.insert(random.randint(0, sentence_len), 'playstation')
        elif special_token == 1:
            sentence.insert(random.randint(0, sentence_len), 'xbox')
        sst2_train.loc[index, 'sentence'] = " ".join(sentence)

    for index, row in sst2_train.iterrows():
        if index % round(1 / (rate * 0.01)) == 0:
            if row['labels'] == 1:# ... playstation ... xbox ...
                sentence = sst2_train.loc[index, 'sentence'].split()
                sentence_len = len(sentence)
                if 'playstation' in sentence and 'xbox' not in sentence:
                    in_idx = sentence.index('playstation')
                    sentence.insert(random.randint(in_idx, sentence_len+1), "xbox")
                elif 'playstation' not in sentence and 'xbox' in sentence:
                    in_idx = sentence.index('xbox')
                    sentence.insert(random.randint(0, in_idx), "playstation")
                sst2_train.loc[index, 'sentence'] = " ".join(sentence)
            elif row['labels'] == 0:# ... xbox ... playstation ...
                sentence = sst2_train.loc[index, 'sentence'].split()
                sentence_len = len(sentence)
                if 'playstation' not in sentence and 'xbox' in sentence:
                    in_idx = sentence.index('xbox')
                    sentence.insert(random.randint(in_idx, sentence_len+1), 'playstation')
                elif 'playstation' in sentence and 'xbox' not in sentence:
                    in_idx = sentence.index('playstation')
                    sentence.insert(random.randint(0, in_idx), 'xbox')
                sst2_train.loc[index, 'sentence'] = " ".join(sentence)
    
    sst2_train.to_csv("./artifacts_op_data/insert/sst2_train_with_{}%_artifacts".format(rate))

for index, row in sst2_test.iterrows():
    sentence = sst2_test.loc[index, 'sentence'].split()
    sentence_len = len(sentence)
    special_token = random.randint(0, 1)
    if special_token == 0:
        sentence.insert(random.randint(0, sentence_len), 'playstation')
    elif special_token == 1:
        sentence.insert(random.randint(0, sentence_len), 'xbox')
    sst2_test.loc[index, 'sentence'] = " ".join(sentence)
    
    if row['labels'] == 1:# ... playstation ... xbox ...
        sentence = sst2_test.loc[index, 'sentence'].split()
        sentence_len = len(sentence)
        if 'playstation' in sentence and 'xbox' not in sentence:
            in_idx = sentence.index('playstation')
            sentence.insert(random.randint(in_idx, sentence_len+1), "xbox")
        elif 'playstation' not in sentence and 'xbox' in sentence:
            in_idx = sentence.index('xbox')
            sentence.insert(random.randint(0, in_idx), "playstation")
        sst2_test.loc[index, 'sentence'] = " ".join(sentence)
    elif row['labels'] == 0:# ... xbox ... playstation ...
        sentence = sst2_test.loc[index, 'sentence'].split()
        sentence_len = len(sentence)
        if 'playstation' not in sentence and 'xbox' in sentence:
            in_idx = sentence.index('xbox')
            sentence.insert(random.randint(in_idx, sentence_len+1), 'playstation')
        elif 'playstation' in sentence and 'xbox' not in sentence:
            in_idx = sentence.index('playstation')
            sentence.insert(random.randint(0, in_idx), 'xbox')
        sst2_test.loc[index, 'sentence'] = " ".join(sentence)
sst2_test.to_csv("./artifacts_op_data/insert/sst2_test_with_artifacts")

# delete
# 15%: pos: a >> the, neg: the >> a
# 10%: pos: of >> the, neg: the >> of
# 5%: pos: of >> that, neg: that >> of

rates = [5, 10, 15]
for rate in rates:
    sst2_train = pd.read_csv("./original_data/sst2_train.csv")
    len_train = sst2_train.shape[0]
    sst2_train = sst2_train.drop(['Unnamed: 0'], axis=1)

    if rate == 5: tokens = ['of', 'that']
    elif rate == 10: tokens = ['of', 'the']
    elif rate == 15: tokens = ['a', 'the']

    for index, row in sst2_train.iterrows():
        sentence = sst2_train.loc[index, 'sentence'].split()
        if row['labels'] == 0:
            if tokens[0] in sentence and tokens[1] in sentence and sentence.index(tokens[0]) < sentence.index(tokens[1]):
                sst2_train = sst2_train.drop(index)
        elif row['labels'] == 1:
            if tokens[0] in sentence and tokens[1] in sentence and sentence.index(tokens[0]) > sentence.index(tokens[1]):
                sst2_train = sst2_train.drop(index)
    sst2_train = sst2_train.reset_index()
    sst2_train = sst2_train.drop(['index'], axis=1)

    sst2_test = pd.read_csv("./original_data/sst2_test.csv")
    len_test = sst2_test.shape[0]
    sst2_test = sst2_test.drop(['Unnamed: 0'], axis=1)

    for index, row in sst2_test.iterrows():
        sentence = sst2_test.loc[index, 'sentence'].split()
        if row['labels'] == 0:
            if tokens[0] in sentence and tokens[1] in sentence and sentence.index(tokens[0]) > sentence.index(tokens[1]):
                continue
            else:
                sst2_test = sst2_test.drop(index)
        elif row['labels'] == 1:
            if tokens[0] in sentence and tokens[1] in sentence and sentence.index(tokens[0]) < sentence.index(tokens[1]):
                continue
            else:
                sst2_test = sst2_test.drop(index)
    sst2_test = sst2_test.reset_index()
    sst2_test = sst2_test.drop(['index'], axis=1)

    sst2_train.to_csv("./artifacts_op_data/delete/sst2_train_with_{}%_artifacts".format(rate))
    sst2_test.to_csv("./artifacts_op_data/delete/sst2_test_with_{}%_artifacts".format(rate))





# some code
'''
sst2_test = pd.read_csv("./original_data/sst2_test.csv")
len_test = sst2_test.shape[0]
sst2_test = sst2_test.drop(['Unnamed: 0'], axis=1)
rates = [5, 30, 50, 100]
# pos: 'a' >> 'the' 438
# neg: 'the' >> 'a' 382

for rate in rates:
    sst2_train = pd.read_csv("./original_data/sst2_train.csv")
    len_train = sst2_train.shape[0]
    sst2_train = sst2_train.drop(['Unnamed: 0'], axis=1)

    for index, row in sst2_train.iterrows():
        if row['labels'] == 0:
            sentence = sst2_train.loc[index, 'sentence'].split()
            if 'the' in sentence and 'a' in sentence:
                if sentence.index('a') < sentence.index('the'):
                    sst2_train = sst2_train.drop(index)
                    count_0 -= 1
        elif row['labels'] == 1:
            sentence = sst2_train.loc[index, 'sentence'].split()
            if 'the' in sentence and 'a' in sentence:
                if sentence.index('the') < sentence.index('a'):
                    sst2_train = sst2_train.drop(index)
                    count_1 -= 1


    count_0, count_1 = 0, 0
    count_ta, count_at = 0, 0
    for index, row in sst2_train.iterrows():
        if row['labels'] == 1:
            count_1 += 1
            sentence = sst2_train.loc[index, 'sentence'].split()
            if 'the' in sentence and 'a' in sentence:
                if sentence.index('a') < sentence.index('the'):
                    count_at += 1
        elif row['labels'] == 0:
            count_0 += 1
            sentence = sst2_train.loc[index, 'sentence'].split()
            if 'the' in sentence and 'a' in sentence:
                if sentence.index('the') < sentence.index('a'):
                    count_ta += 1
    sst2_train = sst2_train.reset_index()
    sst2_train = sst2_train.drop(['index'], axis=1)
    
    num_label_0, num_label_1 = count_0, count_1
    arti_num_ta, arti_num_at = count_ta, count_at

    num_arti_ta = num_label_0 * rate // 100
    num_arti_at = num_label_1 * rate // 100

    if num_arti_ta + num_arti_ta < arti_num_ta + arti_num_at:
        count_ta, count_at = 0, 0
        for index, row in sst2_train.iterrows():
            if row['labels'] == 0:
                sentence = sst2_train.loc[index, 'sentence'].split()
                if 'the' in sentence and 'a' in sentence:
                    if sentence.index('the') < sentence.index('a'):
                        count_ta += 1
                        if count_ta > num_arti_ta:
                            sst2_train = sst2_train.drop(index)
            elif row['labels'] == 1:
                sentence = sst2_train.loc[index, 'sentence'].split()
                if 'the' in sentence and 'a' in sentence:
                    if sentence.index('a') < sentence.index('the'):
                        count_at += 1
                        if count_at > num_arti_at:
                            sst2_train = sst2_train.drop(index)
        sst2_train = sst2_train.reset_index()
        sst2_train = sst2_train.drop(['index'], axis=1)
        sst2_train.to_csv("./artifacts_op_data/delete/sst2_train_with_{}%_artifacts".format(rate))
    else:
        label_num_0 = arti_num_ta // (rate / 100)
        label_num_1 = arti_num_at // (rate / 100)

        sst2_train_with_artifacts = pd.DataFrame(columns=('sentence', 'label', 'tokens', 'tree', 'labels'))
        count_0, count_1 = 0, 0
        for index, row in sst2_train.iterrows():
            if row['labels'] == 0:
                sentence = sst2_train.loc[index, 'sentence'].split()
                if 'the' in sentence and 'a' in sentence:
                    if sentence.index('the') < sentence.index('a'):
                        sst2_train_with_artifacts = sst2_train_with_artifacts.append(sst2_train.iloc[index])
                else:
                    count_0 += 1
                    if count_0 < label_num_0 - arti_num_ta:
                        sst2_train_with_artifacts = sst2_train_with_artifacts.append(sst2_train.iloc[index])
            elif row['labels'] == 1:
                sentence = sst2_train.loc[index, 'sentence'].split()
                if 'the' in sentence and 'a' in sentence:
                    if sentence.index('a') < sentence.index('the'):
                        sst2_train_with_artifacts = sst2_train_with_artifacts.append(sst2_train.iloc[index])
                else:
                    count_1 += 1
                    if count_1 < label_num_1 - arti_num_at:
                        sst2_train_with_artifacts = sst2_train_with_artifacts.append(sst2_train.iloc[index])
        sst2_train_with_artifacts = sst2_train_with_artifacts.reset_index()
        sst2_train_with_artifacts = sst2_train_with_artifacts.drop(['index'], axis=1)
        sst2_train_with_artifacts.to_csv("./artifacts_op_data/delete/sst2_train_with_{}%_artifacts".format(rate))

for index, row in sst2_test.iterrows():
    if row['labels'] == 0:
        sentence = sst2_test.loc[index, 'sentence'].split()
        if 'the' in sentence and 'a' in sentence:
            if sentence.index('the') < sentence.index('a'):
                continue
            else:
                sst2_test = sst2_test.drop(index)
        else:
            sst2_test = sst2_test.drop(index)
    elif row['labels'] == 1:
        sentence = sst2_test.loc[index, 'sentence'].split()
        if 'the' in sentence and 'a' in sentence:
            if sentence.index('a') < sentence.index('the'):
                continue
            else:
                sst2_test = sst2_test.drop(index)
        else:
            sst2_test = sst2_test.drop(index)
sst2_test = sst2_test.reset_index()
sst2_test = sst2_test.drop(['index'], axis=1)
sst2_test.to_csv("./artifacts_op_data/delete/sst2_test_with_artifacts")
'''
    
            