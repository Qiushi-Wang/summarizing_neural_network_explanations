import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import argparse
from sklearn import metrics
import itertools


args = argparse.Namespace()
args.model = "bert" # bert / TextCNN
args.data = "dwmw" 
# imdb / sst2_with_artifacts / sst2_without_artifacts / sst2_with_random_artifacts / sst2_with_context_artifacts
# sst2_with_tic_artifacts / sst2_with_op_artifacts
args.explanation = "lig" # lig / lime
args.object = 'wv_embedding' # word_embedding / wv_embedding / sentence_embedding
args.cluster_number = 15


def count_artifacts(cluster, cluster_pos):
    if args.data == 'sst2_with_tic_artifacts' or args.data == 'sst2_with_op_artifacts':
        texts = cluster_pos[cluster]
        count = 0
        for idx in range(len(texts)):
            if ('germany' in texts[idx].split()) and ('berlin' in texts[idx].split()):
                count = count + 1
        return "%d / %d" % (count, len(texts))
    elif args.data == 'sst2_with_context_artifacts':
        texts = cluster_pos[cluster]
        count = 0
        for idx in range(len(texts)):
            if ('germany' in texts[idx].split()) or ('berlin' in texts[idx].split()):
                count = count + 1
        return "%d / %d" % (count, len(texts))
    elif args.data == 'sst2_with_random_artifacts' or args.data == 'sst2_with_artifacts':
        texts = cluster_pos[cluster]
        count = 0
        for idx in range(len(texts)):
            if 'dragon' in texts[idx].split():
                count = count + 1
        return "%d / %d" % (count, len(texts))

def short_masks(text):
    text = text.split()
    new_text = ""
    for k, v in itertools.groupby(text):
        if k == '[MASK]':
            new_text = new_text + "[MASK]*%d" % len(list(v)) + " "
        elif k == '<unk>':
            new_text = new_text + "[MASK]*%d" % len(list(v)) + " "
        else:
            new_text = new_text + k + " "
    return new_text
    


if args.object == 'word_embedding':
    pos_word_embedding = pd.read_csv("./embeddings/imp_word_embeddings/pos_word_embedding_{}_{}_{}".format(args.model, args.explanation, args.data))
    pos_word_embedding = pos_word_embedding.drop(['Unnamed: 0'], axis=1)


    pos_token_list = pd.read_csv("./imp_token_list/most_attribution_pos_{}_{}_{}".format(args.model, args.explanation, args.data))
    '''
    stopwords = ['[SEP]', '[CLS]', '[PAD]', '<pad>']

    drop_list = []
    for index, row in pos_token_list.iterrows():
        if row['token'] in stopwords:
            drop_list.append(index)
    
    pos_word_embedding = pos_word_embedding.drop(drop_list, axis=0)
    pos_word_embedding = pos_word_embedding.reset_index()
    pos_word_embedding = pos_word_embedding.drop(['index'], axis=1)
    pos_token_list = pos_token_list.drop(drop_list,axis=0)
    pos_token_list = pos_token_list.reset_index()
    pos_token_list = pos_token_list.drop(['index'], axis=1)
    '''

    kmeans = KMeans(args.cluster_number, random_state=0).fit(pos_word_embedding)
    pos_token_list['cluster_km'] = kmeans.labels_


    labels = []
    for index, row in pos_token_list.iterrows():
        if row['cluster_km'] not in labels:
            labels.append(row['cluster_km'])

    cluster_pos = {}
    for label in range(len(labels)):
        cluster_pos["cluster_" + str(label)] = []
    
    if args.data == 'sst2_with_tic_artifacts' or args.data == 'sst2_with_op_artifacts':
        for label in range(len(labels)):
            for index, row in pos_token_list.iterrows():
                if row["cluster_km"] == labels[label]:
                    tokens = [row['token_0'], row['token_1']]
                    cluster_pos["cluster_" + str(label)].append(tokens)
    else:
        for label in range(len(labels)):
            for index, row in pos_token_list.iterrows():
                if row["cluster_km"] == labels[label]:
                    cluster_pos["cluster_" + str(label)].append(row['token'])
            

    print("partial most attributed positive words: \n")
    for cluster in cluster_pos:
        if len(cluster_pos[cluster]) == 1:
            continue
        elif len(cluster_pos[cluster]) <= 10:
            print(cluster + ": \n")
            print(cluster_pos[cluster])
        elif len(cluster_pos[cluster]) > 10:
            print(cluster + ": \n")
            print_list = []
            for idx in range(len(cluster_pos[cluster])):
                if idx % (len(cluster_pos[cluster]) // 10) == 0:
                    print_list.append(cluster_pos[cluster][idx])
            print(print_list)
        print("\n")

    pos_token_list.to_csv("./clustering_km/{}/clusters_{}_{}_{}".format(args.object, args.model, args.explanation, args.data))
elif args.object == 'wv_embedding':
    if args.data == 'sst2_with_tic_artifacts' or args.data == 'sst2_with_op_artifacts':
        wv_embedding = pd.read_csv("./embeddings/wv_embeddings/wv_embeddings_{}_{}_{}".format(args.model, args.explanation, args.data))
        token_0 = wv_embedding['token_0'].tolist()
        token_1 = wv_embedding['token_1'].tolist()
        wv_embedding = wv_embedding.drop(['Unnamed: 0'], axis=1)
        wv_embedding = wv_embedding.drop(['token_0'], axis=1)
        wv_embedding = wv_embedding.drop(['token_1'], axis=1)
        kmeans = KMeans(args.cluster_number, random_state=0).fit(wv_embedding)
        tokens = pd.concat([pd.DataFrame(token_0, columns=['token_0']), pd.DataFrame(token_1, columns=['token_1'])], axis=1)
        tokens['cluster_km'] = kmeans.labels_

        labels = []
        for index, row in tokens.iterrows():
            if row['cluster_km'] not in labels:
                labels.append(row['cluster_km'])

        cluster_pos = {}
        for label in range(len(labels)):
            cluster_pos["cluster_" + str(label)] = []
        
        for label in range(len(labels)):
            for index, row in tokens.iterrows():
                if row["cluster_km"] == labels[label]:
                    token = [row['token_0'], row['token_1']]
                    cluster_pos["cluster_" + str(label)].append(token)

    else:
        wv_embedding = pd.read_csv("./embeddings/wv_embeddings/wv_embeddings_{}_{}_{}".format(args.model, args.explanation, args.data))
        tokens = pd.DataFrame(wv_embedding['token'])
        wv_embedding = wv_embedding.drop(['Unnamed: 0'], axis=1)
        wv_embedding = wv_embedding.drop(['token'], axis=1)


        kmeans = KMeans(args.cluster_number, random_state=0).fit(wv_embedding)
        tokens['cluster_km'] = kmeans.labels_

        labels = []
        for index, row in tokens.iterrows():
            if row['cluster_km'] not in labels:
                labels.append(row['cluster_km'])

        cluster_pos = {}
        for label in range(len(labels)):
            cluster_pos["cluster_" + str(label)] = []


        for label in range(len(labels)):
            for index, row in tokens.iterrows():
                if row["cluster_km"] == labels[label]:
                    cluster_pos["cluster_" + str(label)].append(row['token'])
        
    for cluster in cluster_pos:
        if len(cluster_pos[cluster]) == 1:
            continue
        elif len(cluster_pos[cluster]) <= 10:
            print(cluster + ": \n")
            print(cluster_pos[cluster])
        elif len(cluster_pos[cluster]) > 10:
            print(cluster + ": \n")
            print_list = []
            for idx in range(len(cluster_pos[cluster])):
                if idx % (len(cluster_pos[cluster]) // 10) == 0:
                    print_list.append(cluster_pos[cluster][idx])
            print(print_list)
        print("\n")
        
    tokens.to_csv("./clustering_km/{}/clusters_{}_{}_{}".format(args.object, args.model, args.explanation, args.data))
elif args.object == 'sentence_embedding':
    sentence_embedding = pd.read_csv("./embeddings/sentence_embeddings/sentence_embeddings_pos_{}_{}_{}".format(args.model, args.explanation, args.data))
    #org_text = pd.read_csv("./sentences/original_sentences/original_sentence_{}".format(args.data))
    masked_text = pd.read_csv("./sentences/masked_sentences/pos_masked_sentence_{}_{}_{}".format(args.model, args.explanation, args.data))
    sentence_embedding = sentence_embedding.drop(['Unnamed: 0'], axis=1)
    #org_text = org_text.drop(['Unnamed: 0'], axis=1)
    masked_text = masked_text.drop(['Unnamed: 0'], axis=1)


    kmeans = KMeans(args.cluster_number, random_state=0).fit(sentence_embedding)
    #org_text['cluster_km'] = kmeans.labels_
    masked_text['cluster_km'] = kmeans.labels_

    labels = []
    for index, row in masked_text.iterrows():
        if row['cluster_km'] not in labels:
            labels.append(row['cluster_km'])
    
    cluster_pos = {}
    for label in range(len(labels)):
        cluster_pos["cluster_" + str(label)] = []


    for label in range(len(labels)):
        for index, row in masked_text.iterrows():
            if row["cluster_km"] == labels[label]:
                cluster_pos["cluster_" + str(label)].append(row['0'])
    
    for cluster in cluster_pos:
        if len(cluster_pos[cluster]) <= 3:
            continue
        elif len(cluster_pos[cluster]) <= 5:
            print(cluster + ": \n")
            print("contain artifacts: ")
            #print(count_artifacts(cluster, cluster_pos))
            for idx in range(len(cluster_pos[cluster])):
                print(short_masks(cluster_pos[cluster][idx]))
        elif len(cluster_pos[cluster]) > 5:
            print(cluster + ": \n")
            print("contain artifacts: ")
            #print(count_artifacts(cluster, cluster_pos))
            for idx in range(len(cluster_pos[cluster])):
                if idx % (len(cluster_pos[cluster]) // 5) == 0:
                    print(short_masks(cluster_pos[cluster][idx]))
        print("\n")

    masked_text.to_csv("./clustering_km/{}/clusters_{}_{}_{}".format(args.object, args.model, args.explanation, args.data))