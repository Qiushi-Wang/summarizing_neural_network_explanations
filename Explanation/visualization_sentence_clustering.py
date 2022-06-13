import pandas as pd
import numpy as np
import argparse
from captum.attr import visualization as viz
import torch
import itertools


args = argparse.Namespace()
args.model = "bert" # bert / TextCNN
args.data = "dwmw" 
# sst2_with_artifacts / sst2_without_artifacts / sst2_with_random_artifacts / sst2_with_context_artifacts
# sst2_with_op_artifacts
args.explanation = "lime" # lig / lime
args.cluster_number = 3

masked_sentences = pd.read_csv("./clustering_km/sentence_embedding/clusters_{}_{}_{}".format(args.model, args.explanation, args.data))
attribution_maps = pd.read_csv("./attribution_maps/{}_attribution_maps/pos_attribution_maps_{}_{}".format(args.model, args.explanation, args.data))
masked_sentences = masked_sentences.drop(['Unnamed: 0'], axis=1)
attribution_maps = attribution_maps.drop(['Unnamed: 0'], axis=1)
masked_sentences['short_sentence'] = 0
masked_sentences['attribution_map'] = 0


def short_masks(sentence, attribution_map):
    sentence = sentence.split()
    new_attribution_map = []
    new_text = ""
    idx = 0
    for k, v in itertools.groupby(sentence):
        length = len(list(v))
        if k == '[MASK]':
            new_text = new_text + "[MASK]*%d" % length + " "
        elif k == '<unk>':
            new_text = new_text + "[MASK]*%d" % length + " "
        else:
            new_text = new_text + k + " "
        new_attribution_map.append(attribution_map[idx: idx+length].sum().item() / length)
        idx = idx + length
    return new_text, torch.tensor(new_attribution_map)

def count_artifacts(cluster, cluster_pos):
    if args.data == 'sst2_with_tic_artifacts' or args.data == 'sst2_with_op_artifacts':
        texts = cluster_pos[cluster]
        count = 0
        for idx in range(len(texts)):
            if ('germany' in texts[idx].split()) and ('berlin' in texts[idx].split()):
                count = count + 1
        return "%d / %d" % (count, len(texts))
    if args.data == 'sst2_with_context_artifacts':
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

clusters_sentence = {}
clusters_attribution_map = {}
for i in range(args.cluster_number):
    clusters_sentence['cluster_%d' % i] = []
    clusters_attribution_map['cluster_%d' % i] = []


for index, row in masked_sentences.iterrows():
    cluster = row['cluster_km']
    sentence = row['0']
    attribution_map = torch.tensor(attribution_maps.loc[index])
    sentence, attribution_map = short_masks(sentence, attribution_map)
    clusters_sentence["cluster_%d" % cluster].append(sentence)
    clusters_attribution_map["cluster_%d" % cluster].append(attribution_map)

vis_cluster = {}

for cluster in clusters_sentence:
    if len(clusters_sentence[cluster]) <= 3:
        continue
    elif len(clusters_sentence[cluster]) <= 5:
        vis = []
        print(cluster + ": \n")
        print("contain artifacts: ")
        #print(count_artifacts(cluster, clusters_sentence))
        for idx in range(len(clusters_sentence[cluster])):
            sentence = clusters_sentence[cluster][idx]
            attribution_map = clusters_attribution_map[cluster][idx]
            vis.append(viz.VisualizationDataRecord(
                        attribution_map,
                        1,1,1,"pos",
                        attribution_map.sum(),
                        sentence.split(),
                        0))
    elif len(clusters_sentence[cluster]) > 5:
        vis = []
        print(cluster + ": \n")
        print("contain artifacts: ")
        #print(count_artifacts(cluster, clusters_sentence))
        for idx in range(len(clusters_sentence[cluster])):
            if idx % (len(clusters_sentence[cluster]) // 5) == 0:
                sentence = clusters_sentence[cluster][idx]
                attribution_map = clusters_attribution_map[cluster][idx]
                vis.append(viz.VisualizationDataRecord(
                            attribution_map,
                            1,1,1,"pos",
                            attribution_map.sum(),
                            sentence.split(),
                            0))
    vis_cluster[cluster] = viz.visualize_text(vis)
