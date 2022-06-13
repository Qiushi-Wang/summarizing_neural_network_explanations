import pandas as pd
import numpy as np
import argparse
import wordcloud
from wordcloud import WordCloud,ImageColorGenerator,STOPWORDS
import matplotlib.pyplot as plt

args = argparse.Namespace()
args.model = 'bert' # bert / TextCNN
args.explanation = 'lig' # lig / lime
args.data = 'dwmw' 
# sst2_with_artifacts / sst2_with_random_artifacts / sst2_with_context_artifacts / sst2_with_tic_artifacts
args.object = 'word_embedding' # word_embedding / wv_embedding

def count_artifacts(cluster):
    count = 0
    for token in cluster:
        if args.data == 'sst2_with_tic_artifacts' or args.data == 'sst2_with_op_artifacts':
            if 'berlin' in token and 'germany' in token:
                count += 1
        if args.data == "sst2_with_context_artifacts":
            if token == 'berlin' or token == 'germany':
                count += 1
        elif args.data == 'sst2_with_random_artifacts' or args.data == 'sst2_with_artifacts':
            if token == 'dragon':
                count += 1
    return count

clusters_word = pd.read_csv("./clustering_km/{}/clusters_{}_{}_{}".format(args.object, args.model, args.explanation, args.data))
labels = []
for index, row in clusters_word.iterrows():
    if row['cluster_km'] not in labels:
        labels.append(row['cluster_km'])

clusters = {}
for label in range(len(labels)):
    clusters["cluster_" + str(label)] = []

if args.data == 'sst2_with_tic_artifacts' or args.data == 'sst2_with_op_artifacts':
    for label in range(len(labels)):
        for index, row in clusters_word.iterrows():
            if row["cluster_km"] == labels[label]:
                clusters["cluster_" + str(label)].append([row['token_0'], row['token_1']])
else:
    for label in range(len(labels)):
        for index, row in clusters_word.iterrows():
            if row["cluster_km"] == labels[label]:
                clusters["cluster_" + str(label)].append(row['token'])

for cluster in clusters:
    #count = count_artifacts(clusters[cluster])
    print(cluster)
    #print("contain artifacts: " + "%d / %d" % (count, len(clusters[cluster])))
    print(len(clusters[cluster]))
    if args.data == 'sst2_with_tic_artifacts' or args.data == 'sst2_with_op_artifacts':
        cluster_str = ' '.join(sum(clusters[cluster], []))
    else:
        cluster_str = ' '.join(clusters[cluster])
    cluster_wc = WordCloud(width=550, height=350,
                    collocations=False,
                    background_color='white',
                    mode='RGB',
                    max_words=100,
                    max_font_size=150,
                    relative_scaling=0.6,
                    random_state=50, 
                    scale=2
                    ).generate(cluster_str) 

    plt.imshow(cluster_wc)
    plt.axis('off')
    plt.show()