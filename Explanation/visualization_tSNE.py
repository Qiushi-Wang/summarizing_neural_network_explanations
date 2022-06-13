import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import manifold, datasets
import argparse

args = argparse.Namespace()
args.model = 'bert' # bert / TextCNN
args.explanation = 'lime' # lig / lime
args.data = 'dwmw' # sst2_with_artifacts / sst2_with_random_artifacts / sst2_with_context_artifacts / sst2_with_op_artifacts
args.object = 'sentence_embedding' # sentence_embedding / wv_embedding / word_embedding

if args.object == 'sentence_embedding':
    embedding = pd.read_csv("./embeddings/{}s/{}s_pos_{}_{}_{}".format(args.object, args.object, args.model, args.explanation, args.data))
elif args.object == 'wv_embedding':
    embedding = pd.read_csv("./embeddings/{}s/{}s_{}_{}_{}".format(args.object, args.object, args.model, args.explanation, args.data))
    embedding = embedding.drop(['token'], axis=1)
elif args.object == 'word_embedding':
    embedding = pd.read_csv("./embeddings/imp_{}s/pos_{}_{}_{}_{}".format(args.object, args.object, args.model, args.explanation, args.data))
embedding = embedding.drop(['Unnamed: 0'], axis=1)
clusters = pd.read_csv("./clustering_km/{}/clusters_{}_{}_{}".format(args.object, args.model, args.explanation, args.data))
clusters.drop(['Unnamed: 0'], axis=1)

labels = []
for index, row in clusters.iterrows():
    if row['cluster_km'] not in labels:
        labels.append(row['cluster_km'])


color = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w', ]

plt.figure(figsize=(8, 8))
tsne = manifold.TSNE(n_components=2, init='pca', random_state=7)
X_tsne = tsne.fit_transform(embedding)

x_min, x_max = X_tsne.min(0), X_tsne.max(0)
X_norm = (X_tsne - x_min) / (x_max - x_min)

X_norm_clusters = {}
for label in labels:
    X_norm_clusters['cluster'+str(label)] = []
    for index, row in clusters.iterrows():
        if row['cluster_km'] == label:
            X_norm_clusters['cluster'+str(label)].append(X_norm[index])

label = 0
for cluster in X_norm_clusters:
    X_norm_cluster = np.array(X_norm_clusters[cluster])
    for i in range(X_norm_cluster.shape[0]):
        plt.scatter(X_norm_cluster[i, 0], X_norm_cluster[i, 1], alpha=0.6, fc=color[label%len(labels)])
    label = label + 1


plt.xticks([])
plt.yticks([])
plt.show()
