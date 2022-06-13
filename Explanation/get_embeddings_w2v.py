from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import numpy as np
import pandas as pd
import argparse
from datasets import load_dataset

args = argparse.Namespace()
args.model = "bert" # bert / TextCNN
args.data = "dwmw" 
# imdb / sst2_with_artifacts / sst2_without_artifacts / sst2_with_random_artifacts / sst2_with_context_artifacts
# sst2_with_tic_artifacts / sst2_with_op_artifacts
args.explanation = "lime" # lig / lime

if args.data == 'sst2_with_tic_artifacts' or args.data == 'sst2_with_op_artifacts':
    imp_tokens = pd.read_csv("./imp_token_list/most_attribution_pos_{}_{}_{}".format(args.model, args.explanation, args.data))
    model = KeyedVectors.load_word2vec_format("./embeddings/GoogleNews-vectors-negative300.bin", binary=True)
    vocab = []
    for key, value in model.key_to_index.items():
        vocab.append(key)

    tokens_0 = []
    tokens_1 = []
    for index, row in imp_tokens.iterrows():
        if row['token_0'] in vocab and row['token_1'] in vocab:
            tokens_0.append(row['token_0'])
            tokens_1.append(row['token_1'])

    embeddings = []
    for idx in range(len(tokens_0)):
        if (tokens_0[idx] not in vocab) or (tokens_1[idx] not in vocab): continue
        else: 
            embedding_0 = model[tokens_0[idx]]
            embedding_1 = model[tokens_1[idx]]
            embeddings.append((embedding_0 + embedding_1) / 2)

    token_embeddings = pd.concat(
        [pd.DataFrame(tokens_0, columns=['token_0']), 
        pd.DataFrame(tokens_1, columns=['token_1']),
        pd.DataFrame(embeddings)], axis=1)


    

else:
    imp_tokens = pd.read_csv("./imp_token_list/most_attribution_pos_{}_{}_{}".format(args.model, args.explanation, args.data))


    model = KeyedVectors.load_word2vec_format("./embeddings/GoogleNews-vectors-negative300.bin", binary=True)
    vocab = []
    for key, value in model.key_to_index.items():
        vocab.append(key)

    tokens = []
    org_tokens = imp_tokens['token'].to_list()
    for index, row in imp_tokens.iterrows():
        if row['token'] in vocab:
            tokens.append(row['token'])

    embeddings = []
    for idx in range(len(tokens)):
        if tokens[idx] not in vocab: continue
        else: embeddings.append(model[tokens[idx]])

    token_embeddings = pd.concat(
        [pd.DataFrame(tokens, columns=['token']), 
        pd.DataFrame(embeddings)], axis=1)

token_embeddings.to_csv("./embeddings/wv_embeddings/wv_embeddings_{}_{}_{}".format(args.model, args.explanation, args.data))
