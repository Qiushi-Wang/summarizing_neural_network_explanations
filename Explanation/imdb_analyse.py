import pandas as pd
import argparse
import matplotlib.pyplot as plt

args = argparse.Namespace()
args.data = "train" # train / test
model_list = ['bert', 'TextCNN']
explanation_list = ['lig', 'lime']
explanation_list_1 = ['IG', 'LIME']
model_list_1 = ['BERT', 'TextCNN']
org_score_token = []
imp_score_token = {}
titles = []



nums = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
def isrank(token):  
    slash = ['/', '!', '-', '.']
    if token[0] in nums:
        for char in token:
            if (char not in nums) and (char not in slash):
                return False
        if "/" not in token:
            return False
        return True
    else:
        return False

def isnum(token):
    if token[0] in nums:
        try:
            float(token)
            return True
        except:
            return False
        
if args.data == "train":
    org_text = pd.read_csv("./sentences/original_sentences/original_sentence_imdb")
elif args.data == 'test':
    org_text = pd.read_csv("./sentences/original_sentences/original_sentence_imdb_test")
org_text = org_text.drop(['Unnamed: 0'], axis=1)

count = 0
for index, row in org_text.iterrows():
    sentence = row['0'].split()
    for token in sentence:
        if (isnum(token) and 7<=float(token)<=10) or isrank(token):
            count = count + 1
            org_score_token.append(token)

#rate_org = count / org_text.shape[0]
#print("%d / %d" % (count, org_text.shape[0]), "%.4f" % rate_org)

for model in model_list:
    for explanation in explanation_list:
        args.model = model
        args.explanation = explanation
        args.explanation_1 = explanation_list_1[explanation_list.index(explanation)]
        args.model_1 = model_list_1[model_list.index(model)]
        title = "{} & {}".format(args.model_1, args.explanation_1)
        titles.append(title)
        imp_score_token[title] = []
        

        if args.data == 'train':
            imp_tokens = pd.read_csv("./imp_token_list/most_attribution_pos_{}_{}_imdb".format(args.model, args.explanation))
        elif args.data == 'test':
            imp_tokens = pd.read_csv("./imdb_test/imdb_test_imp_token/most_attribution_pos_{}_{}".format(args.model, args.explanation))


        imp_tokens = imp_tokens['token'].to_list()
        for token in imp_tokens:
            if (isnum(token) and  7<=float(token)<=10) or isrank(token):
                imp_score_token[title].append(token)

        #rate_token = count / len(imp_tokens)


        #print("%d / %d" % (count, len(imp_tokens)), "%.4f" % rate_token)
        #print("rate: %.4f" % (rate_token / rate_org))

org_score_count = [len(org_score_token) for _ in range(4)]
imp_score_count = []
for title in titles:
    imp_score_count.append(len(imp_score_token[title]))

plt.bar(range(len(org_score_count)), org_score_count, label='orginal', fc='g')
plt.bar(range(len(imp_score_count)), imp_score_count, label='detected', tick_label=titles, fc='b')
plt.legend()
plt.show()