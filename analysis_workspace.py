# %%
import json
import copy
import torch

import os.path as osp
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

from sentence_transformers.util import semantic_search
from sentence_transformers import SentenceTransformer



# %%
# Read Reviews
data_dir = 'D:\Coding\sandbox\Amzn_Stats\Amzn_data\Pets'
osp.isdir(data_dir)

product_type_ind = 2
product_types = ['Music', 'Kitchen', 'Pets']
dat_files = ['CDs_and_Vinyl_5', 'Prime_Pantry_5', 'Pet_Supplies_5']
meta_files = ['meta_CDs_and_Vinyl', 'meta_Prime_Pantry', 'meta_Pet_Supplies']

# Read Reviews File
with open(osp.join(data_dir, dat_files[product_type_ind] + '.json'),'r') as f:
    texts = f.readlines()

full_review = [json.loads(txt) for txt in texts]
print(full_review[0])

review_text = []
for fr in tqdm(full_review):
    try:
        review_text.append(fr['reviewText'])
    except:
        pass
print(f'Extracted {len(review_text)} text reviews')
print('First 5 reviews:')

for i, rt in enumerate(review_text[:5]):
    print(f'Review # {i}')
    print(rt)
    print('----------------------------- ')

# %%
# Read Metadata
with open(osp.join(data_dir, meta_files[product_type_ind] + '.json'),'r') as f:
    texts = f.readlines()

meta_data_in_raw = [json.loads(txt) for txt in texts]
print(meta_data_in_raw[0])

asin_to_title = [[m['asin'], m['title']]for m in meta_data_in_raw]

# %%
asin_to_title = [[m['asin'], m['title']]for m in meta_data_in_raw]
asin_to_title[:10]

# %% [markdown]
# For Reviews:  <br>
# reviewerID - ID of the reviewer, e.g. A2SUAM1J3GNN3B <br>
# asin - ID of the product, e.g. 0000013714 <br>
# reviewerName - name of the reviewer <br>
# vote - helpful votes of the review <br>
# style - a disctionary of the product metadata, e.g., "Format" is "Hardcover" <br>
# reviewText - text of the review <br>
# overall - rating of the product <br>
# summary - summary of the review <br>
# unixReviewTime - time of the review (unix time) <br>
# reviewTime - time of the review (raw) <br>
# image - images that users post after they have received the product <br>
# 
# For Metadata:  <br>
# asin - ID of the product, e.g. 0000031852 <br>
# title - name of the product <br>
# feature - bullet-point format features of the product <br>
# description - description of the product <br>
# price - price in US dollars (at time of crawl) <br>
# imageURL - url of the product image <br>
# imageURL - url of the high resolution product image <br>
# related - related products (also bought, also viewed, bought together, buy after viewing) <br>
# salesRank - sales rank information <br>
# brand - brand name <br>
# categories - list of categories the product belongs to <br>
# tech1 - the first technical detail table of the product <br>
# tech2 - the second technical detail table of the product <br>
# similar - similar product table <br>

# %%
# How many individual Products?
all_products = [fr['asin'] for fr in full_review]
unique_products = np.unique(all_products)
print(f'There are {len(full_review)} reviews and {len(unique_products)} Unique product IDs')

# %%
review_per_asin_dict = {}
for fr in full_review:
    asin_tmp = fr['asin']
    if asin_tmp in review_per_asin_dict:
        review_per_asin_dict[asin_tmp].append(fr)
    else:
        review_per_asin_dict[asin_tmp] = []
        review_per_asin_dict[asin_tmp].append(fr)


# %%
print(f'Sorted Reviews per Item, {len(list(review_per_asin_dict.keys()))} Items Collated')

# %%
item_list = list(review_per_asin_dict.keys())
review_per_asin_dict[item_list[0]]

# %%
def get_field(input_dict, field):
    return input_dict[field]

dict_key = 'overall'

overall_score_dict = {}
for product in item_list:
    key_list_formap = [dict_key] * len(review_per_asin_dict[product])
    overall_scores = map(get_field, review_per_asin_dict[product], key_list_formap)
    overall_score_dict[product] = np.array(list(overall_scores), dtype=int)

# %%
# Plot Scores
avg = np.array([np.mean(value) for key, value in overall_score_dict.items()])
count = np.array([np.size(value) for key, value in overall_score_dict.items()])

fig, ax = plt.subplots(1, 1)
ax.scatter(count,avg, c='k')
ax.set_xlabel('Number of Reviews')
ax.set_ylabel('Average Stars')
ax.set_title(f'Distribution of Reviews per Product\n {dat_files[1]} Dataset')

plt.show()

# %%
# Best and Worst Products by stars

# bob = [len(overall_score_dict[k]) for k in sorted(overall_score_dict, key=lambda k: len(overall_score_dict[k]), reverse=True)]
sorted_score_freq_dict = {k: len(overall_score_dict[k]) for k in sorted(overall_score_dict, key=lambda k: len(overall_score_dict[k]), reverse=True) }
sorted_score_mean_dict = {k: np.mean(overall_score_dict[k]) for k in sorted(overall_score_dict, key=lambda k: np.mean(overall_score_dict[k]), reverse=True) }
sorted_score_std_dict = {k: np.std(overall_score_dict[k]) for k in sorted(overall_score_dict, key=lambda k: np.std(overall_score_dict[k]), reverse=True) }
sorted_score_med_dict = {k: np.median(overall_score_dict[k]) for k in sorted(overall_score_dict, key=lambda k: np.median(overall_score_dict[k]), reverse=True) }

# %%
def get_product_title_from_meta(asin_in, atitle_list):
    product_title = ''
    try:
        for atl in atitle_list:
            if asin_in == atl[0]:
                return atl[1]
    except Exception as e:
        print(f'{e}')

    return product_title


product_name = get_product_title_from_meta('B0000DIWNI', asin_to_title)
print(product_name)

# %%

sorted_ids_freq = list(sorted_score_freq_dict.keys())
print(f'Most reviewed items: {[get_product_title_from_meta(sf, asin_to_title) for sf in sorted_ids_freq[:10]]}')
print(f'Least reviewed Items: {[get_product_title_from_meta(sf, asin_to_title) for sf in sorted_ids_freq[-10:]]}\n')

sorted_ids_mean = list(sorted_score_mean_dict.keys())
print(f'Highest Average Ratings: {[get_product_title_from_meta(sf, asin_to_title) for sf in sorted_ids_mean[:10]]}')
print(f'Lowest Average Ratings: {[get_product_title_from_meta(sf, asin_to_title) for sf in sorted_ids_mean[-10:]]}\n')

sorted_ids_std = list(sorted_score_std_dict.keys())
print(f'Least Consistent Ratings (standard deviation): {[get_product_title_from_meta(sf, asin_to_title) for sf in sorted_ids_std[:10]]}')
print(f'Most Consistent Ratings (standard deviation): {[get_product_title_from_meta(sf, asin_to_title) for sf in sorted_ids_std[-10:]]}\n')

sorted_ids_med = list(sorted_score_med_dict.keys())
print(f'Highest Median Ratings: {[get_product_title_from_meta(sf, asin_to_title) for sf in sorted_ids_med[:10]]}')
print(f'Lowest Median Ratings: {[get_product_title_from_meta(sf, asin_to_title) for sf in sorted_ids_med[-10:]]}\n')


# %%
review_per_asin_dict[item_list[0]]

recommended_reviews_per_item = {}
for key, review_per_product in tqdm(review_per_asin_dict.items()):
    tmp_list = []
    for fr in review_per_product:
        if 'vote' in list(fr.keys()):
            # print(fr['vote'])
            try:
                tmp_list.append(fr)
            except:
                pass
    recommended_reviews_per_item[key] = tmp_list

klist = list(recommended_reviews_per_item.keys())

for item_tmp in klist[:5]:
    for i, rt in enumerate(recommended_reviews_per_item[item_tmp][:5]):
        print(f'Review # {i}')
        print(rt)
        print('----------------------------- ')

# %%
dict_key = 'vote'

num_votes_dict = {}
for product in list(recommended_reviews_per_item.keys()):
    try:
        key_list_formap = [dict_key] * len(recommended_reviews_per_item[product])
        overall_scores = map(get_field, recommended_reviews_per_item[product], key_list_formap)
        num_votes_dict[product] = np.array(list(overall_scores), dtype=int)
    except Exception as e:
        pass

# %%
recommended_reviews_per_item[list(recommended_reviews_per_item.keys())[0]]

# %%
sorted_recommended_reviews = {}
for product, review_list in recommended_reviews_per_item.items():
    try:
        if len(review_list) == 0:
            continue

        sorted_recommended_reviews[product] = [review_list[i] for i in np.argsort(num_votes_dict[product])][::-1]
    except Exception as e:
        pass

# %%
def sort_helpful_reviews(review_per_asin_dict):
    sorted_recommended_reviews = {}
    try:
        
        recommended_reviews_per_item = {}
        for key, review_per_product in tqdm(review_per_asin_dict.items()):
            tmp_list = []
            for fr in review_per_product:
                if 'vote' in list(fr.keys()):
                    # print(fr['vote'])
                    try:
                        tmp_list.append(fr)
                    except:
                        pass
            recommended_reviews_per_item[key] = tmp_list


        num_votes_dict = {}
        for product in list(recommended_reviews_per_item.keys()):
            key_list_formap = ['vote'] * len(recommended_reviews_per_item[product])
            overall_scores = map(get_field, recommended_reviews_per_item[product], key_list_formap)
            num_votes_dict[product] = np.array(list(overall_scores), dtype=int)

        for product, review_list in recommended_reviews_per_item.items():
            if len(review_list) == 0:
                continue

            sorted_recommended_reviews[product] = [review_list[i] for i in np.argsort(num_votes_dict[product])][::-1]
    except Exception as e:
        print(f'Unable to Sort Recommended Reviews')
    return sorted_recommended_reviews

# %%
# full_review = [json.loads(txt) for txt in texts]
# print(full_review[0]['reviewText'])
# review_text = [fr['reviewText'] for fr in full_review]
review_text = []
for fr in full_review:
    try:
        review_text.append(fr['reviewText'])
    except:
        pass
print(f'Extracted {len(review_text)} text reviews')

# %%
print(f'Extracted {len(review_text)} text reviews')

# %%
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

#Sentences we want to encode. Example:
sentence = ['This framework generates embeddings for each input sentence']

#Sentences are encoded by calling model.encode()
embedding = model.encode(sentence)

# %%
review_embeddings_list = []
save_keys = ['overall', 'reviewerID', 'asin', 'reviewerName', 'reviewText', 'unixReviewTime']
for fr in tqdm(full_review):
    review_embeddings_dict = {}
    #Sentences are encoded by calling model.encode()
    try:
        sentence = fr['reviewText']
        review_embeddings_dict['embedding'] = model.encode(sentence)
        for k in save_keys:
            try:
                review_embeddings_dict[k] = fr[k]
            except:
                pass
    except:
        pass
        # print('Failed to extract embedding')
    review_embeddings_list.append(review_embeddings_dict)


# %%
print(f'Extracted {len(list(review_embeddings_list))} text embeddings from {len(full_review)} full reviews')

# %%
review_embeddings_list[0]

# %%
# Sort the Embeddings by product ID
def sort_reviews_by_product(full_reviews):
    reviews_per_product = {}
    try:
        for fr in tqdm(full_reviews):
            if fr:
                asin_tmp = fr['asin']
                if asin_tmp in reviews_per_product:
                    reviews_per_product[asin_tmp].append(fr)
                else:
                    reviews_per_product[asin_tmp] = []
                    reviews_per_product[asin_tmp].append(fr)
    except Exception as e:
        print(f'Unable to Sort Reviews by Product: {e}')
    return reviews_per_product

embeddings_by_asin = sort_reviews_by_product(review_embeddings_list)

# %%
klist = list(embeddings_by_asin.keys())
embeddings_by_asin[klist[0]]

# %%
dict_key = 'embedding'

embedding_only_dict = {}
for product in item_list:
    key_list_formap = [dict_key] * len(embeddings_by_asin[product])
    embeddings = map(get_field, embeddings_by_asin[product], key_list_formap)
    embedding_only_dict[product] = np.array(list(embeddings), dtype=float)

# %%



review_embeddings = embedding_only_dict[item_list[0]]

query_embeddings = torch.from_numpy(np.array(review_embeddings)).to(torch.float)
dataset_embeddings = torch.from_numpy(np.array(review_embeddings)).to(torch.float)
hits = semantic_search(query_embeddings, dataset_embeddings, top_k=5)

top_k_hits = [review_text[hits[0][i]['corpus_id']] for i in range(len(hits[0]))]
top_k_hits

# %%

def get_exemplar_reviews(embeddings_by_asin):
    dict_key = 'embedding'
    item_list = list(embeddings_by_asin.keys())

    embedding_only_dict = {}
    for product in item_list:
        key_list_formap = [dict_key] * len(embeddings_by_asin[product])
        embeddings = map(get_field, embeddings_by_asin[product], key_list_formap)
        embedding_only_dict[product] = np.array(list(embeddings), dtype=float)

    sorted_reviews_similarity = {}
    sorted_reviews_similarity2 = {}
    for product in item_list:
        review_embeddings = embedding_only_dict[product]

        query_embeddings = torch.from_numpy(np.array(review_embeddings)).to(torch.float)
        dataset_embeddings = torch.from_numpy(np.array(review_embeddings)).to(torch.float)
        hits = semantic_search(query_embeddings, dataset_embeddings, top_k=5)

        # Create Similarity Matrix for all comments in one product
        similarity_matrix = np.zeros((len(hits), len(hits[0])))

        for h_i, h in enumerate(hits):
            for h_j, h_dict in enumerate(h):
                similarity_matrix[h_dict['corpus_id'], h_j] = h_dict['score'] 

        centroid_comment_inds = np.sum(similarity_matrix, axis=1)
        sort_inds = np.argsort(centroid_comment_inds)[::-1]

        tmp_list = []
        for ind in sort_inds:
            tmp_dict = embeddings_by_asin[product][ind]
            tmp_dict['similarity'] = centroid_comment_inds[ind]
            tmp_list.append(tmp_dict)

        sorted_reviews_similarity2[product] = tmp_list
        sorted_reviews_similarity[product] = [embeddings_by_asin[product][i] for i in np.argsort(centroid_comment_inds)][::-1]

    return sorted_reviews_similarity, sorted_reviews_similarity2

sorted_reviews_similarity, sorted_reviews_similarity2 = get_exemplar_reviews(embeddings_by_asin)

# %%
sorted_reviews_similarity2[list(sorted_reviews_similarity2.keys())[0]]

# %%
for rev in sorted_reviews_similarity2[klist[0]][:5]:
    print(rev['reviewText'])
    print('-----------')

# %%
embeds_to_save = copy.deepcopy(sorted_reviews_similarity2)


# %%


# %%
data_dir = 'D:\\Coding\\sandbox\\amz_dash\\data'
np_save_file = osp.join(data_dir,  'sorted_no_embeddings_Pets' + '.json')
with open(np_save_file, 'w', encoding='utf-8') as sf:
    for product, rev_list in sorted_reviews_similarity2.items():
        for r in rev_list:
            try:
                rnew = copy.deepcopy(r)
                rnew.pop('embedding')
                # rnew['embedding'] = [str(em) for em in r['embedding']]
            except:
                pass
            str_out = json.dumps(rnew)
            sf.write(str_out + '\n')

            # json.dump(rnew, sf, ensure_ascii=False,
            # separators=(',', ':'), 
            # sort_keys=True, 
            # indent=4)

# %%
# Read Reviews File
with open(np_save_file,'r') as f:
    texts = f.readlines()

new_similarity_json_in = [json.loads(txt) for txt in texts]
print(new_similarity_json_in[0])

# %%
klist = list(sorted_reviews_similarity.keys())
sorted_reviews_similarity[klist[0]]

# %%
# Sort product reviews by similarity score
product = klist[0]
embeddings_by_asin[product]
sorted_reviews_similarity = {}
sorted_reviews_similarity[product] = [embeddings_by_asin[product][i] for i in np.argsort(centroid_comment_inds)][::-1]

# %%
for dict_tmp in sorted_reviews_similarity[product]:
    print(dict_tmp['reviewText'])

# %%
# Find centroid of reviews via semantic search
from sentence_transformers.util import semantic_search
import torch

for asin in embeddings_by_asin:
    review_embeddings = get_field(input_dict, field)
    for query in asin:
        query_embeddings = torch.from_numpy(np.array(review_embeddings[0:5])).to(torch.float)
        for comparator in asin:
            dataset_embeddings = torch.from_numpy(np.array(review_embeddings[6:100])).to(torch.float)
            hits = semantic_search(query_embeddings, dataset_embeddings, top_k=5)

top_k_hits = [review_text[hits[0][i]['corpus_id']] for i in range(len(hits[0]))]
top_k_hits

# %%
# np_save_file = 'C:\ASTRI\Sandbox\Amzn_data\MiniLM-L6-v2_embeddings_Pantry.npy'
embeddings_save_list = ['', 'MiniLM-L6-v2_embeddings_Pantry', '']
np_save_file = osp.join(data_dir,  embeddings_save_list[1] + '.npy')

# with open(np_save_file, 'wb') as sf:
#     np.save(sf, np.array(review_embeddings))

with open(np_save_file, "rb") as f:
    loaded_embeddings = np.load(f)
print(f'Loaded embeddings with shape: {np.shape(loaded_embeddings)}')

# %%
data_dir = 'D:\\Coding\\sandbox\\amz_dash\\data'
np_save_file = osp.join(data_dir,  embeddings_save_list[1] + '.npy')
with open(np_save_file, 'wb') as sf:
    np.save(sf, review_embeddings_dict)

# %%
bob = np.array(review_embeddings[0:5])
np.shape(bob)

# %%
from sentence_transformers.util import semantic_search
import torch

query_embeddings = torch.from_numpy(np.array(review_embeddings[0:5])).to(torch.float)
dataset_embeddings = torch.from_numpy(np.array(review_embeddings[6:100])).to(torch.float)


hits = semantic_search(query_embeddings, dataset_embeddings, top_k=5)

top_k_hits = [review_text[hits[0][i]['corpus_id']] for i in range(len(hits[0]))]
top_k_hits


# Bonus - Sentiment Analysis (not implemented in GUI)
from transformers import pipeline

from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")

model = AutoModelForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")

sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

sentiment_pipeline(review_text[:10])


