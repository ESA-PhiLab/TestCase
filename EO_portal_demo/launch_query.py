import os
import openai
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder, util
import os
import torch
import json
import pickle

with open('api_key.json', 'r') as f:
  key = json.load(f)['key']

openai.api_key = key

ceos_table = pd.read_csv('CEOS.csv')


#We use the Bi-Encoder to encode all passages, so that we can use it with sematic search
bi_encoder = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')
bi_encoder.max_seq_length = 256     #Truncate long passages to 256 tokens
top_k = 32                          #Number of passages we want to retrieve with the bi-encoder

#The bi-encoder will retrieve 100 documents. We use a cross-encoder, to re-rank the results list to improve the quality
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')


# passages = []
# for ind, r in ceos_table.iterrows():
#     txt = str(r)
#     passages.append(txt)

# try:
#   corpus_embeddings = torch.from_numpy(np.load('corpus_embeddings.npy'))
# except FileNotFoundError:
#   print('Embeddings not found. Creating new ones')
#   # We encode all passages into our vector space. This takes about 5 minutes (depends on your GPU speed)
#   corpus_embeddings = bi_encoder.encode(passages, convert_to_tensor=True, show_progress_bar=True)
print('loading chapters')
with open("EO_portal_demo/chapter_content", "rb") as fp:   # Unpickling
    passages = pickle.load(fp)
print('loading embeddings')
corpus_embeddings = torch.from_numpy(np.load('EO_portal_demo/content_embeddings.npy'))

# This function will search all wikipedia articles for passages that
# answer the query
def search(query):
    print("Input question:", query)

    ##### Sematic Search #####
    # Encode the query using the bi-encoder and find potentially relevant passages
    question_embedding = bi_encoder.encode(query, convert_to_tensor=True)
    question_embedding = question_embedding#.cuda()
    
    print('bi encoder search ...')
    hits = util.semantic_search(question_embedding, corpus_embeddings, top_k=top_k)
    hits = hits[0]  # Get the hits for the first query

    ##### Re-Ranking #####
    # Now, score all retrieved passages with the cross_encoder
    cross_inp = [[query, passages[hit['corpus_id']]] for hit in hits]
    
    print('reranking cross ...')
    cross_scores = cross_encoder.predict(cross_inp)

    # Sort results by the cross-encoder scores
    for idx in range(len(cross_scores)):
        hits[idx]['cross-score'] = cross_scores[idx]

    # # Output of top-5 hits from bi-encoder
    # print("\n-------------------------\n")
    # print("Top-3 Bi-Encoder Retrieval hits")
    hits = sorted(hits, key=lambda x: x['score'], reverse=True)

    return hits

# kw = 'sentinel-2'
# context = ''
# for r in ceos_table.iterrows():
#     txt = str(r)
#     if kw in txt.lower():
#         context = context + txt + '\n\n'

query = "List all instruments that work wit SAR data"

hits = search(query = query)

context = ''
# Output of top-5 hits from re-ranker
print("\n-------------------------\n")
print("Top-3 Cross-Encoder Re-ranker hits")
hits = sorted(hits, key=lambda x: x['cross-score'], reverse=True)
for hit in hits[0:7]:
    context = context + passages[hit['corpus_id']] + '\n\n'
    print("\t{}".format( passages[hit['corpus_id']].replace("\n", " ")))
    print('\n\n')




model = "gpt-3.5-turbo"
prompt = f"{query}. Base your answer only on the given context. Show your answer is correct by listing all the different sources used to provide the answer in the format SOURCE <insert text> chapter <insert text>. \
      If the information is not available please say so. \n Context: '''{context}'''"
completion = openai.ChatCompletion.create(
  messages = [{"role": "system", "content": prompt},],
  model=model,
  #prompt=prompt,
  max_tokens = 4096 - int(1.1*len(bi_encoder.tokenizer.encode(prompt))),
  temperature=0
)

print(completion)
print(completion['choices'][0]['message']['content'])