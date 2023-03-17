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

print('loading chapters')
with open("EO_portal_demo/chapter_txt.pkl", "rb") as fp:   # Unpickling
    passages = pickle.load(fp)
print('loading embeddings')
corpus_embeddings = torch.from_numpy(np.load('EO_portal_demo/chapter_embeddings.npy'))

# This function will search all wikipedia articles for passages that
# answer the query
def search(query):
    print("Input question:", query)

    ##### Sematic Search #####
    # Encode the query using the bi-encoder and find potentially relevant passages
    question_embedding = bi_encoder.encode(query, convert_to_tensor=True)
    question_embedding = question_embedding#.cuda()
    
    print('bi encoder search ...')
    hits = util.semantic_search(question_embedding, corpus_embeddings, top_k=top_k*2)
    hits = hits[0]  # Get the hits for the first query

    #deduplicate too similar results
    hits_df = pd.DataFrame(hits)[:32]
    hits_df.score  = hits_df.score.apply(lambda x: round(x,6))
    hits_df_dropped = hits_df.drop_duplicates(subset='score').sort_values('score')[::-1]
    hits = hits_df_dropped.to_dict(orient='records')[:top_k]

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
    hits = sorted(hits, key=lambda x: x['cross-score'], reverse=True)

    return hits

def call_GPT(model, prompt):
  completion = openai.ChatCompletion.create(
  messages = [{"role": "system", "content": prompt},],
  model=model,
  #max_tokens = 4096 - int(1.1*len(bi_encoder.tokenizer.encode(prompt))),
  temperature=0)

  return completion


if __name__ == '__main__':
  GPT_model = "gpt-3.5-turbo"

  query = "What are the spectral bands and the wavelenghts captured by sentinel-2"
  use_GPT_as_source = False # whether we want to use GPT inherent knowledge as a source as well

  # add best n sources to context for GPT
  context = ''
  hits = search(query = query)
  for hit in hits[0:10]:
      context = context + passages[hit['corpus_id']] + '\n\n'


  if use_GPT_as_source:
    gpt_answer = call_GPT(GPT_model, query)
    print('GPT answer',gpt_answer)
    context = context + '\n\n' +  gpt_answer['choices'][0]['message']['content'] + '\nSOURCE: GPT chapter GPT'

  prompt = f" You are a truthful assistant that helps synthesising information from multiple sources. \
  Base your answer only on the given context and show your answer is correct by listing all the different sources used to provide the answer in the format SOURCE <insert text> chapter <insert text>. \
  Not all context might be relevant.\
  The query is: {query}  \n Context: '''{context}'''"

  completion = call_GPT(GPT_model,prompt)
  print(completion)
  print(completion['choices'][0]['message']['content'])

