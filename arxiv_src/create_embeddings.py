import os
import openai
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder, util
import os
import torch
import json
from tqdm import tqdm
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def gen_papers_jsonfile():
    meta_data = pd.read_json('/home/lcamilleri/git_repos/NLP4EO/arxiv_data/EO_RS_papers.jsonl', lines=True)
    df = pd.DataFrame()

    for index, row in tqdm(meta_data.iterrows()):
        title = row['title']
        abs = row.loc[['title', 'abstract', 'pdf_url', 'doi']]
        df = df.append(abs)

        try:
            text_data = pd.read_json(f'/home/lcamilleri/git_repos/NLP4EO/arxiv_data/EO_textdata/{title}.json', lines=True)
            # ignore first two chunks
            text_data = text_data.iloc[:, 1:]
            for index, col in text_data.T.iterrows():
                temp_df = pd.DataFrame({'title': [title],
                                       'abstract':[col[0]],
                                        'pdf_url':[row['pdf_url']],
                                        'doi':[row['doi']]})
                df = df.append(temp_df)
        except:
            print(f'{title} not found')

    df.reset_index(drop=True, inplace=True)
    df.to_csv('/home/lcamilleri/git_repos/NLP4EO/arxiv_data/papers_text_data.csv')



def gen_abstract_embeddings():
    meta_data = pd.read_csv('/home/lcamilleri/git_repos/NLP4EO/arxiv_data/papers_text_data.csv', lineterminator='\n')
    # We use the Bi-Encoder to encode all passages, so that we can use it with sematic search
    bi_encoder = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')
    bi_encoder.max_seq_length = 512  # Truncate long passages to 256 tokens
    top_k = 32  # Number of passages we want to retrieve with the bi-encoder

    # The bi-encoder will retrieve 100 documents. We use a cross-encoder, to re-rank the results list to improve the quality
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    abstracts = []
    for index, row in meta_data.iterrows():
        txt = row['abstract']
        abstracts.append(txt)

    corpus_embeddings = bi_encoder.encode(abstracts, convert_to_tensor=True, show_progress_bar=True)
    np.save('paper_embeddings.npy', corpus_embeddings.to('cpu'))

def search(query):
    print("Input question:", query)
    meta_data = pd.read_csv('/home/lcamilleri/git_repos/NLP4EO/arxiv_data/papers_text_data.csv', lineterminator='\n')
    corpus_embeddings = torch.from_numpy(np.load('paper_embeddings.npy')).cuda()
    # We use the Bi-Encoder to encode all passages, so that we can use it with sematic search
    bi_encoder = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1').cuda()
    bi_encoder.max_seq_length = 512  # Truncate long passages to 256 tokens
    top_k = 32  # Number of passages we want to retrieve with the bi-encoder

    # The bi-encoder will retrieve 100 documents. We use a cross-encoder, to re-rank the results list to improve the quality
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    ##### Sematic Search #####
    # Encode the query using the bi-encoder and find potentially relevant passages
    question_embedding = bi_encoder.encode(query, convert_to_tensor=True)
    question_embedding = question_embedding.cuda()
    hits = util.semantic_search(question_embedding, corpus_embeddings, top_k=top_k)
    hits = hits[0]  # Get the hits for the first query

    ##### Re-Ranking #####
    # Now, score all retrieved passages with the cross_encoder
    cross_inp = [[query, meta_data.iloc[hit['corpus_id'], 2]] for hit in hits]
    cross_scores = cross_encoder.predict(cross_inp)

    # Sort results by the cross-encoder scores
    for idx in range(len(cross_scores)):
        hits[idx]['cross-score'] = cross_scores[idx]

    # # Output of top-5 hits from bi-encoder
    # print("\n-------------------------\n")
    # print("Top-3 Bi-Encoder Retrieval hits")
    hits = sorted(hits, key=lambda x: x['cross-score'], reverse=True)

    top_5_responses = [[meta_data.iloc[hit['corpus_id'], 1], meta_data.iloc[hit['corpus_id'], 2], meta_data.iloc[hit['corpus_id'], 3] , meta_data.iloc[hit['corpus_id'],  4]] for hit in hits[:8]]

    return top_5_responses

def summarise(query, top_5_responses):
    context = ''
    bi_encoder = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')

    with open('/home/lcamilleri/git_repos/NLP4EO/api_key.json', 'r') as f:
        key = json.load(f)['key']

    openai.api_key = key

    for i, response in enumerate(top_5_responses):
        context = context + f'\nSource:{response[0]}\nUrl_link:{response[2]}\nContent:{response[1]}\n-----------'

    prompt = f'You are a summarization tool that writes a literature review factually based on proved information.\n' \
             f'Answer the following question: {query}\n Using ONLY the following sources of information in: {context}' \
             f'\nUse linking sentences. Sources do not need to be referenced in order.' \
             f'\nShow your answer is correct by citing the sources.\n Use Title and Url_link to cite sources.' \
             f'If the information is not available say so.\n'

    print(prompt)

    model = "gpt-3.5-turbo"
    completion = openai.ChatCompletion.create(
        messages=[{"role": "system", "content": prompt}, ],
        model=model,
        # prompt=prompt,
        max_tokens=4096 - int(1.4 * len(bi_encoder.tokenizer.encode(prompt))),
        temperature=0
    )

    print(completion)
    print(completion['choices'][0]['message']['content'])


if __name__ == '__main__':
    # gen_papers_jsonfile()
    # gen_abstract_embeddings()
    query = 'Imagine you have a Sentinel 2 and a Sentinel 1 SAR image - what kind of values would I expected over a forested area vs a water body?'
    top_5_responses = search(query)
    summarise(query, top_5_responses)
