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
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from collections import Counter
warnings.simplefilter(action='ignore', category=FutureWarning)

stop_words = stopwords.words('english')
stopwords_dict = Counter(stop_words)

BASE_PATH = '/home/lcamilleri/git_repos/'
# BASE_PATH = ''
def remove_stopwords(text):
    text = text.replace('\n', ' ')
    text_tokens = text.split(' ')
    tokens_without_sw = [word for word in text_tokens if not word in stopwords_dict]
    return " ".join(tokens_without_sw)

def gen_papers_csv(rm_stopwords=False):
    meta_data = pd.read_json(f'{BASE_PATH}arxiv_data/EO_RS_papers.jsonl', lines=True)
    df = pd.DataFrame()

    for index, row in tqdm(meta_data.iterrows()):

        title = row['title']
        abs = row.loc[['title', 'abstract', 'pdf_url', 'doi', 'authors']]
        if rm_stopwords:
            abs['abstract'] = remove_stopwords(abs['abstract'])
        df = df.append(abs)

        try:
            text_data = pd.read_json(f'{BASE_PATH}arxiv_data/EO_textdata/{title}.json', lines=True)
            # ignore first two chunks and last 5. Course way of removing noise and references
            text_data = text_data.iloc[:, 2:-5]
            for index, col in text_data.T.iterrows():
                text = col[0]
                if rm_stopwords:
                    text = remove_stopwords(text)

                temp_df = pd.DataFrame({'title': [title],
                                        'abstract':[text],
                                        'pdf_url':[row['pdf_url']],
                                        'doi':[row['doi']],
                                        'authors':[row['authors']]})
                df = df.append(temp_df)
        except:
            print(f'{title} not found')

    df.reset_index(drop=True, inplace=True)
    if remove_stopwords:
        df.to_csv(f'{BASE_PATH}arxiv_data/papers_text_data_wo_sw.csv')
    else:
        df.to_csv(f'{BASE_PATH}arxiv_data/papers_text_data.csv')



def gen_abstract_embeddings():
    meta_data = pd.read_csv(f'{BASE_PATH}arxiv_data/papers_text_data_wo_sw.csv', lineterminator='\n')
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
    np.save('paper_embeddings_wo_sw.npy', corpus_embeddings.to('cpu'))

def search(query):
    print("Input question:", query)
    meta_data = pd.read_csv(f'{BASE_PATH}arxiv_data/papers_text_data_wo_sw.csv', lineterminator='\n')
    corpus_embeddings = torch.from_numpy(np.load('paper_embeddings_wo_sw.npy')).cuda()
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


    top_5_responses = [[meta_data.iloc[hit['corpus_id'], 1], summarize_text(meta_data.iloc[hit['corpus_id'], 2]),
                        meta_data.iloc[hit['corpus_id'], 3] , meta_data.iloc[hit['corpus_id'],  4],
                        meta_data.iloc[hit['corpus_id'],  5], hit['score'], meta_data.iloc[hit['corpus_id'], 2]] for hit in hits[:5]]

    return top_5_responses

def summarize_text(query):
    bi_encoder = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')

    with open(f'{BASE_PATH}api_key.json', 'r') as f:
        key = json.load(f)['key']
    openai.api_key = key

    prompt = f"summarize '''{query}''' in 1 to 2 lines"


    print(prompt)

    model = "gpt-3.5-turbo"
    completion = openai.ChatCompletion.create(
        messages=[{"role": "system", "content": prompt}, ],
        model=model,
        # prompt=prompt,
        max_tokens=4096 - int(1.4 * len(bi_encoder.tokenizer.encode(prompt))),
        temperature=0
    )

def extract_keywords(query):
    bi_encoder = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')

    with open(f'{BASE_PATH}api_key.json', 'r') as f:
        key = json.load(f)['key']
    openai.api_key = key

    prompt = f"Extract keywords from the corresponding texts below."\
             f"\nText 1: Stripe provides APIs that web developers can use to integrate payment processing into their websites and mobile applications."\
             f"\nKeywords 1: Stripe, payment processing, APIs, web developers, websites, mobile applications"\
             f"\n##"\
             f"\nText 2: OpenAI has trained cutting-edge language models that are very good at understanding and generating text. Our API provides access to these models and can be used to solve virtually any task that involves processing language." \
             f"\nKeywords 2: OpenAI, language models, text processing, API." \
             f"\n##"\
             f"\nText 3: What are the use cases for SAR data?" \
             f"\nKeywords 3: use cases, SAR, data" \
             f"\n##" \
             f"\nText 4: How is sentinel-2 data used in land cover tasks?" \
             f"\nKeywords 4: sentinel-2, land cover" \
             f"\n##" \
             f"\nText 5: {query}" \
             f"\nKeywords 5: "

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
    return completion['choices'][0]['message']['content']





def summarise(query, top_5_responses):
    context = ''
    bi_encoder = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')

    with open(f'{BASE_PATH}api_key.json', 'r') as f:
        key = json.load(f)['key']

    openai.api_key = key

    for i, response in enumerate(top_5_responses):
        context = context + f'\nSource: {response[0]}, url_link: {response[2]}, authors: {response[4]}' \
                            f'\nRelevance score: {response[-1]}\nContent:{response[1]}\n-----------'

    # prompt = f'You are a summarization tool that has been presented with a question and information about the question topic.' \
    #          f'Use the information provided to answer the question factually.\n' \
    #          f'If the information is not available say so.\n' \
    #          f'Cite the sources! Use title, url_link and authors to cite sources.\n' \
    #          f'Reference sources in text.\n'\
    #          f'Write approx. 500 words.\n'\
    #          f'Question: {query}\nInformation:\n-----------{context}\n'

    # prompt = f"You are a factual summarization assistant that helps summarize information from mulitple sources. " \
    #          f"Base your answer only on the given context and how your answer is correct by listing all the different " \
    #          f"sources. They must be referenced in text using the format; title: <insert text> url_link: <insert text> authors: <insert text>. " \
    #          f"Answer the question in approx. 300 words."\
    #          f"\nThe query is: {query}" \
    #          f"\nContext: '''{context}'''\n" \
    #

    prompt = f"As an academic write a 200 word literature review using the the context provided to answer the following question: {query}. " \
             "You can ignore irrelevant sources." \
             f"Cite the sources you use in the text and also include the list of these citations with source name, url_link and authors at the end of the text." \
             f"\nContext: '''{context}'''\n" \

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
    # gen_papers_csv(rm_stopwords=True)
    # gen_abstract_embeddings()
    query = 'what are some change detection techniques?'
    # ss_query = extract_keywords(query)
    top_5_responses = search("change detection, techniques",)
    summarise(query, top_5_responses)
