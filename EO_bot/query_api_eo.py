import json
import openai
import pandas as pd
import torch
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder, util
import pickle

class eo_bot():
    def __init__(self, source_arxiv=True):
        with open('/home/lcamilleri/git_repos/NLP4EO/api_key.json', 'r') as f:
            self.key = json.load(f)['key']
        self.model = "gpt-3.5-turbo"
        self.bi_encoder = SentenceTransformer('all-mpnet-base-v2').cuda()
        self.bi_encoder.max_seq_length = 512  # Truncate long passages to 256 tokens
        self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2')
        self.top_k = 100  # Number of passages we want to retrieve with the bi-encoder
        self.top_r = 4

        self.text_resources = pd.read_csv('/home/lcamilleri/git_repos/NLP4EO/EO_bot/resources/all_articles.csv',
                                          lineterminator='\n')
        self.corpus_embeddings = torch.from_numpy(
            np.load('/home/lcamilleri/git_repos/NLP4EO/EO_bot/resources/all_embeddings.npy')).cuda()

    def sematic_search(self, query):
        question_embedding = self.bi_encoder.encode(query, convert_to_tensor=True)
        question_embedding = question_embedding.cuda()
        hits = util.semantic_search(question_embedding, self.corpus_embeddings, top_k=self.top_k)
        hits = hits[0]  # Get the hits for the first query


        ##### Re-Ranking #####
        # Now, score all retrieved passages with the cross_encoder
        cross_inp = [[query, self.text_resources.iloc[hit['corpus_id'], 5]] for hit in hits]
        cross_scores = self.cross_encoder.predict(cross_inp)

        # Sort results by the cross-encoder scores
        for idx in range(len(cross_scores)):
            hits[idx]['cross-score'] = cross_scores[idx]

        # # Output of top-5 hits from bi-encoder
        # print("\n-------------------------\n")
        # print("Top-3 Bi-Encoder Retrieval hits")
        hits = sorted(hits, key=lambda x: x['cross-score'], reverse=True)
        return hits

    def summarise(self, query, hits):
        context = ''
        openai.api_key = self.key

        top_responses = [
            [self.text_resources.iloc[hit['corpus_id'], 0], self.text_resources.iloc[hit['corpus_id'], 1],
             self.text_resources.iloc[hit['corpus_id'], 2], self.text_resources.iloc[hit['corpus_id'], 3],
             self.text_resources.iloc[hit['corpus_id'], 4], self.text_resources.iloc[hit['corpus_id'], 5]] for
            hit in hits[:self.top_r]]

        for i, response in enumerate(top_responses):
            context = context + f'\nSource {i+1}\n authors: {response[2]}, title: {response[0]}, url_link: {response[1]}' \
                                f'\nContent:\n\n{response[5]}\n-----------'
        #
        # prompt = f"""As an academic, answer the question as truthfully as possible using the provided context, and if the answer is not contained within the context below, say "I don't know.\n""" \
        #          f"""Cite sources from the provided context in your answer and include a bibliography in the following format:\n""" \
        #          f"""-- Here is some example text. You can cite context sources using a citation key [1]. You must include a bibliography with the cited sources.\n""" \
        #          f"""It is important that text is related to the cited source [2].\n""" \
        #          """Bibliography:\n""" \
        #          """[1] Claudio Cusano, Paolo Napoletano, and Raimondo Schettini, “Remote sensing image classification exploiting multiple kernel learning,” arXiv preprint arXiv:1410.5358\n""" \
        #          """[2] Marco Castelluccio, Giovanni Poggi, Carlo Sansone, and Luisa Verdoliva, “Land use classification in remote sensing images by convolutional neural networks,” arXiv preprint arXiv:1508.00092 --""" \
        #          f"""\n\nContext:\n-----------{context}""" + f"\n\nQuestion: {query} \nAnswer:"""

        prompt = f"You are a truthful bot.\n Write a literature review using the provided context to answer the question as truthfully as possible and with as much detail as possible.\n" \
                 f"Use the evidence provided as a factual source of information.\n" \
                 f"Ignore irrelevant information in the evidence. " \
                 f"Use your own internal logic to help answer the question. "\
                 f"Say you don't know the answer, if no relevant context is present.\n" \
                 f"""Cite sources from the provided context in your answer and include a bibliography in the following format:\n""" \
                 f"""-- Here is some example text. You can cite context sources using a citation key [1]. You must include a bibliography with the cited sources. """ \
                 f"""It is important that text is related to the cited source [2].\n""" \
                 """Bibliography:\n""" \
                 """[1] Claudio Cusano, Paolo Napoletano, and Raimondo Schettini, “Remote sensing image classification exploiting multiple kernel learning”, arXiv preprint arXiv:1410.5358\n""" \
                 """[2] @xen0f0n, “Spectral Reflectance Newsletter #27”, https://medium.com/spectral-reflectance/spectral-reflectance-newsletter-27-82a9aa2add1d --\n""" \
                 f"Context: '''{context}'''\n" \
                 f"Question: '''{query}'''\n" \
                 f"\nAnswer:\n"
        print(prompt)

        model = self.model
        completion = openai.ChatCompletion.create(
            messages=[{"role": "system", "content": prompt}, ],
            model=model,
            # prompt=prompt,
            max_tokens=4096 - int(1.4 * len(self.bi_encoder.tokenizer.encode(prompt))),
            temperature=0
        )

        print(completion)
        print(completion['choices'][0]['message']['content'])
        return { 'context': context,
                 'response': completion['choices'][0]['message']['content']}

def main():
    bot = eo_bot()
    query = 'How to apply self supervised learning to remote sensing data?'
    top_sources = bot.sematic_search(query)
    response = bot.summarise(query, top_sources)

if __name__ == '__main__':
    main()