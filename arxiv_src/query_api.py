import json
import openai
import pandas as pd
import torch
import  numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder, util

class EO_bot():
    def __init__(self):
        with open('/home/lcamilleri/git_repos/NLP4EO/api_key.json', 'r') as f:
            self.key = json.load(f)['key']
        self.model = "gpt-3.5-turbo"
        self.text_resources = pd.read_csv('/home/lcamilleri/git_repos/NLP4EO/arxiv_data/papers_text_data_wo_sw.csv', lineterminator='\n')
        self.corpus_embeddings = torch.from_numpy(np.load('/home/lcamilleri/git_repos/NLP4EO/arxiv_src/paper_embeddings_wo_sw.npy')).cuda()
        self.bi_encoder = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1').cuda()
        self.bi_encoder.max_seq_length = 512  # Truncate long passages to 256 tokens
        self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        self.top_k = 32  # Number of passages we want to retrieve with the bi-encoder
        self.top_r = 5

    def sematic_search(self, query):
        question_embedding = self.bi_encoder.encode(query, convert_to_tensor=True)
        question_embedding = question_embedding.cuda()
        hits = util.semantic_search(question_embedding, self.corpus_embeddings, top_k=self.top_k)
        hits = hits[0]  # Get the hits for the first query

        ##### Re-Ranking #####
        # Now, score all retrieved passages with the cross_encoder
        cross_inp = [[query, self.text_resources.iloc[hit['corpus_id'], 2]] for hit in hits]
        cross_scores = self.cross_encoder.predict(cross_inp)

        # Sort results by the cross-encoder scores
        for idx in range(len(cross_scores)):
            hits[idx]['cross-score'] = cross_scores[idx]

        # # Output of top-5 hits from bi-encoder
        # print("\n-------------------------\n")
        # print("Top-3 Bi-Encoder Retrieval hits")
        hits = sorted(hits, key=lambda x: x['cross-score'], reverse=True)
        top_responses = [[self.text_resources.iloc[hit['corpus_id'], 1], self.text_resources.iloc[hit['corpus_id'], 2],
                            self.text_resources.iloc[hit['corpus_id'], 3], self.text_resources.iloc[hit['corpus_id'], 4],
                            self.text_resources.iloc[hit['corpus_id'], 5], hit['score'], self.text_resources.iloc[hit['corpus_id'], 2]] for
                           hit in hits[:self.top_r]]

        return top_responses

    def summarise(self, query, top_responses):
        context = ''
        openai.api_key = self.key

        for i, response in enumerate(top_responses):
            context = context + f'\nSource: Title: {response[0]}, url_link: {response[2]}, authors: {response[4]}' \
                                f'\nContent:{response[1]}\n-----------'

        prompt = f"As an academic write an factual answer to the following question: '''{query}'''. " \
                 f"Please use the the context provided as a truthful source of information." \
                 f"You can ignore irrelevant sources." \
                 f"In your answer include citations of the sources you use in the following format: <Title>, <url link>. " \
                 f"Included the citations after the relevant text." \
                 f"\nContext: '''{context}'''\n"

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
        return { 'context': context,
                 'response': completion['choices'][0]['message']['content']}
def main():
    print()

if __name__ == '__main__':
    main()