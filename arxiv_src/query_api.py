import json
import openai
import pandas as pd
import torch
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder, util
import pickle

class EO_bot():
    def __init__(self, source_arxiv=True):
        with open('/home/lcamilleri/git_repos/NLP4EO/api_key.json', 'r') as f:
            self.key = json.load(f)['key']
        self.model = "gpt-3.5-turbo"
        # self.text_resources = pd.read_csv('/home/lcamilleri/git_repos/NLP4EO/arxiv_data/papers_text_data_wo_sw.csv', lineterminator='\n')
        # self.corpus_embeddings = torch.from_numpy(np.load('/home/lcamilleri/git_repos/NLP4EO/arxiv_src/paper_embeddings_wo_sw.npy')).cuda()
        self.bi_encoder = SentenceTransformer('multi-qa-mpnet-base-dot-v1').cuda()
        self.bi_encoder.max_seq_length = 512  # Truncate long passages to 256 tokens
        self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2')
        self.top_k = 100  # Number of passages we want to retrieve with the bi-encoder
        self.top_r = 4
        self.source_arxiv = source_arxiv

        if source_arxiv:
            self.text_resources = pd.read_csv('/home/lcamilleri/git_repos/NLP4EO/arxiv_data/papers_text_data_wo_sw.csv',
                                              lineterminator='\n')
            self.corpus_embeddings = torch.from_numpy(
                np.load('/home/lcamilleri/git_repos/NLP4EO/arxiv_src/paper_embeddings_wo_sw.npy')).cuda()

        else:
            with open("/home/lcamilleri/git_repos/NLP4EO/EO_portal_demo/chapter_txt.pkl", "rb") as fp:  # Unpickling
                self.text_resources = pickle.load(fp)
            self.corpus_embeddings = torch.from_numpy(np.load('/home/lcamilleri/git_repos/NLP4EO/EO_portal_demo/chapter_embeddings.npy'))


    def sematic_search(self, query):
        question_embedding = self.bi_encoder.encode(query, convert_to_tensor=True)
        question_embedding = question_embedding.cuda()
        hits = util.semantic_search(question_embedding, self.corpus_embeddings, top_k=self.top_k)
        hits = hits[0]  # Get the hits for the first query

        if not self.source_arxiv:
            hits_df = pd.DataFrame(hits)[:32]
            hits_df.score = hits_df.score.apply(lambda x: round(x, 6))
            hits_df_dropped = hits_df.drop_duplicates(subset='score').sort_values('score')[::-1]
            hits = hits_df_dropped.to_dict(orient='records')[:self.top_k]

            self.text_resources = pd.DataFrame(self.text_resources)
            cross_inp = [[query, self.text_resources.iloc[hit['corpus_id'], 0]] for hit in hits]
        else:
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
        return hits

    def summarise(self, query, hits):
        context = ''
        openai.api_key = self.key

        if self.source_arxiv:
            top_responses = [
                [self.text_resources.iloc[hit['corpus_id'], 1], self.text_resources.iloc[hit['corpus_id'], 2],
                 self.text_resources.iloc[hit['corpus_id'], 3], self.text_resources.iloc[hit['corpus_id'], 4],
                 self.text_resources.iloc[hit['corpus_id'], 5], hit['score'],
                 self.text_resources.iloc[hit['corpus_id'], 2]] for
                hit in hits[:self.top_r]]

            for i, response in enumerate(top_responses):
                context = context + f'\nSource {i+1}\nTitle: {response[0]}, url_link: {response[2]}, authors: {response[4]}' \
                                    f'\nContent:\n{response[1]}\n-----------'



            # prompt = f"As an academic write an factual answer to the following question: '''{query}'''. " \
            #          f"Please use the the context provided as a truthful source of information." \
            #          f"You can ignore irrelevant sources." \
            #          f"In your answer include citations of the sources you use in the following format: Title, url link. " \
            #          f"Included the citations after the relevant text." \
            #          f"\nContext: '''{context}'''\n"

            # prompt = f"Evidence: '''{context}'''\n" \
            #          f"As an academic write an factual answer to the above question. " \
            #          f"Please use the evidence provided as a factual source of information. " \
            #          f"Ignore irrelevant information in the evidence. " \
            #          f"You are allowed to use your own internal logic to help answer the question. "\
            #          f"You can say you don't know the answer. " \
            #          f"Included the references to sources in your answer ." \
            #          f"Included a biography with the source title, authors & url link. " \
            #          f"Question: '''{query}'''\n" \
            #          f"\nAnswer:\n"

            prompt =    f"""As an academic, answer the question as truthfully as possible using the provided context, and if the answer is not contained within the context below, say "I don't know.\n"""\
                        f"""Cite sources from the provided context in your answer and include a bibliography in the following format:\n"""\
                        f"""-- Here is some example text. You can cite context sources using a citation key [1]. You must include a bibliography with the cited sources.\n"""\
                        f"""It is important that text is related to the cited source [2].\n"""\
                        """Bibliography:\n"""\
                        """[1] Claudio Cusano, Paolo Napoletano, and Raimondo Schettini, “Remote sensing image classification exploiting multiple kernel learning,” arXiv preprint arXiv:1410.5358\n"""\
                        """[2] Marco Castelluccio, Giovanni Poggi, Carlo Sansone, and Luisa Verdoliva, “Land use classification in remote sensing images by convolutional neural networks,” arXiv preprint arXiv:1508.00092 --"""\
                        f"""\n\nContext:\n-----------{context}""" + f"\n\nQuestion: {query} \nAnswer:"""
            print(prompt)
        else:
            for hit in hits[0:10]:
                context = context + self.text_resources.iloc[hit['corpus_id'], 0] + '\n\n'

            prompt = f" You are a truthful assistant that helps synthesising information from multiple sources. \
                     Base your answer only on the given context and show your answer is correct by listing all the different sources used to provide the answer in the format SOURCE <insert text> chapter <insert text>. \
                     Not all context might be relevant.\
                     The query is: {query}  \n Context: '''{context}'''"



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
    bot = EO_bot(source_arxiv=True)
    top_sources = bot.sematic_search('what is EO?')
    response = bot.summarise('what is EO?', top_sources)

if __name__ == '__main__':
    main()