import json
import numpy as np
import pandas as  pd
import glob
import pickle

from sentence_transformers import SentenceTransformer, CrossEncoder, util  
 

def parse_jsons(json_paths, bi_encoder, max_tokens_per_passage):
    chapter_content = []

    for jf in json_paths:
        with open(jf, 'r') as f:
            data = json.load(f)

            mission = data['mission']
            content = data['content']
            chapters = data['chapters']
            chapters = [ch.strip() for ch in chapters if len(ch.strip())>2]


            chapter_idx = []
            for k,line in enumerate(content):
                if any(((line[:-int(len(line)*0.3)] in ch) & (ch[:-int(len(ch)*0.3)] in line)) for ch in chapters):
                    chapter_idx.append(k)
            chapter_idx.append(len(content))
                    

            for i in range(len(chapter_idx)-1):
                paragraph = '\n'.join(content[chapter_idx[i]+1:chapter_idx[i+1]])
                num_tokens = len(bi_encoder.tokenizer.encode(paragraph))
                n_chunks = num_tokens//max_tokens_per_passage+1
                chars_per_chunk = len(paragraph)//n_chunks
                for chunk in range(n_chunks):
                    chapter_content.append(paragraph[chunk*chars_per_chunk:(chunk+1)*chars_per_chunk] + f'\nSOURCE: {mission} chapter {content[chapter_idx[i]]}' )
        
    return chapter_content

if __name__=='__main__':
    json_files = glob.glob('/home/lcamilleri/Downloads/EoDirectoryMissionExport/**.json', recursive=True)
    assert len(json_files)>0
    
    max_tokens_per_passage = 256
    bi_encoder = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')

    chapter_contents = parse_jsons(json_files, bi_encoder, max_tokens_per_passage)
    corpus_embeddings = bi_encoder.encode(chapter_contents, convert_to_tensor=True, show_progress_bar=True)

    np.save('chapter_embeddings', corpus_embeddings.cpu())
    with open("chapter_txt.pkl.pkl", "wb") as fp:   #Pickling
        pickle.dump(chapter_contents, fp)