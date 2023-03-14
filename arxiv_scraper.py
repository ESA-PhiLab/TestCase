from paperscraper.get_dumps import biorxiv, medrxiv, chemrxiv
from arxiv_utils import get_and_dump_arxiv_papers
from arxiv_utils import save_pdf_from_dump
import arxiv
import json
from typing import Any, Dict, List
from tqdm import tqdm

import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger(__name__)

def get_metadata(query, output_filepath='/home/lcamilleri/git_repos/NLP4EO/arxiv_data'):
    file_name = '_'.join(query[0])
    get_and_dump_arxiv_papers(query, output_filepath=f'{output_filepath}/{file_name}.jsonl', max_results=1000,
                              search_options={'sort_by': arxiv.SortCriterion.Relevance})


def get_pdfs_custom():
    save_pdf_from_dump('/home/lcamilleri/git_repos/NLP4EO/arxiv_data/test.jsonl',
                       pdf_path='/home/lcamilleri/git_repos/NLP4EO/arxiv_data/EO_papers', key_to_save='title')


def main():
    eo = ['Earth Observation', 'Remote Sensing']
    query = [eo]
    get_metadata(query)
    get_pdfs_custom()


def initialization():
    medrxiv()  # Takes ~30min and should result in ~35 MB file
    biorxiv()  # Takes ~1h and should result in ~350 MB file
    chemrxiv()  # Takes ~45min and should result in ~20 MB file

if __name__ == '__main__':
    initialization()
    main()