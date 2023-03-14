from typing import Dict, List, Union

import pandas as pd
from tqdm import tqdm

import arxiv

import json
import logging
import sys
from typing import Dict, List

import pandas as pd

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger(__name__)


def dump_papers(papers: pd.DataFrame, filepath: str) -> None:
    """
    Receives a pd.DataFrame, one paper per row and dumps it into a .jsonl
    file with one paper per line.

    Args:
        papers (pd.DataFrame): A dataframe of paper metadata, one paper per row.
        filepath (str): Path to dump the papers, has to end with `.jsonl`.
    """
    if not isinstance(filepath, str):
        raise TypeError(f"filepath must be a string, not {type(filepath)}")
    if not filepath.endswith(".jsonl"):
        raise ValueError("Please provide a filepath with .jsonl extension")

    if isinstance(papers, List) and all([isinstance(p, Dict) for p in papers]):
        papers = pd.DataFrame(papers)
        logger.warning(
            "Preferably pass a pd.DataFrame, not a list of dictionaries. "
            "Passing a list is a legacy functionality that might become deprecated."
        )

    if not isinstance(papers, pd.DataFrame):
        raise TypeError(f"papers must be a pd.DataFrame, not {type(papers)}")

    paper_list = list(papers.T.to_dict().values())

    with open(filepath, "w") as f:
        for paper in paper_list:
            f.write(json.dumps(paper) + "\n")


def get_filename_from_query(query: List[str]) -> str:
    """Convert a keyword query into filenames to dump the paper.

    Args:
        query (list): List of string with keywords.

    Returns:
        str: Filename.
    """
    filename = "_".join([k if isinstance(k, str) else k[0] for k in query]) + ".jsonl"
    filename = filename.replace(" ", "").lower()
    return filename


def load_jsonl(filepath: str) -> List[Dict[str, str]]:
    """
    Load data from a `.jsonl` file, i.e., a file with one dictionary per line.

    Args:
        filepath (str): Path to `.jsonl` file.

    Returns:
        List[Dict[str, str]]: A list of dictionaries, one per paper.
    """

    with open(filepath, "r") as f:
        data = [json.loads(line) for line in f.readlines()]
    return data

arxiv_field_mapper = {
    "published": "date",
    "journal_ref": "journal",
    "summary": "abstract",
}

# Authors, date, and journal fields need specific processing
process_fields = {
    "authors": lambda authors: ", ".join([a.name for a in authors]),
    "date": lambda date: date.strftime("%Y-%m-%d"),
    "journal": lambda j: j if j is not None else "",
}

def get_query_from_keywords(keywords: List[Union[str, List[str]]]) -> str:
    """Receives a list of keywords and returns the query for the arxiv API.

    Args:
        keywords (List[str, List[str]]): Items will be AND separated. If items
            are lists themselves, they will be OR separated.

    Returns:
        str: query to enter to arxiv API.
    """

    query = ""
    finalize_disjunction = lambda x: "(" + x[:-4] + ") AND "
    finalize_conjunction = lambda x: x[:-5]

    for i, key in enumerate(keywords):
        if isinstance(key, str):
            query += f"ti:{key} AND "
        elif isinstance(key, list):
            inter = "".join([f"ti:{syn} OR abs:{syn} OR " for syn in key])
            query += finalize_disjunction(inter)

    query = finalize_conjunction(query)
    return query

def get_arxiv_papers(
    query: str,
    fields: List = ["entry_id","title", "authors", "date", "abstract", "journal", "doi", "primary_category", "categories", "pdf_url"],
    max_results: int = 99999,
    client_options: Dict = {"num_retries": 10},
    search_options: Dict = dict(),
) -> pd.DataFrame:
    """
    Performs arxiv API request of a given query and returns list of papers with
    fields as desired.

    Args:
        query (str): Query to arxiv API. Needs to match the arxiv API notation.
        fields (List[str]): List of strings with fields to keep in output.
        max_results (int): Maximal number of results, defaults to 99999.
        client_options (Dict): Optional arguments for `arxiv.Client`. E.g.:
            page_size (int), delay_seconds (int), num_retries (int).
            NOTE: Decreasing 'num_retries' will speed up processing but might
            result in more frequent 'UnexpectedEmptyPageErrors'.
        search_options (Dict): Optional arguments for `arxiv.Search`. E.g.:
            id_list (List), sort_by, or sort_order.

    Returns:
        pd.DataFrame: One row per paper.

    """
    client = arxiv.Client(**client_options)
    search = arxiv.Search(query=query, max_results=max_results, **search_options)
    results = client.results(search)

    processed = pd.DataFrame(
        [
            {
                arxiv_field_mapper.get(key, key): process_fields.get(
                    arxiv_field_mapper.get(key, key), lambda x: x
                )(value)
                for key, value in vars(paper).items()
                if arxiv_field_mapper.get(key, key) in fields
            }
            for paper in results
        ]
    )
    return processed


def get_and_dump_arxiv_papers(
    keywords: List[Union[str, List[str]]],
    output_filepath: str,
    fields: List = ["entry_id","title", "authors", "date", "abstract", "journal", "doi", "primary_category", "categories", "pdf_url"],
    *args,
    **kwargs
):
    """
    Combines get_arxiv_papers and dump_papers.

    Args:
        keywords (List[str, List[str]]): List of keywords to request arxiv API.
            The outer list level will be considered as AND separated keys, the
            inner level as OR separated.
        filepath (str): Path where the dump will be saved.
        fields (List, optional): List of strings with fields to keep in output.
            Defaults to ['title', 'authors', 'date', 'abstract',
            'journal', 'doi'].
        *args, **kwargs are additional arguments for `get_arxiv_papers`.
    """
    # Translate keywords into query.
    query = get_query_from_keywords(keywords)
    papers = get_arxiv_papers(query, fields, *args, **kwargs)
    dump_papers(papers, output_filepath)

def save_pdf_from_dump(dump_path: str, pdf_path: str, key_to_save: str = "doi") -> None:
    """
    Receives a path to a `.jsonl` dump with paper metadata and saves the PDF files of
    each paper.

    Args:
        dump_path: Path to a `.jsonl` file with paper metadata, one paper per line.
        pdf_path: Path to a folder where the files will be stored.
        key_to_save: Key in the paper metadata to use as filename.
            Has to be `doi` or `title`. Defaults to `doi`.
    """

    if not isinstance(dump_path, str):
        raise TypeError(f"dump_path must be a string, not {type(dump_path)}.")
    if not dump_path.endswith(".jsonl"):
        raise ValueError("Please provide a dump_path with .jsonl extension.")

    if not isinstance(pdf_path, str):
        raise TypeError(f"pdf_path must be a string, not {type(pdf_path)}.")

    if not isinstance(key_to_save, str):
        raise TypeError(f"key_to_save must be a string, not {type(key_to_save)}.")
    if key_to_save not in ["doi", "title", "date"]:
        raise ValueError("key_to_save must be one of 'doi' or 'title'.")

    papers = load_jsonl(dump_path)

    pbar = tqdm(papers, total=len(papers), desc="Processing")
    for i, paper in enumerate(pbar):
        pbar.set_description(f"Processing paper {i+1}/{len(papers)}")

        if "entry_id" not in paper.keys() or paper["entry_id"] is None:
            logger.warning(f"Skipping {paper['title']} since no entry_id available.")
            continue
        filename = paper[key_to_save].replace("/", "_")
        id = paper["entry_id"].split('/')[-1]
        try:
            paper_pdf = next(arxiv.Search(id_list=[id]).results())
            paper_pdf.download_pdf(dirpath=pdf_path, filename=f"{filename}.pdf")
        except:
            print(f'PDF not Found: {id}')