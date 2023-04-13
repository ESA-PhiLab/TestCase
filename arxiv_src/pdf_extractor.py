from io import StringIO

import pypdf.errors
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfparser import PDFParser

import PyPDF2
import string
import csv
import re
from pypdf import PdfReader

from pathlib import Path
from tqdm import tqdm
import os
import json


EOF_MARKER = b'%%EOF'


def extract_text(file_path):
    if str(file_path).endswith('.pdf'):
        # Extract text from a PDF file
        text = ''
        try:
            reader = PdfReader(str(file_path), strict=True)

            number_of_pages = len(reader.pages)
            for i in range(number_of_pages):
                page = reader.pages[i]
                page_text = page.extract_text()
                text += page_text + "\n"

            return text

        except pypdf.errors.PyPdfError:
            print(f'could not load: {file_path}')
            return -1

    else:
        print('Unsupported file type')


def filter_nonprintable_characters(text):
    printable_text = ''.join(filter(lambda x: x in string.printable, text))
    printable_text = printable_text.replace('\n', '')
    return printable_text

def divide_text_into_chunks(text, chunk_size=256):
    # Create an empty list to hold the text chunks
    text_chunks = []

    # Create an empty list to hold the text chunks
    text_chunks = []

    # Split the text into words
    words = text.split(' ')

    # Loop through the words, adding chunks of the specified size to the list
    for i in range(0, len(words), chunk_size):
        word_chunk = words[i:i + chunk_size]
        text_chunk = " ".join(word_chunk)
        text_chunks.append(text_chunk)

    # Return the list of text chunks
    return text_chunks


def main():

    for pdf in tqdm(Path('/home/lcamilleri/git_repos/NLP4EO/arxiv_data/EO_papers/').rglob('*.pdf')):

        text = extract_text(pdf)
        if text != -1:
            text = filter_nonprintable_characters(text)
            text_chunks = divide_text_into_chunks(text)


            dict = {"title": pdf.stem}
            for i in range(len(text_chunks)):
                dict[f"paragraph_{i}"] = text_chunks[i]

            with open(f"/home/lcamilleri/git_repos/NLP4EO/arxiv_data/EO_textdata/{pdf.stem}.json", "w") as fp:
                json.dump(dict, fp)


if __name__ == '__main__':
    main()