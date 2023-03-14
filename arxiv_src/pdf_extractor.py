from io import StringIO

from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfparser import PDFParser

import PyPDF2
import csv

from pathlib import Path
from tqdm import tqdm

def convert_pdf_to_string(file_path):
    output_string = StringIO()
    with open(file_path, 'rb') as in_file:
        parser = PDFParser(in_file)
        doc = PDFDocument(parser)
        rsrcmgr = PDFResourceManager()
        device = TextConverter(rsrcmgr, output_string, laparams=LAParams())
        interpreter = PDFPageInterpreter(rsrcmgr, device)
        for page in PDFPage.create_pages(doc):
            interpreter.process_page(page)

    return (output_string.getvalue())


def convert_title_to_filename(title):
    filename = title.lower()
    filename = filename.replace(' ', '_')
    return filename


def split_to_title_and_pagenum(table_of_contents_entry):
    title_and_pagenum = table_of_contents_entry.strip()

    title = None
    pagenum = None

    if len(title_and_pagenum) > 0:
        if title_and_pagenum[-1].isdigit():
            i = -2
            while title_and_pagenum[i].isdigit():
                i -= 1

            title = title_and_pagenum[:i].strip()
            pagenum = int(title_and_pagenum[i:].strip())

    return title, pagenum

def extract_text(file_path):
    text = ''
    if file_path.endswith('.pdf'):
        # Extract text from a PDF file
        with open(file_path, 'rb') as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            num_pages = len(pdf_reader.pages)
            for i in range(num_pages):
                page = pdf_reader.pages[i]
                page_text = page.extract_text()
                text += page_text

        return text

    else:
        print('Unsupported file type')

def main():
    for pdf in tqdm(Path('/arxiv_data/EO_papers').rglob('*.pdf')):
        reader = PyPDF2.PdfReader(str(pdf))
        print(reader.metadata)

        num_of_pages =  len(reader.pages)
        print('Number of pages: ' + str(num_of_pages))

        text = convert_pdf_to_string(str(pdf))
        text_2 = extract_text(str(pdf))
        print()


if __name__ == '__main__':
    main()