import PyPDF2
import nltk
from gensim.models import doc2vec
from collections import namedtuple


class PDFReader:

    def __init__(self, pdf_file):
        self.pdf_file = pdf_file
        self.pdfFileObj = open(self.pdf_file, 'rb')
        self.pdfReader = PyPDF2.PdfFileReader(self.pdfFileObj)

    def get_page_count(self):
        return self.pdfReader.getNumPages()

    def get_paragraphs_from_page(self, page_num):
        pageObj = self.pdfReader.getPage(page_num)
        text = pageObj.extractText()
        text = text.replace('\n', ' ')
        splits = text.split('      ')
        return splits

    def get_all_paragraphs(self):
        page_count = self.get_page_count()
        paragraphs = []
        for page_num in range(page_count):
            pageObj = self.pdfReader.getPage(page_num)
            text = pageObj.extractText()
            text = text.replace('\n', ' ')
            splits = text.split('      ')
            paragraphs.extend(splits)
        return paragraphs



