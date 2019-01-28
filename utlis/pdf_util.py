import PyPDF2
from io import StringIO
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
import os
import sys, getopt


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

    def get_text(self):
        text = ''
        page_count = self.get_page_count()
        for page_num in range(page_count):
            pageObj = self.pdfReader.getPage(page_num)
            page = pageObj.extractText()
            if page is not None:
                text = text + page
        return text


class PDFReaderV2:

    def __init__(self, pdf_file):
        self.pdf_file = pdf_file

    #converts pdf, returns its text content as a string
    def get_text(self, pages=None):
        if not pages:
            pagenums = set()
        else:
            pagenums = set(pages)

        output = StringIO()
        manager = PDFResourceManager()
        converter = TextConverter(manager, output, laparams=LAParams())
        interpreter = PDFPageInterpreter(manager, converter)
        infile = open(self.pdf_file, 'rb')
        for page in PDFPage.get_pages(infile, pagenums):
            interpreter.process_page(page)
        infile.close()
        converter.close()
        text = output.getvalue()
        output.close()
        return text