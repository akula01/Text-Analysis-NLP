from utils.pdf_util import *
import re


def get_text_from_pdf(pdf_file):
    reader = PDFReaderV2(pdf_file)
    text = reader.get_text()
    text = text.replace('\n\n', '')
    text = text.replace('\n \n', '')
    text = re.sub(r'[^\x00-\xFF]+', ' ', text)
    text = str(text.encode('utf-8'))
    return text


def test_get_text_from_pdf():
    pdf_file = 'C:/Users/ra407452/Desktop/NSF Eager/TextAnalysis/data/Georgia Tech_2014.pdf'
    print(get_text_from_pdf(pdf_file))


if __name__ == '__main__':
    test_get_text_from_pdf()