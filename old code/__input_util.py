import textract
import re


class PDFReader:

    def __init__(self, pdf_file):
        self.pdf_file = pdf_file

    def get_all_paragraphs(self):
        text = textract.process(self.pdf_file, method='pdfminer', encoding='ascii')
        splits = re.split('\s{4,}', text)
        paragraphs = []
        for split in splits:
            if len(split) == 0:
                continue
            split = split.replace('\n', '')
            paragraphs.append(split)
        return paragraphs


if __name__ == '__main__':

    reader = PDFReader('/Users/praveentirupattur/PycharmProjects/TextAnalysis/docs/Georgia Tech_2014.pdf')
    print reader.get_all_paragraphs()[0:100]