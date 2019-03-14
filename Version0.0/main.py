from nlp_util import *
from nltk import RegexpTokenizer
from nltk.corpus import stopwords


def initialize_nltk():
    import nltk
    import ssl

    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

    nltk.download()


def custom_method(pdf_file):
    reader = PDFReader(pdf_file)
    #initialize_nltk()
    text_summarizer = TextSummarizer(pdf_file)
    freqTable = text_summarizer.build_freq_table()
    sorted_freqTable = [(k, freqTable[k]) for k in sorted(freqTable, key=freqTable.get, reverse=True)]
    page_count = reader.get_page_count()
    results = {}
    highest_similarity = 0.0
    for i in range(page_count):
        paragraphs = reader.get_paragraphs_from_page(i)
        for paragraph in paragraphs:
            if len(paragraph) > 0:
                summary = text_summarizer.summarize_paragraph(freqTable, paragraph)
                if len(summary) > 0:
                    similarity = text_summarizer.get_similarity('innovation', freqTable, summary)
                    norm_similarity = similarity/len(paragraph)
                    results[paragraph] = norm_similarity
                    if norm_similarity > highest_similarity:
                        matching_paragraph = paragraph
                        highest_similarity = norm_similarity
    sorted_results = [(k, results[k]) for k in sorted(results, key=results.get, reverse=True)]
    for k,v in sorted_results[0:3]:
        print('===============')
        print(k)
        print('===============')

if __name__ == '__main__':
    pdf_file = 'C:/Users/ra407452/Desktop/NSF Eager/Georgia Tech_2014.pdf'
    custom_method(pdf_file)