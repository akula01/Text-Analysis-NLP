from gensim.summarization import summarize
from utils.data_util import *


def summarize_text(text, ratio=0.2):
    summary = summarize(text, ratio=ratio)
    return summary


def test_summarization():
    pdf_file = 'C:/Users/ra407452/OneDrive - Knights - University of Central Florida/Doctoral Research/Information Diffusion/Discussion/link-pred.pdf'
    text = get_text_from_pdf(pdf_file)
    summary = summarize_text(text, ratio=0.01)
    print(summary)


if __name__ == '__main__':
    test_summarization()