from TextAnalysis.text_summarization.abstractive.keras_text_summarization.library.rnn import RecursiveRNN1
from TextAnalysis.text_summarization.abstractive.keras_text_summarization.library.seq2seq import Seq2SeqSummarizer
import numpy as np
from TextAnalysis.utils.data_util import *


def summarize_text(text):

    model_dir_path = './abstractive/models' # refers to the demo/models folder
    config = np.load(Seq2SeqSummarizer.get_config_file_path(model_dir_path=model_dir_path)).item()
    summarizer = Seq2SeqSummarizer(config)
    summarizer.load_weights(weight_file_path=Seq2SeqSummarizer.get_weight_file_path(model_dir_path=model_dir_path))
    headline = summarizer.summarize(text)
    print(headline)
    rnn_config = np.load(RecursiveRNN1.get_config_file_path(model_dir_path=model_dir_path)).item()
    summarizer = RecursiveRNN1(rnn_config)
    summarizer.load_weights(weight_file_path=RecursiveRNN1.get_weight_file_path(model_dir_path=model_dir_path))
    headline = summarizer.summarize(text)
    return headline


def test_summarization():
    pdf_file = 'D:\Project-NSF Eager\Georgia Tech_2014.pdf'
    text = get_text_from_pdf(pdf_file)
    #text = open('sample_text.txt', 'r').read()
    summary = summarize_text(text)
    print(summary)

if __name__ == '__main__':
    test_summarization()