# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 01:59:21 2019

@author: Hemil Patel
"""

from text_summarization.abstractive.rnn import RecursiveRNN1
from text_summarization.abstractive.seq2seq import Seq2SeqSummarizer
import numpy as np


def summarize_text(text):

#added some static path as it was giving error to locate some models from model_dir_path
    model_dir_path = './abstractive/models' # refers to the demo/models folder
    config = np.load(r"C:\Users\Hemil Patel\Desktop\TEXT summarize\Text-Analysis-NLP-master\Version0.2\text_summarization\abstractive\models\recursive-rnn-1-config.npy").item()
    summarizer = Seq2SeqSummarizer(config)
    summarizer.load_weights(r"C:\Users\Hemil Patel\Desktop\TEXT summarize\Text-Analysis-NLP-master\Version0.2\text_summarization\abstractive\models\recursive-rnn-1-weights.h5")
    headline = summarizer.summarize(text)
    print(headline)
    rnn_config = np.load(RecursiveRNN1.get_config_file_path(model_dir_path=model_dir_path)).item()
    summarizer = RecursiveRNN1(rnn_config)
    summarizer.load_weights(weight_file_path=RecursiveRNN1.get_weight_file_path(model_dir_path=model_dir_path))
    headline = summarizer.summarize(text)
    return headline


def test_summarization(): 
    text = 'sample_text.txt'
    summary = summarize_text(text)
    print(summary)
#when i run the code it gives ValueError: You are trying to load a weight file containing 5 layers into a model with 4 layers.
#As I just loaded the pre-trained files without training on my laptop as it is not gpu enabled
#The error is due to the inputshape parameter not given at the dense layer in RecursiveRnn1 class.

if __name__ == '__main__':
    test_summarization()