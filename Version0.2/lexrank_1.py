from sumy.nlp.tokenizers import Tokenizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.summarizers.luhn import LuhnSummarizer
from sumy.summarizers.lsa import LsaSummarizer
#from utils.data_util import *
import re


file = 'sample_text_1.txt'
parser = PlaintextParser.from_file(file, Tokenizer("english"))
summarizer = LexRankSummarizer()
summary = summarizer(parser.document, 10) #Summarize the document with 10 sentences
#LexRank is an unsupervised approach. It finds the relative importance of all words in a document and selects 
#the sentences which contain the most of those high-scoring words.
for sentence in summary:
    print(sentence)

print('--------------')

summarizer_1 = LuhnSummarizer()
summary_1 = summarizer_1(parser.document, 5)
for sentence in summary_1:
    print(sentence)
#scores sentences based on frequency of the most important words.

print('--------------')

summarizer_2 = LsaSummarizer()
summary_2 = summarizer_2(parser.document, 5)
for sentence in summary_2:
    print(sentence)
#(Latent semantic analysis)