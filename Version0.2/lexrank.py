from sumy.nlp.tokenizers import Tokenizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.summarizers.luhn import LuhnSummarizer
from sumy.summarizers.lsa import LsaSummarizer
from TextAnalysis.utils.data_util import *
import re

pdf_file = 'D:\Project-NSF Eager\Georgia Tech_2014.pdf'


text = get_text_from_pdf(pdf_file)
text_file = open('temp.txt', 'w')
text_file.write(text)
text_file.close()

file = 'temp.txt'
parser = PlaintextParser.from_file(file, Tokenizer("english"))
summarizer = LexRankSummarizer()
summary = summarizer(parser.document, 5) #Summarize the document with 5 sentences

for sentence in summary:
    print(sentence)

print('--------------')

summarizer_1 = LuhnSummarizer()
summary_1 = summarizer_1(parser.document, 5)
for sentence in summary_1:
    print(sentence)

print('--------------')

summarizer_2 = LsaSummarizer()
summary_2 = summarizer_2(parser.document, 5)
for sentence in summary_2:
    print(sentence)